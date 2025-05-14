# import all the dependecies from tutorial
import os
import numpy as np
import h5py
import torch
import librosa
from tqdm.notebook import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from moviepy.editor import VideoFileClip
from transformers import BertTokenizer, BertModel
from torchvision.transforms import Compose, Lambda, CenterCrop
from torchvision.models.feature_extraction import create_feature_extractor
from pytorchvideo.transforms import Normalize, UniformTemporalSubsample, ShortSideScale


def define_frames_transform():
    """Defines the preprocessing pipeline for the video frames. Note that this
    transform is specific to the slow_r50 model."""
    transform = Compose(
        [
            UniformTemporalSubsample(8),
            Lambda(lambda x: x/255.0),
            Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
            ShortSideScale(size=256),
            CenterCrop(256)
        ]
  )
    return transform


def get_vision_model(device):
    """
    Load a pre-trained slow_r50 video model and set up the feature extractor.

    Parameters
    ----------
    device : torch.device
        The device on which the model will run (i.e., 'cpu' or 'cuda').

    Returns
    -------
    feature_extractor : torch.nn.Module
        The feature extractor model.
    model_layer : str
        The layer from which visual features will be extracted.

    """

    # Load the model
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50',
        pretrained=True)

    # Select 'blocks.5.pool' as the feature extractor layer
    model_layer = 'blocks.5.pool'
    feature_extractor = create_feature_extractor(model,
        return_nodes=[model_layer])
    feature_extractor.to(device)
    feature_extractor.eval()

    return feature_extractor, model_layer


def extract_visual_features(episode_path, tr, feature_extractor, model_layer,
    transform, device, save_dir_temp, save_dir_features):
    """
    Extract visual features from a movie using a pre-trained video model.

    Parameters
    ----------
    episode_path : str
        Path to the movie file for which the visual features are extracted.
    tr : float
        Duration of each chunk, in seconds (aligned with the fMRI repetition
        time, or TR).
    feature_extractor : torch.nn.Module
        Pre-trained feature extractor model.
    model_layer : str
        The model layer from which the visual features are extracted.
    transform : torchvision.transforms.Compose
        Transformation pipeline for processing video frames.
    device : torch.device
        Device for computation ('cpu' or 'cuda').
    save_dir_temp : str
        Directory where the chunked movie clips are temporarily stored for
        feature extraction.
    save_dir_features : str
        Directory where the extracted visual features are saved.

    Returns
    -------
    visual_features : float
        Array containing the extracted visual features.

    """

    # Get the onset time of each movie chunk
    clip = VideoFileClip(episode_path)
    start_times = [x for x in np.arange(0, clip.duration, tr)][:-1]
    # Create the directory where the movie chunks are temporarily saved
    temp_dir = os.path.join(save_dir_temp, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    # Empty features list
    visual_features = []

    # Loop over chunks
    with tqdm(total=len(start_times), desc="Extracting visual features") as pbar:
        for start in start_times:

            # Divide the movie in chunks of length TR, and save the resulting
            # clips as '.mp4' files
            clip_chunk = clip.subclip(start, start+tr)
            chunk_path = os.path.join(temp_dir, 'visual_chunk.mp4')
            clip_chunk.write_videofile(chunk_path, verbose=False, audio=False,
                logger=None)
            # Load the frames from the chunked movie clip
            video_clip = VideoFileClip(chunk_path)
            chunk_frames = [frame for frame in video_clip.iter_frames()]

            # Format the frames to shape:
            # (batch_size, channels, num_frames, height, width)
            frames_array = np.transpose(np.array(chunk_frames), (3, 0, 1, 2))
            # Convert the video frames to tensor
            inputs = torch.from_numpy(frames_array).float()
            # Preprocess the video frames
            inputs = transform(inputs).unsqueeze(0).to(device)

            # Extract the visual features
            with torch.no_grad():
                preds = feature_extractor(inputs)
            visual_features.append(np.reshape(preds[model_layer].cpu().numpy(), -1))

            # Update the progress bar
            pbar.update(1)

    # Convert the visual features to float32
    visual_features = np.array(visual_features, dtype='float32')

    # Save the visual features
    #out_file_visual = os.path.join(
    #    save_dir_features, f'friends_s01e01a_features_visual.h5')
    #with h5py.File(out_file_visual, 'a' if Path(out_file_visual).exists() else 'w') as f:
    #    group = f.create_group("s01e01a")
    #    group.create_dataset('visual', data=visual_features, dtype=np.float32)
    #print(f"Visual features saved to {out_file_visual}")

    # Output
    return visual_features


def extract_audio_features(episode_path, tr, sr, device, save_dir_temp,
    save_dir_features):
    """
    Extract audio features from a movie using Mel-frequency cepstral
    coefficients (MFCCs).

    Parameters
    ----------
    episode_path : str
        Path to the movie file for which the audio features are extracted.
    tr : float
        Duration of each chunk, in seconds (aligned with the fMRI repetition
        time, or TR).
    sr : int
        Audio sampling rate.
    device : str
        Device to perform computations ('cpu' or 'gpu').
    save_dir_temp : str
        Directory where the chunked movie clips are temporarily stored for
        feature extraction.
    save_dir_features : str
        Directory where the extracted audio features are saved.

    Returns
    -------
    audio_features : float
        Array containing the extracted audio features.

    """

    # Get the onset time of each movie chunk
    clip = VideoFileClip(episode_path)
    start_times = [x for x in np.arange(0, clip.duration, tr)][:-1]
    # Create the directory where the movie chunks are temporarily saved
    temp_dir = os.path.join(save_dir_temp, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    # Empty features list
    audio_features = []

    ### Loop over chunks ###
    with tqdm(total=len(start_times), desc="Extracting audio features") as pbar:
        for start in start_times:

            # Divide the movie in chunks of length TR, and save the resulting
            # audio clips as '.wav' files
            clip_chunk = clip.subclip(start, start+tr)
            chunk_audio_path = os.path.join(temp_dir, 'audio_s01e01a.wav')
            clip_chunk.audio.write_audiofile(chunk_audio_path, verbose=False,
                logger=None)
            # Load the audio samples from the chunked movie clip
            y, sr = librosa.load(chunk_audio_path, sr=sr, mono=True)

            # Extract the audio features (MFCC)
            mfcc_features = np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1)
            audio_features.append(mfcc_features)
            # Update the progress bar
            pbar.update(1)

    ### Convert the visual features to float32 ###
    audio_features = np.array(audio_features, dtype='float32')

    # Save the audio features
    #out_file_audio = os.path.join(
    #    save_dir_features, f'friends_s01e01a_features_audio.h5')
    #with h5py.File(out_file_audio, 'a' if Path(out_file_audio).exists() else 'w') as f:
    #    group = f.create_group("s01e01a")
    #    group.create_dataset('audio', data=audio_features, dtype=np.float32)
    #print(f"Audio features saved to {out_file_audio}")

    ### Output ###
    return audio_features


def get_language_model(device):
    """
    Load a pre-trained bert-base-uncased language model and its corresponding
    tokenizer.

    Parameters
    ----------
    device : torch.device
        Device on which the model will run (e.g., 'cpu' or 'cuda').

    Returns
    -------
    model : object
        Pre-trained language model.
    tokenizer : object
        Tokenizer corresponding to the language model.

    """

    ### Load the model ###
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval().to(device)

    ### Load the tokenizer ###
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
        do_lower_case=True)

    ### Output ###
    return model, tokenizer

def load_features(root_data_dir, modality):
    """
    Load the extracted features from the HDF5 file.

    Parameters
    ----------
    root_data_dir : str
        Root data directory.
    modality : str
        The modality of the features ('visual', 'audio', or 'language').

    Returns
    -------
    features : float
        Stimulus features.

    """

    ### Get the stimulus features file directory ###
    data_dir = os.path.join(root_data_dir, 'stimulus_features', 'raw', modality,
        'friends_s01e01a_features_'+modality+'.h5')

    ### Load the stimulus features ###
    with h5py.File(data_dir, 'r') as data:
        for episode in data.keys():
            if modality != 'language':
                features = np.asarray(data[episode][modality])
            else:
                # Vectorize and append pooler_output and last_hidden_state
                # language features
                pooler_output = np.asarray(
                    data[episode][modality+'_pooler_output'])
                last_hidden = np.asarray(np.reshape(
                    data[episode][modality+'_last_hidden_state'],
                    (len(pooler_output), -1)))
                features = np.append(pooler_output, last_hidden, axis=1)
    print(f"{modality} features original shape: {features.shape}")
    print('(Movie samples × Features)')

    ### Output ###
    return features

def preprocess_features(features):
    """
    Rplaces NaN values in the stimulus features with zeros, and z-score the
    features.

    Parameters
    ----------
    features : float
        Stimulus features.

    Returns
    -------
    prepr_features : float
        Preprocessed stimulus features.

    """

    ### Convert NaN values to zeros ###
    features = np.nan_to_num(features)

    ### Z-score the features ###
    scaler = StandardScaler()
    prepr_features = scaler.fit_transform(features)

    ### Output ###
    return prepr_features

def perform_pca(prepr_features, n_components, modality):
    """
    Perform PCA on the standardized features.

    Parameters
    ----------
    prepr_features : float
        Preprocessed stimulus features.
    n_components : int
        Number of components to keep

    Returns
    -------
    features_pca : float
        PCA-downsampled stimulus features.

    """

    ### Set the number of principal components to keep ###
    # If number of PCs is larger than the number of features, set the PC number
    # to the number of features
    if n_components > prepr_features.shape[1]:
        n_components = prepr_features.shape[1]

    ### Perform PCA ###n_init=4, max_iter=300
    pca = PCA(n_components, random_state=20200220)
    features_pca = pca.fit_transform(prepr_features)
    print(f"\n{modality} features PCA shape: {features_pca.shape}")
    print('(Movie samples × Principal components)')

    ### Output ###
    return features_pca