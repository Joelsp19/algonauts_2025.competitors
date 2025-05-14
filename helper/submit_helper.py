# import all the dependecies from tutorial
import os
import numpy as np
import zipfile
from tqdm.notebook import tqdm
from sklearn.linear_model import Ridge


def load_stimulus_features_friends_s7(root_data_dir):
    """
    Load the stimulus features of all modalities (visual + audio + language) for
    Friends season 7.

    Parameters
    ----------
    root_data_dir : str
        Root data directory.

    Returns
    -------
    features_friends_s7 : dict
        Dictionary containing the stimulus features for Friends season 7.

    """

    features_friends_s7 = {}

    ### Load the visual features ###
    stimuli_dir = os.path.join(root_data_dir, 'stimulus_features', 'pca',
        'friends_movie10', 'visual', 'features_test.npy')
    features_friends_s7['visual'] = np.load(stimuli_dir,
        allow_pickle=True).item()

    ### Load the audio features ###
    stimuli_dir = os.path.join(root_data_dir, 'stimulus_features', 'pca',
        'friends_movie10', 'audio', 'features_test.npy')
    features_friends_s7['audio'] = np.load(stimuli_dir,
        allow_pickle=True).item()

    ### Load the language features ###
    stimuli_dir = os.path.join(root_data_dir, 'stimulus_features', 'pca',
        'friends_movie10', 'language', 'features_test.npy')
    features_friends_s7['language'] = np.load(stimuli_dir,
        allow_pickle=True).item()

    ### Output ###
    return features_friends_s7

def align_features_and_fmri_samples_friends_s7(features_friends_s7,
    root_data_dir):
    """
    Align the stimulus feature with the fMRI response samples for Friends season
    7 episodes, later used to predict the fMRI responses for challenge
    submission.

    Parameters
    ----------
    features_friends_s7 : dict
        Dictionary containing the stimulus features for Friends season 7.
    root_data_dir : str
        Root data directory.

    Returns
    -------
    aligned_features_friends_s7 : dict
        Aligned stimulus features for each subject and Friends season 7 episode.

    """

    ### Empty results dictionary ###
    aligned_features_friends_s7 = {}

    ### HRF delay ###
    # fMRI detects the BOLD (Blood Oxygen Level Dependent) response, a signal
    # that reflects changes in blood oxygenation levels in response to activity
    # in the brain. Blood flow increases to a given brain region in response to
    # its activity. This vascular response, which follows the hemodynamic
    # response function (HRF), takes time. Typically, the HRF peaks around 5–6
    # seconds after a neural event: this delay reflects the time needed for
    # blood oxygenation changes to propagate and for the fMRI signal to capture
    # them. Therefore, this parameter introduces a delay between stimulus chunks
    # and fMRI samples for a better correspondence between input stimuli and the
    # brain response. For example, with a hrf_delay of 3, if the stimulus chunk
    # of interest is 17, the corresponding fMRI sample will be 20.
    hrf_delay = 3

    ### Stimulus window ###
    # stimulus_window indicates how many stimulus feature samples are used to
    # model each fMRI sample, starting from the stimulus sample corresponding to
    # the fMRI sample of interest, minus the hrf_delay, and going back in time.
    # For example, with a stimulus_window of 5, and a hrf_delay of 3, if the
    # fMRI sample of interest is 20, it will be modeled with stimulus samples
    # [13, 14, 15, 16, 17]. Note that this only applies to visual and audio
    # features, since the language features were already extracted using
    # transcript words spanning several movie samples (thus, each fMRI sample
    # will only be modeled using the corresponding language feature sample,
    # minus the hrf_delay). Also note that a larger stimulus window will
    # increase compute time, since it increases the amount of stimulus features
    # used to train and validate the fMRI encoding models. Here you will use a
    # value of 5, since this is how the challenge baseline encoding models were
    # trained.
    stimulus_window = 5

    ### Loop over subjects ###
    subjects = [1, 2, 3, 5]
    desc = "Aligning stimulus and fMRI features of the four subjects"
    for sub in tqdm(subjects, desc=desc):
        aligned_features_friends_s7[f'sub-0{sub}'] = {}

        ### Load the Friends season 7 fMRI samples ###
        samples_dir = os.path.join(root_data_dir, 'algonauts_2025.competitors',
            'fmri', f'sub-0{sub}', 'target_sample_number',
            f'sub-0{sub}_friends-s7_fmri_samples.npy')
        fmri_samples = np.load(samples_dir, allow_pickle=True).item()

        ### Loop over Friends season 7 episodes ###
        for epi, samples in fmri_samples.items():
            features_epi = []

            ### Loop over fMRI samples ###
            for s in range(samples):
                # Empty variable containing the stimulus features of all
                # modalities for each sample
                f_all = np.empty(0)

                ### Loop across modalities ###
                for mod in features_friends_s7.keys():

                    ### Visual and audio features ###
                    # If visual or audio modality, model each fMRI sample using
                    # the N stimulus feature samples up to the fMRI sample of
                    # interest minus the hrf_delay (where N is defined by the
                    # 'stimulus_window' variable)
                    if mod == 'visual' or mod == 'audio':
                        # In case there are not N stimulus feature samples up to
                        # the fMRI sample of interest minus the hrf_delay (where
                        # N is defined by the 'stimulus_window' variable), model
                        # the fMRI sample using the first N stimulus feature
                        # samples
                        if s < (stimulus_window + hrf_delay):
                            idx_start = 0
                            idx_end = idx_start + stimulus_window
                        else:
                            idx_start = s - hrf_delay - stimulus_window + 1
                            idx_end = idx_start + stimulus_window
                        # In case there are less visual/audio feature samples
                        # than fMRI samples minus the hrf_delay, use the last N
                        # visual/audio feature samples available (where N is
                        # defined by the 'stimulus_window' variable)
                        if idx_end > len(features_friends_s7[mod][epi]):
                            idx_end = len(features_friends_s7[mod][epi])
                            idx_start = idx_end - stimulus_window
                        f = features_friends_s7[mod][epi][idx_start:idx_end]
                        f_all = np.append(f_all, f.flatten())

                    ### Language features ###
                    # Since language features already consist of embeddings
                    # spanning several samples, only model each fMRI sample
                    # using the corresponding stimulus feature sample minus the
                    # hrf_delay
                    elif mod == 'language':
                        # In case there are no language features for the fMRI
                        # sample of interest minus the hrf_delay, model the fMRI
                        # sample using the first language feature sample
                        if s < hrf_delay:
                            idx = 0
                        else:
                            idx = s - hrf_delay
                        # In case there are fewer language feature samples than
                        # fMRI samples minus the hrf_delay, use the last
                        # language feature sample available
                        if idx >= (len(features_friends_s7[mod][epi]) - hrf_delay):
                            f = features_friends_s7[mod][epi][-1,:]
                        else:
                            f = features_friends_s7[mod][epi][idx]
                        f_all = np.append(f_all, f.flatten())

                ### Append the stimulus features of all modalities for this sample ###
                features_epi.append(f_all)

            ### Add the episode stimulus features to the features dictionary ###
            aligned_features_friends_s7[f'sub-0{sub}'][epi] = np.asarray(
                features_epi, dtype=np.float32)

    return aligned_features_friends_s7

def load_baseline_encoding_models(root_data_dir):
    """
    Load the challenge baseline encoding models for all four challenge subject.
    These models were trained to predict fMRI responses to movies using all
    stimulus modalities (visual + audio + language)

    Parameters
    ----------
    root_data_dir : str
        Root data directory.

    Returns
    -------
    baseline_models : dict
        Pretrained challenge baseline RidgeCV models.

    """

    baseline_models = {}

    ### Loop over subjects ###
    subjects = [1, 2, 3, 5]
    for s in subjects:

        ### Load the trained encoding model weights ###
        weights_dir = os.path.join(root_data_dir, 'trained_encoding_models',
            'trained_encoding_model_sub-0'+str(s)+'_modality-all.npy')
        model_weights = np.load(weights_dir, allow_pickle=True).item()

        ### Initialize the Ridge regression and load the trained weights ###
        model = Ridge()
        model.coef_ = model_weights['coef_']
        model.intercept_ = model_weights['intercept_']
        model.n_features_in_ = model_weights['n_features_in_']

        ### Store the pretrained encoding model into a dictionary ###
        baseline_models['sub-0'+str(s)] = model
        del model

    ### Output ###
    return baseline_models


def predict_fmri_responses_friends_s7(aligned_features_friends_s7, baseline_models):
    # Empty submission predictions dictionary
    submission_predictions = {}

    # Loop through each subject
    desc = "Predicting fMRI responses of each subject"
    for sub, features in tqdm(aligned_features_friends_s7.items(), desc=desc):

        # Initialize the nested dictionary for each subject's predictions
        submission_predictions[sub] = {}

        # Loop through each Friends season 7 episode
        for epi, feat_epi in features.items():

            # Predict fMRI responses for the aligned features of this episode, and
            # convert the predictions to float32
            fmri_pred = baseline_models[sub].predict(feat_epi).astype(np.float32)

            # Store formatted predictions in the nested dictionary
            submission_predictions[sub][epi] = fmri_pred

    return submission_predictions

def save_predictions_to_npy(submission_predictions, save_dir, root_data_dir):

    # Save the predicted fMRI dictionary as a .npy file
    output_file = save_dir + "fmri_predictions_friends_s7.npy"
    np.save(output_file, submission_predictions)
    print(f"Formatted predictions saved to: {output_file}")

    # Zip the saved file for submission
    zip_file = save_dir + "fmri_predictions_friends_s7.zip"
    with zipfile.ZipFile(zip_file, 'w') as zipf:
        zipf.write(output_file, os.path.basename(output_file))
    print(f"Submission file successfully zipped as: {zip_file}")