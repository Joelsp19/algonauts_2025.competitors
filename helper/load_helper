# import all the dependecies from tutorial
import os
import numpy as np
import pandas as pd
import h5py
import cv2
import nibabel as nib
from nilearn import plotting
import ipywidgets as widgets
from ipywidgets import VBox, Dropdown, Button
from IPython.display import Video, display, clear_output
from moviepy.editor import VideoFileClip


def load_mkv_file(movie_path):
    """
    Load video and audio data from the given .mkv movie file, and additionally
    prints related information.

    Parameters
    ----------
    movie_path : str
        Path to the .mkv movie file.

    """

    # Read the .mkv file
    cap = cv2.VideoCapture(movie_path)

    if not cap.isOpened():
        print("Error: Could not open movie.")
        return

    # Get video information
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = video_total_frames / video_fps
    video_duration_minutes = video_duration / 60

    # Print video information
    print(">>> Video Information <<<")
    print(f"Video FPS: {video_fps}")
    print(f"Video Resolution: {video_width}x{video_height}")
    print(f"Total Frames: {video_total_frames}")
    print(f"Video Duration: {video_duration:.2f} seconds or {video_duration_minutes:.2f} minutes")

    # Release the video object
    cap.release()

    # Audio information
    clip = VideoFileClip(movie_path)
    audio = clip.audio
    audio_duration = audio.duration
    audio_fps = audio.fps
    print("\n>>> Audio Information <<<")
    print(f"Audio Duration: {audio_duration:.2f} seconds")
    print(f"Audio FPS (Sample Rate): {audio_fps} Hz")

    # Extract and display the first 20 seconds of the video
    output_video_path = 'first_20_seconds.mp4'
    video_segment = clip.subclip(0, min(20, video_duration))
    print("\nCreating clip of the first 20 seconds of the video...")
    video_segment.write_videofile(output_video_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)

    # Display the video in the notebook
    display(Video(output_video_path, embed=True, width=640, height=480))

def load_tsv_file(transcript_path):
    """
    Load and visualize language transcript data from the given .TSV file.

    Parameters
    ----------
    transcript_path : str
        Path to the .tsv transcript file.

    """

    # Load the .tsv into a pandas DataFrame
    transcript_df = pd.read_csv(transcript_path, sep='\t')

    # Select the first 20 rows (chunks)
    sample_transcript_data = transcript_df.iloc[:20]

    # Display the first 20 rows (chunks)
    # The first 11 rows are empty since no words were spoken during the
    # beginning of the episode.
    print("Transcript data (Rows 0 to 20):")
    display(sample_transcript_data)

    # Print other transcript info
    print(f"\nTranscript has {transcript_df.shape[0]} rows (chunks of 1.49 seconds) and {transcript_df.shape[1]} columns.")

def load_transcript(transcript_path):
    """
    Loads a transcript file and returns it as a DataFrame.

    Parameters
    ----------
    transcript_path : str
        Path to the .tsv transcript file.

    """
    df = pd.read_csv(transcript_path, sep='\t')
    return df


def get_movie_info(movie_path):
    """
    Extracts the frame rate (FPS) and total duration of a movie.

    Parameters
    ----------
    movie_path : str
        Path to the .mkv movie file.

    """

    cap = cv2.VideoCapture(movie_path)
    fps, frame_count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    return fps, frame_count / fps


def split_movie_into_chunks(movie_path, chunk_duration=1.49):
    """
    Divides a video into fixed-duration chunks.

    Parameters
    ----------
    movie_path : str
        Path to the .mkv movie file.
    chunk_duration : float, optional
        Duration of each chunk in seconds (default is 1.49).

    """

    _, video_duration = get_movie_info(movie_path)
    chunks = []
    start_time = 0.0

    # Create chunks for the specified time
    while start_time < video_duration:
        end_time = min(start_time + chunk_duration, video_duration)
        chunks.append((start_time, end_time))
        start_time += chunk_duration
    return chunks

def extract_movie_segment_with_sound(movie_path, start_time, end_time,
    output_path='output_segment.mp4'):
    """
    Extracts a specific segment of a video with sound and saves it.

    Parameters
    ----------
    movie_path : str
        Path to the .mkv movie file.
    start_time : float
        Start time of the segment in seconds.
    end_time : float
        End time of the segment in seconds.
    output_path : str, optional
        Path to save the output segment (default is 'output_segment.mp4').

    """

    # Create movie segment
    movie_segment = VideoFileClip(movie_path).subclip(start_time, end_time)
    print(f"\nWriting movie file from {start_time}s until {end_time}s")

    # Write video file
    movie_segment.write_videofile(output_path, codec="libx264",
        audio_codec="aac", verbose=False, logger=None)
    return output_path


def display_transcript_and_movie(chunk_index, transcript_df, chunks,
    movie_path):
    """
    Displays transcript, movie, onset, and duration for a selected chunk.

    Parameters
    ----------
    chunk_index : int
        Index of the selected chunk.
    transcript_df : DataFrame
        DataFrame containing transcript data.
    chunks : list
        List of (start_time, end_time) tuples for video chunks.
    movie_path : str
        Path to the .mkv movie file.

    """
    # Retrieve the start and end times for the selected chunk
    start_time, end_time = chunks[chunk_index]

    # Get the corresponding transcript row if it exists in the DataFrame
    transcript_chunk = transcript_df.iloc[chunk_index] if chunk_index < len(transcript_df) else None

    # Display the stimulus chunk number
    print(f"\nChunk number: {chunk_index + 1}")

    # Display transcript details if available; otherwise, indicate no dialogue
    if transcript_chunk is not None and pd.notna(transcript_chunk['text_per_tr']):
        print(f"\nText: {transcript_chunk['text_per_tr']}")
        print(f"Words: {transcript_chunk['words_per_tr']}")
        print(f"Onsets: {transcript_chunk.get('onsets_per_tr', 'N/A')}")
        print(f"Durations: {transcript_chunk.get('durations_per_tr', 'N/A')}")
    else:
        print("<No dialogue in this scene>")

    # Extract and display the video segment
    output_movie_path = extract_movie_segment_with_sound(movie_path, start_time,
        end_time)
    display(Video(output_movie_path, embed=True, width=640, height=480))


def create_dropdown_by_text(transcript_df):
    """
    Creates a dropdown widget for selecting chunks by their text.

    Parameters
    ----------
    transcript_df : DataFrame
        DataFrame containing transcript data.

    """

    options = []

    # Iterate over each row in the transcript DataFrame
    for i, row in transcript_df.iterrows():
        if pd.notna(row['text_per_tr']):  # Check if the transcript text is not NaN
            options.append((row['text_per_tr'], i))
        else:
            options.append(("<No dialogue in this scene>", i))
    return widgets.Dropdown(options=options, description='Select scene:')


def interface_display_transcript_and_movie(movie_path, transcript_path):
    """
    Interactive interface to align movie and transcript chunks.

    Parameters
    ----------
    movie_path : str
        Path to the .mkv movie file.
    transcript_path : str
        Path to the transcript file (.tsv).

    """

    # Load the transcript data from the provided path
    transcript_df = load_transcript(transcript_path)

    # Split the video file into chunks of 1.49 seconds
    chunks = split_movie_into_chunks(movie_path)

    # Create a dropdown widget with transcript text as options
    dropdown = create_dropdown_by_text(transcript_df)

    # Create an output widget to display video and transcript details
    output = widgets.Output()

    # Display the dropdown and output widgets
    display(dropdown, output)

    # Define the function to handle dropdown value changes
    def on_chunk_select(change):
        with output:
            output.clear_output()  # Clears previous content
            chunk_index = dropdown.value
            display_transcript_and_movie(chunk_index, transcript_df, chunks,
                movie_path)

    dropdown.observe(on_chunk_select, names='value')

# Function to list available subjects based on folder names
def list_subjects(fmri_dir):
    return sorted([d for d in os.listdir(fmri_dir) if d.startswith('sub-')])

# Function to explore HDF5 file structure and organize datasets by season/movie
def explore_h5_file(file_path, selected_dataset):
    season_movie_dict = {}
    with h5py.File(file_path, 'r') as h5_file:
        for name, obj in h5_file.items():
            if isinstance(obj, h5py.Dataset):
                if selected_dataset == 'Friends':
                    season_movie = name.split('_')[1].split('-')[1][:3]  # Extract season (e.g., 's01')
                elif selected_dataset == 'Movie10':
                    season_movie = name.split('_')[1].split('-')[1][:-2]  # Extract movie (e.g., 'bourne')
                season_movie_dict.setdefault(season_movie, []).append(f"{name} (Shape: {obj.shape})")
    return season_movie_dict

# Function to display datasets in a DataFrame
def display_datasets_in_table(season_dict):
    max_len = max(len(v) for v in season_dict.values())
    df = pd.DataFrame({k: v + [''] * (max_len - len(v)) for k, v in sorted(season_dict.items())})
    display(df)

# Create subject and dataset selector widget
def create_subject_selector(fmri_dir):
    subjects = list_subjects(fmri_dir)
    dataset_options = ['Friends', 'Movie10']

    subject_dropdown = Dropdown(options=subjects, description='Select Subject:')
    dataset_dropdown = Dropdown(options=dataset_options, description='Select Dataset:')
    button = Button(description="Explore File", button_style='primary')

    def on_button_click(b):
        clear_output(wait=True)
        display(VBox([subject_dropdown, dataset_dropdown, button]))

        selected_subject = subject_dropdown.value
        selected_dataset = dataset_dropdown.value

        if selected_dataset == 'Friends':
            h5_file_path = os.path.join(
                fmri_dir, selected_subject, 'func',
                f"{selected_subject}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5"
            )
        elif selected_dataset == 'Movie10':
            h5_file_path = os.path.join(
                fmri_dir, selected_subject, 'func',
                f"{selected_subject}_task-movie10_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_bold.h5"
            )

        if os.path.exists(h5_file_path):
            season_movie_data = explore_h5_file(h5_file_path, selected_dataset)
            display_datasets_in_table(season_movie_data)
        else:
            print("Error: HDF5 file not found.")

    button.on_click(on_button_click)
    display(VBox([subject_dropdown, dataset_dropdown, button]))


def plot_fmri_on_brain(chunk_index, fmri_file_path, atlas_path, dataset_name,
    hrf_delay):
    """
    Map fMRI responses to brain parcels and plot it on a glass brain.

    Parameters
    ----------
    chunk_index : pandas.Series
        The selected chunk from the transcript, used to determine the fMRI
        sample.
    fmri_file_path : str
        Path to the HDF5 file containing fMRI data.
    atlas_path : str
        Path to the atlas NIfTI file.
    dataset_name : str
        Name of the dataset inside the HDF5 file.
    hrf_delay : int
        fMRI detects the BOLD (Blood Oxygen Level Dependent) response, a signal
        that reflects changes in blood oxygenation levels in response to
        activity in the brain. Blood flow increases to a given brain region in
        response to its activity. This vascular response, which follows the
        hemodynamic response function (HRF), takes time. Typically, the HRF
        peaks around 5–6 seconds after a neural event: this delay reflects the
        time needed for blood oxygenation changes to propagate and for the fMRI
        signal to capture them. Therefore, this parameter introduces a delay
        between stimulus chunks and fMRI samples for a better correspondence
        between input stimuli and the brain response. For example, with a
        hrf_delay of 3, if the stimulus chunk of interest is 17, the
        corresponding fMRI sample will be 20.

    """

    print(f"\nLoading fMRI file: {fmri_file_path}")

    # Load the atlas image
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata()

    # Open the fMRI reeponses file, and extract the specific dataset
    with h5py.File(fmri_file_path, 'r') as f:
        print(f"Opening fMRI dataset: {dataset_name}")
        fmri_data = f[dataset_name][()]
        print(f"fMRI dataset shape: {fmri_data.shape}")

    # Extract the corresponding sample from the fMRI responses based on the
    # selected transcript chunk, and on the hrf_delay
    if (chunk_index + hrf_delay) > len(fmri_data):
        selected_sample = len(fmri_data)
    else:
        selected_sample = chunk_index + hrf_delay
    fmri_sample_data = fmri_data[selected_sample]
    print(f"Extracting fMRI sample {selected_sample+1}.")

    # Map fMRI sample values to the brain parcels in the atlas
    output_data = np.zeros_like(atlas_data)
    for parcel_index in range(1000):
        output_data[atlas_data == (parcel_index + 1)] = \
            fmri_sample_data[parcel_index]

    # Create the output NIfTI image
    output_img = nib.Nifti1Image(output_data, affine=atlas_img.affine)

    # Plot the glass brain with the mapped fMRI data
    display = plotting.plot_glass_brain(
        output_img,
        display_mode='lyrz',
        cmap='inferno',
        colorbar=True,
        plot_abs=False)
    colorbar = display._cbar
    colorbar.set_label("fMRI activity", rotation=90, labelpad=12, fontsize=12)
    plotting.show()


    # Main interactive interface with brain visualization

def interface_display_transcript_movie_brain(movie_path, transcript_path,
                                             

    fmri_file_path, atlas_path, dataset_name, hrf_delay):
    """
    Interactive interface to display movie and transcripts chunks along with
    the fMRI response from the corresponding sample.

    This code uses functions from Section 1.2.3.

    Parameters
    ----------
    movie_path : str
        Path to the .mkv movie file.
    transcript_path : str
        Path to the .tsv transcript file.
    fmri_file_path : str
        Path to the fMRI data file.
    atlas_path : str
        Path to the brain atlas file.
    dataset_name : str
        Name of the dataset to display fMRI data from.
    hrf_delay : int
        fMRI detects the BOLD (Blood Oxygen Level Dependent) response, a signal
        that reflects changes in blood oxygenation levels in response to
        activity in the brain. Blood flow increases to a given brain region in
        response its activity. This vascular response, which follows the
        hemodynamic response function (HRF), takes time. Typically, the HRF
        peaks around 5–6 seconds after a neural event: this delay reflects the
        time needed for blood oxygenation changes to propagate and for the fMRI
        signal to capture them. Therefore, this parameter introduces a delay
        between stimulus chunks and fMRI samples. For example, with a hrf_delay
        of 3, if the stimulus chunk of interest is 17, the corresponding fMRI
        sample will be 20.

    """

    # Load the .tsv transcript data from the provided path
    transcript_df = load_transcript(transcript_path)  # from 1.2.3

    # Split the .mkv movie file into chunks of 1.49 seconds
    chunks = split_movie_into_chunks(movie_path)  # from 1.2.3

    # Create a dropdown widget with transcript text as options
    dropdown = create_dropdown_by_text(transcript_df)  # from 1.2.3

    # Create an output widget to display video, transcript, and brain
    # visualization
    output = widgets.Output()

    # Define the function to handle dropdown value changes
    def on_chunk_select(change):
        with output:
            output.clear_output()  # Clear the previous output
            chunk_index = dropdown.value

            # Display video chunk and transcript
            display_transcript_and_movie(chunk_index, transcript_df, chunks,
                movie_path)  # from 1.2.3

            # Visualize brain fMRI data
            plot_fmri_on_brain(chunk_index, fmri_file_path, atlas_path,
                dataset_name, hrf_delay)

    dropdown.observe(on_chunk_select, names='value')
    display(dropdown, output)




