import logging
from typing import Dict
import streamlit as st
import numpy as np
import mne
from src.utils.nil_sleep_analysis import analyze_uncertain_periods
import matplotlib.pyplot as plt

# ''' This page allows users to rescore uncertain periods in sleep data.

# TODO:
# - Minor. Improve caching strategy for processing data to improve code readability.

# '''

SLEEP_STAGE_LABELS = ["Wake", "REM", "N1", "N2", "N3"] # Sleep stage labels.
CACHE_PATH = "./cache/"

def main():
    # Set the title of the Streamlit app.
    st.title("Rescoring Uncertain Periods")
    st.write("This page allows users to rescore uncertain periods in sleep data.")
    # Check if the session state is initialized. Likely caused by a refresh.
    if "initialized" not in st.session_state:
        logging.error("Session state not initialized.")
        st.error("Refresh detected. Please go back to the homepage.")
        return
    # Check if user has downloaded a dataset.
    subject_id_choice = st.session_state["subject_id"]
    subject_id_download = st.session_state["dataset_downloaded"]["subject_id"]
    if subject_id_choice is None:
        st.warning("Please select a subject from the homepage to rescore uncertain periods.")
        logging.warning("No subject ID selected for rescoring.")
        return
    elif subject_id_choice != subject_id_download:
        st.warning(f"Please go back to homepage and allow for download to complete.")
        logging.warning(f"Subject ID {subject_id_choice} does not match downloaded subject ID {subject_id_download}.")
        return
    # Get pointers from session state.
    dataset_downloaded = st.session_state["dataset_downloaded"]
    dataset_processed = st.session_state["dataset_processed"]
    dataset_rescored = st.session_state["dataset_rescored"]
    current_epoch = st.session_state["current_epoch"]
    fig_config = st.session_state["fig_config"]
    # Handle loading of new subject.
    if dataset_processed["subject_id"] != subject_id_download: # Handle new subject.
        ## Process downloaded data.
        logging.info(f"Processing data for subject {subject_id_download}.")
        # Process for uncertain periods.
        dataset_processed["scoring"] = process_scoring_data(subject_id_download)
        # Set the current epoch to be the first uncertain period, if any.
        current_epoch = 0
        if dataset_processed["scoring"]["n_uncertain_periods"] == 0:
            logging.info("No uncertain periods found.")
            st.warning("No uncertain periods found.")
        else:
            first_uncertain_hour = dataset_processed["scoring"]["uncertain_periods"][0]["start_hour"]
            first_uncertain_epoch = int(first_uncertain_hour * 3600 / 30)  # Convert hours to epochs.
            current_epoch = first_uncertain_epoch
        # Load a copy of the raw data for the current epoch
        dataset_processed["processed_biosignals"] = process_biosignals(current_epoch, subject_id_download)
        # Set name for processed data held in session state.
        dataset_processed["subject_id"] = subject_id_download
        ## Initialize rescoring data structure from processed data.
        logging.info(f"Initializing rescoring for subject {dataset_processed['subject_id']}")
        scoring_array = dataset_processed["scoring"]["scoring_naive"]
        dataset_rescored["scoring_manual"] = scoring_array.copy()
        dataset_rescored["scoring_manual_mask"] = np.zeros_like(scoring_array, dtype=bool)
        dataset_rescored["subject_id"] = dataset_processed["subject_id"]
        ## Update figure configuration.
        fig_config["subject_id"] = dataset_processed["subject_id"]
        create_figures()
    # Populate UI elements.
    st.write(f"Currently rescoring uncertain periods for subject: {dataset_processed['subject_id']}")
    st.image(fig_config["svg_paths"]["scoring"])



@st.cache_data
def process_scoring_data(subject_id: str) -> Dict:
    """Process scoring data to extract uncertain periods."""
    # Get pointer to scoring data from session state
    confidence_data = st.session_state["dataset_downloaded"]["scoring"]
    # Analyze uncertain periods in the scoring data
    scoring_processed = analyze_uncertain_periods(confidence_data)
    return scoring_processed

@st.cache_data
def process_biosignals(n_epoch: int, subject_id: str) -> Dict:
    """Process biosignals for visualization."""
    # Get pointer to raw object
    raw_obj = st.session_state["dataset_downloaded"]["raw_obj"]
    # Get the sampling frequency from the raw data
    fs = raw_obj.info["sfreq"]
    # Get the time slice for the current epoch
    time_start = n_epoch * 30
    time_stop = (n_epoch + 1) * 30
    raw_selection = raw_obj.copy().crop(tmin=time_start, tmax=time_stop)
    # Create a dictionary to store the processed data
    processed_data = {
        "raw_obj_cropped": raw_selection,
        "fs": fs,
    }
    return processed_data

def create_figures():
    """Create figures for visualization."""
    # Get pointers fr information inside session state.
    current_epoch = st.session_state["current_epoch"]
    fig_config = st.session_state["fig_config"]
    scoring_naive = st.session_state["dataset_processed"]["scoring"]["scoring_naive"]
    time_hrs = st.session_state["dataset_processed"]["scoring"]["time_hrs"]
    subject_id = fig_config["subject_id"]

    # Create figure to indicate state of the scoring.
    fig_scoring = plt.figure(figsize=(20, 2))
    ax_scoring = fig_scoring.add_subplot(1, 1, 1)
    ax_scoring.step(time_hrs, scoring_naive, where="mid", color="black")
    ax_scoring.axvline(x=current_epoch * 30 / 3600, color="red", linestyle="--", label="Current Epoch")
    ax_scoring.scatter(current_epoch * 30 / 3600, scoring_naive[current_epoch], color="red", s=100, zorder=5)
    ax_scoring.set_title("Scoring")
    ax_scoring.set_ylabel("Sleep Stage")
    ax_scoring.set_xlabel("Time (hours)")
    ax_scoring.set_yticks(ticks=range(5), labels=SLEEP_STAGE_LABELS)

    # Save figure to session state.
    fig_config["figures"]["scoring"] = fig_scoring

    # Save figure to cache
    fig_scoring_path = fig_config["svg_paths"]["scoring"] 
    fig_scoring.savefig(fig_scoring_path)

if __name__ == "__main__":
    main()
    