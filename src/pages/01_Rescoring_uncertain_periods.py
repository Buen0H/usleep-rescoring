import logging
from datetime import datetime
from zoneinfo import ZoneInfo
import os
from typing import Dict
import streamlit as st
from streamlit_shortcuts import add_shortcuts
import numpy as np
from utils.nil_sleep_analysis import analyze_uncertain_periods
from utils.figures import draw_hypnogram, draw_polysomnography, update_hypnogram, update_polysomnography
from utils.streamlit_connection_to_radboud import upload_file_to_repository
from utils.utils import safely_startup_page
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')

# ''' This page allows users to rescore uncertain periods in sleep data.

# TODO:
# - Minor. Improve caching strategy for processing data to improve code readability.

# '''

SLEEP_STAGE_LABELS = ["Wake", "REM", "N1", "N2", "N3"] # Sleep stage labels. Display only.
DISPLAY_ORDER_MAP = {0: 0, 1: 2, 2: 3, 3: 4, 4: 1}  # Desired display order: Wake, REM, N1, N2, N3

def main():
    # Set the title of the Streamlit app.
    st.title("Rescoring Uncertain Periods")
    # Startup sequence.
    if not safely_startup_page():
        return
    # Get pointers from session state.
    subject_id_download = st.session_state["dataset_downloaded"]["subject_id"]
    dataset_processed = st.session_state["dataset_processed"]
    dataset_rescored = st.session_state["dataset_rescored"]
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
        st.session_state["current_epoch"] = current_epoch
        # Load a copy of the raw data for the current epoch
        dataset_processed["biosignals"] = process_biosignals(current_epoch, subject_id_download)
        # Set name for processed data held in session state.
        dataset_processed["subject_id"] = subject_id_download
        curr_time_str = get_CET_time_str()
        manual_scoring_filename = f"{subject_id_download}_scoring_manual_{curr_time_str}.npy"
        st.session_state["manual_scoring_filename"] = manual_scoring_filename
        ## Initialize rescoring data structure from processed data.
        logging.info(f"Initializing rescoring for subject {dataset_processed['subject_id']}")
        scoring_array = dataset_processed["scoring"]["scoring_naive"]
        dataset_rescored["scoring_manual"] = scoring_array.copy()
        dataset_rescored["scoring_manual_mask"] = np.zeros_like(scoring_array, dtype=bool)
        dataset_rescored["subject_id"] = dataset_processed["subject_id"]
        ## Update figure configuration.
        fig_config["subject_id"] = dataset_processed["subject_id"]
        draw_hypnogram(draw_scoring_mask=True)
        draw_polysomnography()
    # Update figure configuration if current epoch changed.
    current_epoch = st.session_state["current_epoch"]
    if current_epoch != fig_config["current_epoch"]:
        # Update figures for the new current epoch.
        logging.info(f"Updating figures for current epoch {st.session_state['current_epoch']}.")
        update_hypnogram(current_epoch, draw_scoring_mask=True)
        dataset_processed["biosignals"] = process_biosignals(current_epoch, subject_id_download)
        update_polysomnography()
        fig_config["current_epoch"] = st.session_state["current_epoch"]
    # Save manually scored information.
    np.save(os.path.join(st.secrets['CACHE_PATH'],st.session_state["manual_scoring_filename"]), dataset_rescored["scoring_manual"])
    # Populate UI elements.
    st.write(f"Currently rescoring uncertain periods for subject: {dataset_processed['subject_id']}")
    st.image(fig_config["svg_paths"]["scoring"], width="stretch")
    st.image(fig_config["svg_paths"]["biosignals"], width="stretch")
    ## Sidebar for graph configuration.
    # scale_config = st.session_state["fig_config"]["scaling"]
    # with st.sidebar:
    #     st.header("Configuration")
    #     st.subheader("Scale signals.")
    #     scale_config["EOG"] = st.number_input("EOG scale (µV)", min_value=10, max_value=500, value=scale_config["EOG"], step=10, key="eog_scale",)
    #     scale_config["EMG"] = st.number_input("EMG scale (µV)", min_value=10, max_value=500, value=scale_config["EMG"], step=10, key="emg_scale",)
    #     scale_config["EEG"] = st.number_input("EEG scale (µV)", min_value=10, max_value=500, value=scale_config["EEG"], step=10, key="eeg_scale",)
    ## Variables with the current epoch and previous/next epochs.
    current_epoch = st.session_state["current_epoch"]
    previous_epoch = current_epoch - 1
    next_epoch = current_epoch + 1
    previous_uncertain_epoch, next_uncertain_epoch = find_closest_uncertain_periods(current_epoch)
    ## Mechanism to grade uncertain periods. Button() args match input npy format.
    is_uncertain = dataset_processed["scoring"]["mask_uncertain"][current_epoch]
    is_graded = dataset_rescored["scoring_manual_mask"][current_epoch]
    human_scoring = DISPLAY_ORDER_MAP[dataset_rescored["scoring_manual"][current_epoch]] if is_graded else None
    auto_scoring = DISPLAY_ORDER_MAP[dataset_processed["scoring"]["scoring_naive"][current_epoch]]
    button_labels = SLEEP_STAGE_LABELS.copy()
    button_labels[auto_scoring] += " :desktop_computer:"
    if is_graded:
        button_labels[human_scoring] += " :nerd_face:"
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.button(button_labels[0], key="wake", width="stretch",
                on_click=update_scoring, args=(0, ), disabled=not is_uncertain)
    col2.button(button_labels[1], key="rem", width="stretch",
                on_click=update_scoring, args=(4, ), disabled=not is_uncertain)
    col3.button(button_labels[2], key="n1", width="stretch",
                on_click=update_scoring, args=(1, ), disabled=not is_uncertain)
    col4.button(button_labels[3], key="n2", width="stretch",
                on_click=update_scoring, args=(2, ), disabled=not is_uncertain)
    col5.button(button_labels[4], key="n3", width="stretch",
                on_click=update_scoring, args=(3, ), disabled=not is_uncertain)
    ## Mechanism to navigate through recording.
    col1, col2, col3, col4 = st.columns(4)
    col1.button("Rewind", key="rewind", width="stretch",
                # disabled=(current_epoch == previous_uncertain_epoch),
                on_click=update_epoch, args=(previous_uncertain_epoch, ))
    col2.button("Back", key="back", width="stretch",
                disabled=(current_epoch == 0),
                on_click=update_epoch, args=(previous_epoch, ))
    col3.button("Forward", key="forward", width="stretch",
                disabled=(current_epoch == len(dataset_processed["scoring"]["scoring_naive"]) - 1),
                on_click=update_epoch, args=(next_epoch, ))
    col4.button("Fast forward", key="fast_forward", width="stretch",
                disabled=(current_epoch == next_uncertain_epoch),
                on_click=update_epoch, args=(next_uncertain_epoch, ))
    # Add shortcuts for the buttons
    add_shortcuts(
        wake="w",
        rem="4",
        n1="1",
        n2="2",
        n3="3",
        rewind="arrowdown",
        back="arrowleft",
        forward="arrowright",
        fast_forward="arrowup",
    )

    # Buttons to download locally or upload files to the repository.
    col1, col2 = st.columns(2)
    with open(os.path.join(st.secrets["CACHE_PATH"], st.session_state["manual_scoring_filename"]), "rb") as f:
        col1.download_button(
            label="Download scoring file",
            data=f,
            file_name=st.session_state["manual_scoring_filename"],
            mime="application/octet-stream",
            width="stretch",
        )
    if col2.button("Upload file to repository", width="stretch"):
        logging.info("Upload file to repository button clicked.")
        err = upload_file_to_repository(os.path.join(st.secrets["CACHE_PATH"], st.session_state["manual_scoring_filename"]))
        if err:
            st.success("File uploaded successfully.")
        else:
            st.error("Failed to upload file. Please check the logs for more details.")

@st.cache_data
def process_scoring_data(subject_id: str) -> Dict:
    """Process scoring data to extract uncertain periods."""
    # Get pointer to scoring data from session state
    confidence_data = st.session_state["dataset_downloaded"]["scoring"]
    # Analyze uncertain periods in the scoring data
    scoring_processed = analyze_uncertain_periods(confidence_data)
    return scoring_processed

def process_biosignals(n_epoch: int, subject_id: str) -> Dict:
    """Process biosignals for visualization."""
    # Get pointer to raw object
    raw_obj = st.session_state["dataset_downloaded"]["raw_obj"]
    # Get the sampling frequency from the raw data
    fs = raw_obj.info["sfreq"]
    # Get the time slice for the current epoch. Added safeguards.
    time_start = max(0, n_epoch * 30)
    time_stop = min((n_epoch + 1) * 30, raw_obj.times[-1])
    raw_selection = raw_obj.copy().crop(tmin=time_start, tmax=time_stop)
    # Sort channels for visualization
    indeces = raw_selection.ch_names
    eog_channels = [ch for ch in indeces if "EOG" in ch]
    emg_channels = [ch for ch in indeces if "EMG" in ch]
    egg_channels = [ch for ch in indeces if ch not in eog_channels and ch not in emg_channels]
    ordered_channels = eog_channels + egg_channels + emg_channels
    raw_selection.reorder_channels(ordered_channels)
    # Get the data and channel labels
    signals, time = raw_selection[:, :]
    ch_labels = raw_selection.ch_names
    # Handle case with single EOG channel by duplicating it
    if "EOG" in ch_labels:
        idx_eog = ch_labels.index("EOG")
        ch_labels.insert(idx_eog + 1, "EOG2")
        signals = np.insert(signals, idx_eog + 1, signals[idx_eog], axis=0)
    # Create a dictionary to store the processed data
    processed_data = {
        "signals": signals,
        "time": time,
        "ch_labels": ch_labels,
        "fs": fs,
    }
    return processed_data

def find_closest_uncertain_periods(current_epoch: int) -> tuple[int, int]:
    """Find the closest uncertain period to the current epoch in both directions."""
    # Get the current time in hours
    current_hour = current_epoch * 30 / 3600  # 30 seconds per epoch

    # Get the uncertain periods from scoring data
    uncertain_periods = st.session_state["dataset_processed"]["scoring"]["uncertain_periods"]
    start_hours = [period["start_hour"] for period in uncertain_periods]

    # Find the next uncertain period (smallest start_hour > current_hour)
    next_period_hr = None
    for hour in sorted(start_hours):
        if hour > current_hour:
            next_period_hr = hour
            break
    if next_period_hr is None:
        next_period_hr = current_hour  # No next, stay at current

    # Find the previous uncertain period (largest start_hour < current_hour)
    prior_period_hr = None
    for hour in sorted(start_hours, reverse=True):
        if hour < current_hour:
            prior_period_hr = hour
            break
    if prior_period_hr is None:
        prior_period_hr = current_hour  # No previous, stay at current

    # Convert hours to epochs.
    prior_epoch = int(prior_period_hr * 3600 / 30)
    next_epoch = int(next_period_hr * 3600 / 30)
    return prior_epoch, next_epoch

def update_epoch(epoch: int):
    """Update the current epoch in session state."""
    st.session_state["current_epoch"] = epoch
    logging.info(f"Current epoch updated to {epoch}.")

def update_scoring(scoring: int):
    """Update the scoring for the current epoch."""
    # Get pointers for information inside session state.
    dataset_rescored = st.session_state["dataset_rescored"]
    current_epoch = st.session_state["current_epoch"]
    # Update the scoring manual and mask.
    dataset_rescored["scoring_manual"][current_epoch] = scoring
    dataset_rescored["scoring_manual_mask"][current_epoch] = True
    logging.info(f"Scoring updated for epoch {current_epoch} to {scoring}.")
    # Move to the next epoch.
    next_epoch = current_epoch + 1
    if next_epoch < len(dataset_rescored["scoring_manual"]):
        st.session_state["current_epoch"] = next_epoch
        logging.info(f"Moving to next epoch {next_epoch}.")

def get_CET_time_str():
    """Return current Central European Time as YYYYMMDD_HHMMSS."""
    cet_dt = datetime.now(ZoneInfo("Europe/Amsterdam"))
    return cet_dt.strftime("%Y%m%d_%H%M%S")

if __name__ == "__main__":
    main()
    