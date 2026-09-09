import logging
from typing import Dict
import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np

from utils.nil_sleep_analysis import analyze_uncertain_periods

def safely_startup_page():
    """Safely execute startup sequence for a page."""
    # Check if the session state is initialized. Likely caused by a refresh.
    if "initialized" not in st.session_state:
        logging.error("Session state not initialized.")
        st.error("Refresh detected. Please go back to the homepage.")
        return False
    
    # Check if user has downloaded a dataset.
    subject_id_choice = st.session_state["subject_id"]
    subject_id_download = st.session_state["dataset_downloaded"]["subject_id"]
    if subject_id_choice is None:
        st.warning("Please select a subject from the homepage to continue.")
        logging.warning("No subject ID selected for rescoring.")
        return False
    elif subject_id_choice != subject_id_download:
        st.warning(f"Please go back to homepage and allow for download to complete.")
        logging.warning(f"Subject ID {subject_id_choice} does not match downloaded subject ID {subject_id_download}.")
        return False
    return True

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

def get_CET_time_str():
    """Return current Central European Time as YYYYMMDD_HHMMSS."""
    cet_dt = datetime.now(ZoneInfo("Europe/Amsterdam"))
    return cet_dt.strftime("%Y%m%d_%H%M%S")