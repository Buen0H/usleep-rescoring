import logging
from typing import Dict
import streamlit as st
from streamlit_shortcuts import add_shortcuts
import numpy as np
from utils.nil_sleep_analysis import analyze_uncertain_periods
from utils.streamlit_connection_to_radboud import upload_file_to_repository
import matplotlib.pyplot as plt

# ''' This page allows users to rescore uncertain periods in sleep data.

# TODO:
# - Minor. Improve caching strategy for processing data to improve code readability.

# '''

SLEEP_STAGE_LABELS = ["Wake", "REM", "N1", "N2", "N3"] # Sleep stage labels.

def main():
    # Set the title of the Streamlit app.
    st.title("Rescoring Uncertain Periods")
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
        ## Initialize rescoring data structure from processed data.
        logging.info(f"Initializing rescoring for subject {dataset_processed['subject_id']}")
        scoring_array = dataset_processed["scoring"]["scoring_naive"]
        dataset_rescored["scoring_manual"] = scoring_array.copy()
        dataset_rescored["scoring_manual_mask"] = np.zeros_like(scoring_array, dtype=bool)
        dataset_rescored["subject_id"] = dataset_processed["subject_id"]
        ## Update figure configuration.
        fig_config["subject_id"] = dataset_processed["subject_id"]
        create_figures()
    # Update figure configuration if current epoch changed.
    current_epoch = st.session_state["current_epoch"]
    if current_epoch != fig_config["current_epoch"]:
        # Update figures for the new current epoch.
        logging.info(f"Updating figures for current epoch {st.session_state['current_epoch']}.")
        update_scoring_figure(current_epoch)
        dataset_processed["biosignals"] = process_biosignals(current_epoch, subject_id_download)
        update_biosignals_figure()
        fig_config["current_epoch"] = st.session_state["current_epoch"]
    # Save manually scored information.
    np.save(f"{st.secrets['CACHE_PATH']}/{subject_id_download}_scoring_manual.npy", dataset_rescored["scoring_manual"])
    # Populate UI elements.
    st.write(f"Currently rescoring uncertain periods for subject: {dataset_processed['subject_id']}")
    st.image(fig_config["svg_paths"]["scoring"], use_container_width=True)
    st.image(fig_config["svg_paths"]["biosignals"], use_container_width=True)
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
    ## Mechanism to grade uncertain periods.
    is_uncertain = dataset_processed["scoring"]["mask_uncertain"][current_epoch]
    is_graded = dataset_rescored["scoring_manual_mask"][current_epoch]
    human_scoring = dataset_rescored["scoring_manual"][current_epoch] if is_graded else None
    auto_scoring = dataset_processed["scoring"]["scoring_naive"][current_epoch]
    button_labels = SLEEP_STAGE_LABELS.copy()
    button_labels[auto_scoring] += " :desktop_computer:"
    if is_graded:
        button_labels[human_scoring] += " :nerd_face:"
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.button(button_labels[0], key="wake", use_container_width=True,
                on_click=update_scoring, args=(0, ), disabled=not is_uncertain)
    col2.button(button_labels[1], key="rem", use_container_width=True,
                on_click=update_scoring, args=(1, ), disabled=not is_uncertain)
    col3.button(button_labels[2], key="n1", use_container_width=True,
                on_click=update_scoring, args=(2, ), disabled=not is_uncertain)
    col4.button(button_labels[3], key="n2", use_container_width=True,
                on_click=update_scoring, args=(3, ), disabled=not is_uncertain)
    col5.button(button_labels[4], key="n3", use_container_width=True,
                on_click=update_scoring, args=(4, ), disabled=not is_uncertain)
    ## Mechanism to navigate through recording.
    col1, col2, col3, col4 = st.columns(4)
    col1.button("Rewind", key="rewind", use_container_width=True,
                # disabled=(current_epoch == previous_uncertain_epoch),
                on_click=update_epoch, args=(previous_uncertain_epoch, ))
    col2.button("Back", key="back", use_container_width=True,
                disabled=(current_epoch == 0),
                on_click=update_epoch, args=(previous_epoch, ))
    col3.button("Forward", key="forward", use_container_width=True,
                disabled=(current_epoch == len(dataset_processed["scoring"]["scoring_naive"]) - 1),
                on_click=update_epoch, args=(next_epoch, ))
    col4.button("Fast forward", key="fast_forward", use_container_width=True,
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
    with open(f"{st.secrets['CACHE_PATH']}/{subject_id_download}_scoring_manual.npy", "rb") as f:
        col1.download_button(
            label="Download scoring file",
            data=f,
            file_name=f"{subject_id_download}_scoring_manual.npy",
            mime="application/octet-stream",
            use_container_width=True,
        )
    if col2.button("Upload file to repository", use_container_width=True):
        logging.info("Upload file to repository button clicked.")
        err = upload_file_to_repository(st.secrets["CACHE_PATH"] + f"{subject_id_download}_scoring_manual.npy")
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
    # Get the time slice for the current epoch
    time_start = n_epoch * 30
    time_stop = (n_epoch + 1) * 30
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

def plot_masked_regions(ax, mask, time, color="green", alpha=0.3):  # Warning with changing alpha
    """Plot regions where the mask is True on the given axes.
    Changing alpha from 0.3 requires changing green_rgba variable in update_scoring_figure.
    """
    # Find transitions
    transitions = np.diff(mask.astype(int))
    starts = np.where(transitions == 1)[0] + 1
    ends = np.where(transitions == -1)[0] + 1

    # Handle edge cases
    if mask[0]:
        starts = np.insert(starts, 0, 0)
    if mask[-1]:
        ends = np.append(ends, len(mask))

    for start, end in zip(starts, ends):
        ax.fill_betweenx([0, 5], time[start], time[end-1] + 30/3600, color=color, alpha=alpha)

def create_figures():
    """Create figures for visualization."""

    # Get pointers for information inside session state.
    current_epoch = st.session_state["current_epoch"]
    current_epoch_hrs = current_epoch * 30 / 3600  # Convert epoch to hours.
    fig_config = st.session_state["fig_config"]
    dataset_processed = st.session_state["dataset_processed"]
    scoring_naive = dataset_processed["scoring"]["scoring_naive"]
    time_hrs = dataset_processed["scoring"]["time_hrs"]
    uncertain_scoring_mask = dataset_processed["scoring"]["mask_uncertain"]
    manual_scoring_mask = st.session_state["dataset_rescored"]["scoring_manual_mask"]
    subject_id = fig_config["subject_id"]
    logging.info(f"Creating figures for {subject_id}.")

    # Create figure to indicate state of the scoring.
    logging.info(f"Creating scoring figure with current epoch at {current_epoch_hrs} hours ({current_epoch}).")
    fig_scoring = plt.figure(figsize=(20, 2))
    ax_scoring = fig_scoring.add_subplot(1, 1, 1)
    ## Populate figure with scoring data.
    ax_scoring.step(time_hrs, scoring_naive, where="mid", color="black")
    ax_scoring.axvline(x=current_epoch_hrs, color="red", linestyle="--", label="Current Epoch")
    ax_scoring.scatter(current_epoch_hrs, scoring_naive[current_epoch], color="red", s=100, zorder=5)
    ## Highlight uncertain periods.
    if np.any(uncertain_scoring_mask):
        plot_masked_regions(ax_scoring, uncertain_scoring_mask, time_hrs, color="red")
    ## Highlight manually graded periods.
    if np.any(manual_scoring_mask):
        plot_masked_regions(ax_scoring, manual_scoring_mask, time, color="green")
    ## Configure axes.
    ax_scoring.set_title(f"Scoring for Subject {subject_id}")
    ax_scoring.set_ylabel("Sleep Stage")
    ax_scoring.set_xlabel("Time (hours)")
    ax_scoring.set_yticks(ticks=range(5), labels=SLEEP_STAGE_LABELS)
    ax_scoring.set_ylim(4.25,-0.25)
    fig_scoring.tight_layout()

    # Create figure to display the raw data for the current epoch.
    logging.info(f"Creating raw data figure for current epoch {current_epoch}.")
    cropped_data = dataset_processed["biosignals"]
    signals = cropped_data["signals"]
    time = cropped_data["time"]
    ch_labels = cropped_data["ch_labels"]
    ## Create figure.
    fig_raw = plt.figure(figsize=(20, 5))
    ax_raw = fig_raw.add_subplot(1, 1, 1)
    ## Populate figure with raw data. Move to processing function?
    scale_config = fig_config["scaling"]
    for idx, (signal, ch_label) in enumerate(zip(signals, ch_labels)):
        # Autoscaling
        if ch_label.startswith("EOG"):
            scale_val = scale_config["EOG"]
            color = "green"
        elif ch_label.startswith("EMG"):
            scale_val = scale_config["EMG"]
            color = "red"
        else:
            scale_val = scale_config["EEG"]
            color = "black"
        c_range = 2 * scale_val 
        signal *= 1e6 # Convert to microvolts
        signal /= c_range
        ax_raw.plot(time, signal + idx, linewidth=0.5, color=color)     
    ## Configure axes.
    ax_raw.set_title(f"Raw Data for Subject {subject_id} - Epoch {current_epoch}")
    ax_raw.set_xlabel("Time (seconds)")
    ax_raw.set_ylabel("Channels")  
    ax_raw.set_yticks(ticks=range(len(ch_labels)), labels=ch_labels)
    ax_raw.set_ylim(len(ch_labels), -1)     # Plot from top to bottom.
    fig_raw.tight_layout()

    # Save figure to session state.
    fig_config["figures"]["scoring"] = fig_scoring
    fig_config["figures"]["biosignals"] = fig_raw

    # Save figure to cache
    fig_scoring_path = fig_config["svg_paths"]["scoring"] 
    fig_raw_path = fig_config["svg_paths"]["biosignals"]
    fig_scoring.savefig(fig_scoring_path)
    fig_raw.savefig(fig_raw_path)

def update_scoring_figure(current_epoch: int):
    """Update the scoring figure with the current epoch."""
    # Get pointers for information inside session state.
    fig_config = st.session_state["fig_config"]
    fig_scoring = fig_config["figures"]["scoring"]
    scoring_naive = st.session_state["dataset_processed"]["scoring"]["scoring_naive"]
    # Get the current time in hours.
    current_epoch_hrs = current_epoch * 30 / 3600  # Convert epoch to hours.
    # Update the vertical line and scatter point for the current epoch.
    ax_scoring = fig_scoring.axes[0]
    ax_scoring.lines[1].set_xdata([current_epoch_hrs, current_epoch_hrs])
    ax_scoring.collections[0].set_offsets([[current_epoch_hrs, scoring_naive[current_epoch]]])
    # Update graded periods.
    manual_scoring_mask = st.session_state["dataset_rescored"]["scoring_manual_mask"]
    time = st.session_state["dataset_processed"]["scoring"]["time_hrs"]
    if np.any(manual_scoring_mask):
        # Remove only green shaded regions (manually graded periods)
        # Matplotlib converts "green" to RGBA (0, 0.5019608, 0, 0.3) for alpha=0.3
        green_rgba = (0.0, 0.5019608, 0.0, 0.3)
        to_remove = []
        for coll in ax_scoring.collections:
            # fill_betweenx returns PolyCollection; get facecolor
            fc = coll.get_facecolor()
            # fc is Nx4 array; check first row
            if fc.shape[0] > 0 and np.allclose(fc[0], green_rgba, atol=0.05):
                to_remove.append(coll)
        for coll in to_remove:
            coll.remove()
        # Plot new shaded regions for manually graded periods.
        plot_masked_regions(ax_scoring, manual_scoring_mask, time, color="green")
    # Update svg image.
    fig_scoring_path = fig_config["svg_paths"]["scoring"]
    fig_scoring.savefig(fig_scoring_path)

def update_biosignals_figure():
    """Update the biosignals figure with the current epoch."""
    # Get pointers for information inside session state.
    fig_config = st.session_state["fig_config"]
    fig_biosignals = fig_config["figures"]["biosignals"]
    cropped_data = st.session_state["dataset_processed"]["biosignals"]
    # Update the raw data for the current epoch.
    signals = cropped_data["signals"]
    time = cropped_data["time"]
    ch_labels = cropped_data["ch_labels"]
    ax_biosignals = fig_biosignals.axes[0]
    for idx, signal in enumerate(signals):
        # Autoscaling
        if ch_labels[idx].startswith("EOG"):
            scale_val = 150
        elif ch_labels[idx].startswith("EMG"):
            scale_val = 50
        else:
            scale_val = 30
        c_range = 2 * scale_val 
        signal *= 1e6
        signal /= c_range
        ax_biosignals.lines[idx].set_ydata(signal + idx)
        ax_biosignals.lines[idx].set_xdata(time)
    # Update svg image.
    fig_biosignals_path = fig_config["svg_paths"]["biosignals"]
    fig_biosignals.savefig(fig_biosignals_path)

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

if __name__ == "__main__":
    main()
    