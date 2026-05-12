import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

import logging

'''Utility functions for creating and updating figures in the Streamlit app.
The concept here is to use the session state objects to draw figures. 
The drawing functions (draw_hypnogram and draw_polysomnography) create the initial figures and save them as SVGs.
The update functions (update_hypnogram and update_polysomnography) then update the existing figures and overwrite the SVGs. 

'''

SLEEP_STAGE_LABELS = ["Wake", "REM", "N1", "N2", "N3"] # Sleep stage labels. Display only.
DISPLAY_ORDER_MAP = {0: 0, 1: 2, 2: 3, 3: 4, 4: 1}  # Desired display order: Wake, REM, N1, N2, N3

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

def draw_hypnogram(draw_scoring_mask: bool = False, draw_wake_events: bool = False):
    """Create figure to indicate state of the scoring."""
    # Get pointers for information inside session state.
    current_epoch = st.session_state["current_epoch"]
    current_epoch_hrs = current_epoch * 30 / 3600  # Convert epoch to hours.
    fig_config = st.session_state["fig_config"]
    dataset_processed = st.session_state["dataset_processed"]
    scoring_naive = dataset_processed["scoring"]["scoring_naive"]
    time_hrs = dataset_processed["scoring"]["time_hrs"]
    uncertain_scoring_mask = dataset_processed["scoring"]["mask_uncertain"]
    manual_scoring_mask = st.session_state["dataset_rescored"]["scoring_manual_mask"]
    if draw_wake_events:
        fs = st.session_state["dataset_downloaded"]["raw_obj"].info["sfreq"]
        events, event_ids = dataset_processed["wake_events"]
    subject_id = fig_config["subject_id"]
    # logging.info(f"Creating hypnogram for {subject_id}.")

    fig_scoring = plt.figure(figsize=(20, 2))
    ax_scoring = fig_scoring.add_subplot(1, 1, 1)
    ## Adjust sleep stage display order.
    scoring_naive_remapped = [DISPLAY_ORDER_MAP[stage] for stage in scoring_naive]
    scoring_naive = np.array(scoring_naive_remapped)
    ## Populate figure with scoring data.
    ax_scoring.step(time_hrs, scoring_naive, where="mid", color="black")
    ax_scoring.axvline(x=current_epoch_hrs, color="red", linestyle="--", label="Current Epoch")
    ax_scoring.scatter(current_epoch_hrs, scoring_naive[current_epoch], color="red", s=100, zorder=5)
    if draw_wake_events and len(events) > 0:
        wake_event_times = events[:, 0] / fs / 3600  # Convert to hours.
        # st.write(f"Wake event times (hours): {wake_event_times}")
        ax_scoring.vlines(wake_event_times, ymin=0, ymax=5, color="blue", linestyle="--", label="Wake Events")
        # ax_scoring.scatter(wake_event_times, [1]*len(wake_event_times), color="blue", marker="s", s=100, label="Wake Events")
        ax_scoring.legend(loc="upper right")
    if draw_scoring_mask:
        ## Highlight uncertain periods.
        if np.any(uncertain_scoring_mask):
            plot_masked_regions(ax_scoring, uncertain_scoring_mask, time_hrs, color="red")
        ## Highlight manually graded periods.
        if np.any(manual_scoring_mask):
            plot_masked_regions(ax_scoring, manual_scoring_mask, time_hrs, color="green")
    ## Configure axes.
    ax_scoring.set_title(f"Scoring for Subject {subject_id}")
    ax_scoring.set_ylabel("Sleep Stage")
    ax_scoring.set_xlabel("Time (hours)")
    ax_scoring.set_yticks(ticks=range(5), labels=SLEEP_STAGE_LABELS)
    ax_scoring.set_ylim(4.25,-0.25)
    fig_scoring.tight_layout()


    # Save figure to session state.
    fig_config["figures"]["scoring"] = fig_scoring

    # Save figure to cache
    fig_scoring_path = fig_config["svg_paths"]["scoring"] 
    fig_scoring.savefig(fig_scoring_path)

def draw_polysomnography(draw_wake_events: bool = False):
    """Create figure to display the raw data for the current epoch."""
    # Get pointers for information inside session state.
    fig_config = st.session_state["fig_config"]
    dataset_processed = st.session_state["dataset_processed"]
    subject_id = fig_config["subject_id"]
    events, event_ids = dataset_processed["wake_events"]
    fs = st.session_state["dataset_downloaded"]["raw_obj"].info["sfreq"]

    cropped_data = dataset_processed["biosignals"]
    signals = cropped_data["signals"]
    time = cropped_data["time"]
    ch_labels = cropped_data["ch_labels"]

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
    if draw_wake_events and len(events) > 0:
        wake_event_times = events[:, 0] / fs  # Convert to seconds.
        wake_event_times = wake_event_times - st.session_state["current_epoch"] * 30  # Center around current epoch.
        for wake_time in wake_event_times:
            if 0 <= wake_time <= 30:  # Only plot wake events that are within the current epoch.
                ax_raw.axvline(x=wake_time, color="blue", linestyle="--", label="Wake Event")
        ax_raw.legend(loc="upper right")
    ## Configure axes.
    ax_raw.set_title(f"Raw Data for Subject {subject_id} - Epoch {st.session_state['current_epoch']}")
    ax_raw.set_xlabel("Time (seconds)")
    ax_raw.set_ylabel("Channels")  
    ax_raw.set_yticks(ticks=range(len(ch_labels)), labels=ch_labels)
    ax_raw.set_ylim(len(ch_labels), -1)     # Plot from top to bottom.
    fig_raw.tight_layout()

    
    # Save figure to session state.
    fig_config["figures"]["biosignals"] = fig_raw

    # Save figure to cache
    fig_raw_path = fig_config["svg_paths"]["biosignals"]
    fig_raw.savefig(fig_raw_path)

def update_hypnogram(current_epoch: int, draw_scoring_mask: bool):
    """Update the scoring figure with the current epoch."""
    # Get pointers for information inside session state.
    fig_config = st.session_state["fig_config"]
    fig_scoring = fig_config["figures"]["scoring"]
    scoring_naive = st.session_state["dataset_processed"]["scoring"]["scoring_naive"]
    ## Adjust sleep stage display order.
    scoring_naive_remapped = [DISPLAY_ORDER_MAP[stage] for stage in scoring_naive]
    scoring_naive = np.array(scoring_naive_remapped)
    # Get the current time in hours.
    current_epoch_hrs = current_epoch * 30 / 3600  # Convert epoch to hours.
    # Update the vertical line and scatter point for the current epoch.
    ax_scoring = fig_scoring.axes[0]
    ax_scoring.lines[1].set_xdata([current_epoch_hrs, current_epoch_hrs])
    ax_scoring.collections[0].set_offsets([[current_epoch_hrs, scoring_naive[current_epoch]]])
    # Update graded periods.
    manual_scoring_mask = st.session_state["dataset_rescored"]["scoring_manual_mask"]
    time = st.session_state["dataset_processed"]["scoring"]["time_hrs"]
    if draw_scoring_mask and np.any(manual_scoring_mask):
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

def update_polysomnography():
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
