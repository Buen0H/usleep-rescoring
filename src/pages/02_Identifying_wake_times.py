import logging
import streamlit as st
import mne
import numpy as np
from streamlit_shortcuts import add_shortcuts

from utils.figures import draw_hypnogram, draw_polysomnography
from utils.utils import safely_startup_page, process_biosignals

st.title("Identifying Wake Times")

def main():
    # Startup sequence.
    if not safely_startup_page():
        return
    # Populate session_state. Get wake events, set current epoch to first wake event, get biosignals.
    current_subject_id = st.session_state["dataset_downloaded"]["subject_id"]
    fs = st.session_state["dataset_downloaded"]["raw_obj"].info["sfreq"]
    ## Initialize page by loading wake events and setting current epoch to first wake event.
    if st.session_state["initialized"]["page_02"] == False:
        # Store subject id for figures
        st.session_state["fig_config"]["subject_id"] = current_subject_id
        # Get wake events from downloaded dataset.
        load_wake_events_to_session_state()
        events, event_ids = st.session_state["dataset_processed"]["wake_events"]
        # Initialize rescored wake time identification dictionary in session state.
        lights_off_offset = st.session_state["wake_time_identification"]["lights_off"]
        st.session_state["dataset_rescored"]["wake_time_identification"] = {
            event_id: {
                "recorded_onset": event_onset + lights_off_offset,  # Store the original recorded onset time.
                "rescored_onset": None,
            }
            for event_onset, event_id in zip(events, event_ids)
        }
        # Initialize wake time selection to downloaded wake events.
        st.session_state["wake_time_identification"]["wake_time_selection"] = events / fs # Convert to seconds.
        # Set current epoch to first wake event.
        if len(events) > 0:
            st.session_state["current_epoch"] = int(events[0] / fs / 30)  # Set to first wake event.
        else:
            st.session_state["current_epoch"] = 0  # Default to first epoch if no wake events.
            st.warning("No wake events detected in the annotations for this subject.")
            return;
        st.session_state["initialized"]["page_02"] = True
        st.session_state["initialized"]["page_01"] = False  # Reset page 1 initialization.
    else: ## For subsequent runtimes, just get wake events.
        events, event_ids = st.session_state["dataset_processed"]["wake_events"]
    ## Get biosignals for the current epoch.
    dataset_processed = st.session_state["dataset_processed"]
    dataset_processed["biosignals"] = process_biosignals(st.session_state["current_epoch"], current_subject_id)
    ## Draw figures.
    draw_hypnogram(draw_wake_events=True)
    draw_polysomnography(draw_wake_events=True)

    # Populate UI elements.
    ## Variables for navigating through recording based on wake events.
    current_epoch = st.session_state["current_epoch"]
    previous_epoch = current_epoch - 1
    next_epoch = current_epoch + 1
    previous_wake_event, current_wake_event, next_wake_event = find_closest_wake_events(current_epoch)
    st.session_state["wake_time_identification"]["current_wake_event"] = current_wake_event  # Store current wake event in session state.
    ## Display hypnogram.
    st.image(st.session_state["fig_config"]["svg_paths"]["scoring"], width="stretch")
    ## Add slider for manual indication of wake times\
    update_slider_default_value()  # Set default value for slider based on current wake time selection.
    _, col, _ = st.columns([0.08, 0.88, 0.045])  # Center the slider by using columns.
    with col:
        st.slider("Manually indicate wake times by sliding to the corresponding epoch. Use the navigation buttons below to move through the recording.",
                min_value=0,
                max_value=int(len(dataset_processed["biosignals"]["signals"][0]) // fs),
                step=1,
                on_change=update_wake_choice,
                key="slider_wake_choice",
                # width=800,    # Optimize so that it aligns with matplotlib figure
                disabled=(len(events) == 0))  # Disable slider if no wake events detected.
    ## Display polysomnography.
    st.image(st.session_state["fig_config"]["svg_paths"]["biosignals"], width="stretch")
    ## Mechanism to navigate through recording.
    col1, col2, col3, col4 = st.columns(4)
    col1.button("Rewind", key="rewind", width="stretch",
                disabled=(current_epoch == previous_wake_event or previous_wake_event is None),
                on_click=update_epoch, args=(previous_wake_event, ))
    col2.button("Back", key="back", width="stretch",
                disabled=(current_epoch == 0),
                on_click=update_epoch, args=(previous_epoch, ))
    col3.button("Forward", key="forward", width="stretch",
                disabled=(current_epoch == len(dataset_processed["scoring"]["scoring_naive"]) - 1),
                on_click=update_epoch, args=(next_epoch, ))
    col4.button("Fast forward", key="fast_forward", width="stretch",
                disabled=(current_epoch == next_wake_event or next_wake_event is None),
                on_click=update_epoch, args=(next_wake_event, ))
    
    # Add shortcuts for the buttons
    add_shortcuts(
        rewind="arrowdown",
        back="arrowleft",
        forward="arrowright",
        fast_forward="arrowup",
    )
    
    # Debugging information.
    st.subheader("Debugging Information")
    st.write("Current recorded file:")
    st.write(st.session_state["dataset_rescored"]["wake_time_identification"])

def is_valid_event_id(event_id):
    return event_id.startswith(("a_NREM", "a_REM")) and event_id.endswith("s")

def filter_for_wake_annotations(s):
    s = str(s)
    return (s.startswith("a_NREM_") or s.startswith("a_REM_")) and s.endswith("_s")

def load_wake_events_to_session_state():
    triggers = st.session_state["dataset_downloaded"]["triggers"]
    #filter for valid events
    if triggers is not None:
        lights_off_offset = triggers[triggers["descriptions"] == "lights_off"]["onsets"].values[0]
        st.session_state["wake_time_identification"]["lights_off"] = lights_off_offset
        triggers_mask = triggers["descriptions"].apply(filter_for_wake_annotations).tolist()
        triggers_filtered = triggers[triggers_mask]
        wake_events_onsets = triggers_filtered["onsets"].values - lights_off_offset
        wake_events_labels = triggers_filtered["descriptions"].tolist()
        st.session_state["dataset_processed"]["wake_events"] = (wake_events_onsets, wake_events_labels)
        return wake_events_onsets, wake_events_labels
    else:
        st.warning("No triggers data found.")
        return None, None

def update_epoch(epoch: int):
    """Update the current epoch in session state."""
    st.session_state["current_epoch"] = epoch
    logging.info(f"Current epoch updated to {epoch}.")
    update_slider_default_value()

def find_closest_wake_events(current_epoch: int):
    """Find the closest wake events to the current epoch."""
    events, _ = st.session_state["dataset_processed"]["wake_events"]
    if len(events) == 0:
        return None, None
    fs = st.session_state["dataset_downloaded"]["raw_obj"].info["sfreq"]
    event_epochs = events / fs / 30  # Convert event times to epochs.
    event_epochs = [int(epoch) for epoch in event_epochs]  # Convert to integers to point to epoch.
    previous_wake_event = max([epoch for epoch in event_epochs if epoch < current_epoch], default=None)
    current_wake_event = min([epoch for epoch in event_epochs if epoch >= current_epoch], default=None)
    next_wake_event = min([epoch for epoch in event_epochs if epoch > current_epoch], default=None)
    return int(previous_wake_event) if previous_wake_event is not None else None, \
            int(current_wake_event) if current_wake_event is not None else None, \
            int(next_wake_event) if next_wake_event is not None else None

def update_slider_default_value():
    """Update the default value for the wake time slider."""
    wake_time_selection = st.session_state["wake_time_identification"]["wake_time_selection"]
    current_epoch = st.session_state["current_epoch"]
    if wake_time_selection is not None:
        idx = np.argmin(np.abs(wake_time_selection - (current_epoch * 30)))  # Find closest wake time to current epoch.
        slider_default_value = int(wake_time_selection[idx] - current_epoch * 30)
        if slider_default_value < 0:
            slider_default_value = 0
        elif slider_default_value > 30:
            slider_default_value = 30
    else:
        slider_default_value = 0 
    st.session_state["slider_wake_choice"] = slider_default_value


def update_wake_choice():
    """Update wake time selection in session state based on manual slider input."""
    wake_time_selection = st.session_state["wake_time_identification"]["wake_time_selection"]   # in seconds.
    wake_choice = st.session_state["slider_wake_choice"] + st.session_state["current_epoch"] * 30  # Convert to seconds.
    idx = np.argmin(np.abs(wake_time_selection - wake_choice))
    logging.info(f"Updating wake time selection from {wake_time_selection[idx]} to {wake_choice}.")
    wake_time_selection[idx] = wake_choice
    # Store the rescored wake time in session state.
    fs = st.session_state["dataset_downloaded"]["raw_obj"].info["sfreq"]
    current_wake_event_label = st.session_state["dataset_processed"]["wake_events"][1][idx]
    lights_off_offset = st.session_state["wake_time_identification"]["lights_off"]
    st.session_state["dataset_rescored"]["wake_time_identification"][current_wake_event_label]["rescored_onset"] = \
        wake_choice * fs + lights_off_offset  # Convert back to seconds.

if __name__ == "__main__":
    main()