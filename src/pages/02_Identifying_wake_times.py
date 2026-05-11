import streamlit as st
import mne
import numpy as np
from utils.figures import draw_hypnogram, draw_polysomnography

st.title("Identifying Wake Times")

def main():
    load_wake_events_to_session_state()
    events, event_ids = st.session_state["dataset_processed"]["wake_events"]
    st.write("Events extracted from annotations:")
    st.write(events)
    st.write(event_ids)
    draw_hypnogram(draw_wake_events=True)
    draw_polysomnography()
    st.image(st.session_state["fig_config"]["svg_paths"]["scoring"], width="stretch")
    st.image(st.session_state["fig_config"]["svg_paths"]["biosignals"], width="stretch")

def is_valid_event_id(event_id):
    return event_id.startswith(("a_NREM", "a_REM")) and event_id.endswith("s")

def load_wake_events_to_session_state():
    """Extract wake events from the polysomnogram."""
    mne_raw_obj = st.session_state["dataset_downloaded"]["raw_obj"]
    events, event_ids = mne.events_from_annotations(mne_raw_obj)
    filtered_event_ids = [id for id in event_ids.keys() if is_valid_event_id(id)]
    filtered_event_idx = [event_ids[id] for id in filtered_event_ids]
    filtered_events = events[np.isin(events[:, 2], filtered_event_idx)]
    st.session_state["dataset_processed"]["wake_events"] = (filtered_events, filtered_event_ids)
    return filtered_events, filtered_event_ids  # Improve by returning labeled dictionary.

if __name__ == "__main__":
    main()