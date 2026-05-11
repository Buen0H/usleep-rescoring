import logging
import streamlit as st

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