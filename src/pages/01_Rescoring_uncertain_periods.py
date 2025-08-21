import streamlit as st
import numpy as np
import logging

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
    # Rescoring logic implementation.
    # Process downloaded data.
    dataset_downloaded = st.session_state["dataset_downloaded"]
    dataset_processed = st.session_state["dataset_processed"]
    if dataset_processed["subject_id"] != subject_id_download:
        logging.info(f"Processing data for subject {subject_id_download}.")

    st.write(f"Currently rescoring uncertain periods for subject: {st.session_state['subject_id']}")
    # Populate the session state with rescoring information.
    dataset_rescored = st.session_state["dataset_rescored"]
    if dataset_rescored["subject_id"] is None:
        scoring_array = dataset_rescored["scoring"]
        dataset_rescored["subject_id"] = subject_id_download
        dataset_rescored["scoring_manual"] = scoring_array.copy()
        dataset_rescored["scoring_manual_mask"] = np.zeros_like(scoring_array, dtype=bool)
        logging.info(f"Initialized rescoring for subject {subject_id_download}")
    # Populate the session state with processed data.
    # Populate the session state with initial figure configuration.
    fig_config = st.session_state["fig_config"]
    if fig_config["current_epoch"] == -1:   # If current epoch is -1, it means no figure has been generated yet.
        # If no epoch is selected, set the current epoch to the first uncertain period
        if data["scoring_processed"]["n_uncertain_periods"] == 0:
            fig_config["current_epoch"] = 0
            logging.warning("No uncertain periods found in the data.")
            st.warning("No uncertain periods found in the data. Please select a different file.")
        else:
            first_uncertain_hour = data["scoring_processed"]["uncertain_periods"][0]["start_hour"]
            first_uncertain_epoch = int(first_uncertain_hour * 3600 / 30)  # Convert hours to epochs.
            fig_config["current_epoch"] = first_uncertain_epoch



@st.cache_data
def process_scoring_data(_data: Dict, subject_id: int) -> Dict:
    """Process scoring data to extract uncertain periods."""
    # Analyze uncertain periods in the scoring data
    _data["scoring_processed"] = analyze_uncertain_periods(_data["scoring"])
    return _data


if __name__ == "__main__":
    main()
    