''' U-Sleep Rescoring GUI.
This tool can allow an expert to rescore low confidence time periods from an
autonomous scoring neural network. In low condifdence time periods, this tool will 
display the 30 seconds of raw data for the experiment and allow the expert to input
the correct sleep stage.

TODO
- Validate analyze_uncertain_periods logic with Sarah.
- Reorder biosignal channels [EOG x2, front EEG, ... , back EEG, EMG x2]
- Mark rescoring periods on the scoring plot
- Add a stage 2 to find start of wake cycles.
- Autoscaling; change scale of individual channels +30, +50 uV +130, +
- Disable channels
- Score artifact
- Handle subject_id switching
- Handle disabling of forward button when out of bounds
QUESTIONS
- What exactly are we recoring? The uncertain periods or the 30 second epochs?
- Reminder on what the stage 2 of the rescoring of wake cycles will look like.



'''

import streamlit as st
import os
import numpy as np
import logging

from utils.streamlit_connection_to_radboud import get_connection, download_dataset_from_repository
from utils.utils import process_scoring_data

LOG_LEVEL = logging.INFO
CACHE_PATH = "./cache/"

def main():
    """Main function to run the Streamlit app."""
    # Skip essential startup if already initialized
    if "initialized" not in st.session_state:
        # Run the startup sequence
        startup_sequence()
    # Display welcome message
    st.title("Welcome to U-Sleep Rescoring Tool.")
    st.info("This tool allows you to rescore low confidence periods from an autonomous scoring neural network.")
    # Choose data to download and rescore. Maybe move logic to seperate page.
    st.subheader("Choose data to download and rescore.")
    client, filenames = get_connection()
    if not client:
        st.error("Error connecting to Radboud Data Repository. Please check your credentials.")
        return
    elif not filenames:
        st.warning("No files found in the Radboud Data Repository. Please check your connection.")
        return
    subject_ids = get_unique_subject_ids(filenames)
    idx = None
    if st.session_state.get("subject_id") in subject_ids:
        idx = subject_ids.index(st.session_state["subject_id"])
    choice_subject_id = st.selectbox(
        label="Select subject ID",
        options=subject_ids,
        index=idx,
        help="Select the subject ID to rescore.",
    )

    # Only update session state if the selection changed
    if choice_subject_id != st.session_state.get("subject_id"):
        st.session_state["subject_id"] = choice_subject_id
        logging.info(f"Selected subject ID: {choice_subject_id}")
        st.rerun()  # Optional: force rerun to immediately reflect the change

    if choice_subject_id:
        # Download data from the repository
        dataset = download_dataset_from_repository(choice_subject_id)
        if not dataset:
            st.error("Error loading data from Radboud Data Repository. Please check your connection.")
            return
        # Store the dataset in session state
        st.session_state["dataset_downloaded"]["scoring"] = dataset["scoring"]
        st.session_state["dataset_downloaded"]["raw_obj"] = dataset["raw_obj"]
        st.session_state["dataset_downloaded"]["triggers"] = dataset.get("triggers", None)
        st.session_state["dataset_downloaded"]["subject_id"] = choice_subject_id
        st.success(f"Data for subject {choice_subject_id} loaded successfully.")
        # Process the dataset and store the processed data in session state
        dataset_processed = st.session_state["dataset_processed"]
        ## Process downloaded data. Currently only processing shared information.
        logging.info(f"Processing data for subject {choice_subject_id}.")
        ## Process for uncertain periods.
        dataset_processed["scoring"] = process_scoring_data(choice_subject_id)
        ## Clear biosignals if any; will populate after navigating to the next page.
        dataset_processed["biosignals"] = {}
        ## Update processed subject ID in session state.
        dataset_processed["subject_id"] = choice_subject_id
        # Unitialize page initilization for future runtimes.
        ## Note: If subject ID does not change then it will reiinitialize unecessarily.
        st.session_state["initialized"]["page_01"] = False
        st.session_state["initialized"]["page_02"] = False
    else:
        st.warning("Please select a subject ID to continue.")
        return

def startup_sequence():
    """Run the startup sequence for the Streamlit app."""
    # Initialize logging
    init_logging()
    logging.info("Starting Streamlit app.")
    # Create cache directory if it doesn't exist
    if not os.path.exists(CACHE_PATH):
        os.makedirs(CACHE_PATH)
    # Initialize session state variables
    init_session_state()
    # Set page configuration
    st.set_page_config(
        page_title = "U-Sleep Rescoring Tool",
        page_icon = ":sleeping:",
        layout = "wide",
        initial_sidebar_state = "expanded",
        # menu_items = {
        #     'Get Help': 'https://www.extremelycoolapp.com/help',
        #     'Report a bug': "https://www.extremelycoolapp.com/bug",
        #     'About': "# This is a header. This is an *extremely* cool app!"
        # }
    )
    # Confirm intialization
    st.session_state["initialized"]["global"] = True

def init_logging():
    """Initialize logging."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    logging.info("Logging initialized.")

def init_session_state():
    """Initialize session state."""
    # Variable to keep track of essential startup.
    if "initialized" not in st.session_state:
        st.session_state["initialized"] = {
            "global": False,   # Indicates if the global startup sequence has been completed.
            "page_01": False,    # Indicates if the startup sequence for page 01 has been completed.
            "page_02": False,    # Indicates if the startup sequence for page 02 has been completed.
            # Add more pages as needed.
        }
    # Variable to keep track of the subject id.
    if "subject_id" not in st.session_state:
        st.session_state["subject_id"] = None
    # Variable to keep track of the uploaded files.
    if "dataset_downloaded" not in st.session_state:
        st.session_state["dataset_downloaded"] = {
            "subject_id": None,     # Will populate with subject ID after data load.
            "scoring": None,        # Will populate with scoring data after data load.
            "raw_obj": None,        # Will populate with raw object after data load.
            "triggers": None,        # Will populate with triggers data after data load.
        }
    # Variable to keep track of processed data.
    if "dataset_processed" not in st.session_state:
        st.session_state["dataset_processed"] = {
            "subject_id": None,         # Will populate with subject ID after data load.
            "scoring": {},    # Will populate with processed scoring data after data load.
            "biosignals": {}, # Will populate with processed biosignals after data load.
            "wake_events": {}, # Will populate with wake events data after data load.
        }
    # Variables to keep track of the manual scoring; move to dataset_processed.
    if "dataset_rescored" not in st.session_state:
        st.session_state["dataset_rescored"] = {
            "subject_id": None,                     # Keep track of rescored subject ID.
            "scoring_manual": np.array([]),         # Initialize empty array for manual scoring.
            "scoring_manual_mask": np.array([]),    # Initialize empty array for keeping track of manual scoring.
            "wake_time_identification": {},         # Keep track of manually identified wake times.
        }
    # Variables for wake time identification; move to dataset_processed.
    if "wake_time_identification" not in st.session_state:
        st.session_state["wake_time_identification"] = {
            "subject_id": None,                     # Keep track of subject ID for wake time identification.
            "wake_time_selection": None,                 # Keep track of manually selected wake time.
            "current_wake_event": None,             # Keep track of corresponding label for manually selected wake time.
            "lights_off": None,                        # Keep track of lights off event.
        }
    # Variable to keep track of the current epoch.
    if "current_epoch" not in st.session_state:
        st.session_state["current_epoch"] = -1  # Initialize to -1 to indicate no epoch selected; will update after data load.
    # Variable to keep track of the current figure configuration.
    if "fig_config" not in st.session_state:
        st.session_state["fig_config"] = {
            "subject_id": None,         # Will populate with subject ID after data load.
            "current_epoch": -1,        # Initialize to -1 to indicate no epoch selected; will update after data load.
            "svg_paths": {
                "scoring": f"{CACHE_PATH}/fig_scoring.svg",
                "biosignals": f"{CACHE_PATH}/fig_biosignals.svg",
            },            # Will populate with figure path after data load.
            "figures": {},                  # Will populate with figure objects after data load.
            "scaling": {
                "EOG": 150,
                "EMG": 50,
                "EEG": 30,
            },                  # Will populate with figure scaling after data load.
            # "signal_properties": {},    # Will populate with channel properties after data load.
        }
    # Variable to keep track of the manual scoring filename.
    if "manual_scoring_filename" not in st.session_state:
        st.session_state["manual_scoring_filename"] = None
    # UI elements.
    if "slider_wake_choice" not in st.session_state:
        st.session_state["slider_wake_choice"] = 0  # Initialize to 0; will update based on current epoch and wake time selection.

def get_unique_subject_ids(filenames):
    """Extract unique subject IDs from filenames."""

    subject_ids = []

    for filename in filenames:
        if filename.endswith("triggers.txt"):
            continue
        parts = filename.split("_")
        subject_id = "_".join(parts[:2])
        subject_ids.append(subject_id)

    return sorted(set(subject_ids))

if __name__ == "__main__":
    main()

            