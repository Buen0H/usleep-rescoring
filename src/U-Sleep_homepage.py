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
from streamlit_shortcuts import shortcut_button, add_shortcuts
from webdav3.client import Client
import os
import numpy as np
import mne
from scipy import ndimage
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging

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
        st.session_state["dataset_downloaded"]["subject_id"] = choice_subject_id
        st.success(f"Data for subject {choice_subject_id} loaded successfully.")
    else:
        st.warning("Please select a subject ID to continue.")
        return

def main_old():
    """Main function to run the Streamlit app."""
    logging.info("Starting Streamlit app.")
    # Set the title of the app
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
    st.title("U-Sleep Rescoring Tool")
    # Initialize session state
    init_session_state()
    fig_config = st.session_state["fig_config"]
    # Create mechanism to import files.
    data = sidebar_import_data()
    if not data:
        logging.error("No data loaded from remote database.")
        st.warning("Please select files for upload to continue with rescoring.")
        return
    # Process uploaded files.
    subject_id = st.session_state["subject_id"]
    data = process_scoring_data(data, subject_id)
    if st.session_state["scoring_manual"].size == 0:
        st.session_state["scoring_manual"] = data["scoring_processed"]["scoring_naive"].copy() # Initialize manual scoring with naive scoring.
        st.session_state["scoring_manual_mask"] = np.zeros_like(st.session_state["scoring_manual"], dtype=bool)  # Initialize mask for manual scoring.
    # Handle the case where no epoch is selected.
    if fig_config["current_epoch"] == -1:
        # If no epoch is selected, set the current epoch to the first uncertain period
        if data["scoring_processed"]["n_uncertain_periods"] == 0:
            fig_config["current_epoch"] = 0
            logging.warning("No uncertain periods found in the data.")
            st.warning("No uncertain periods found in the data. Please select a different file.")
        else:
            first_uncertain_hour = data["scoring_processed"]["uncertain_periods"][0]["start_hour"]
            first_uncertain_epoch = int(first_uncertain_hour * 3600 / 30)  # Convert hours to epochs.
            fig_config["current_epoch"] = first_uncertain_epoch
    n_epoch = fig_config["current_epoch"]
    data = process_series_data(data, subject_id, n_epoch)
    # Plot the uploaded files
    # Draw the figure with the current epoch
    fig = draw_figure(data, n_epoch)
    fig.savefig("cache/usleep_rescoring.svg", bbox_inches='tight')
    st.image("cache/usleep_rescoring.svg", use_container_width=False)

    # with control_panel:
    #     raw_obj = data["processed_biosignals"]["raw_obj_cropped"]
    #     _, _, ch_labels = get_sorted_biosignals(raw_obj)
    #     for ch_label in ch_labels[1:]:  # Skip the first channel (EOG)
    #         label_col, visible_col, range_col = st.columns(3)
    #         # Label column
    #         label_col.markdown(f"{ch_label}")
    #         # Visibility control
    #         visible_col.checkbox(
    #             label="Visible",
    #             value=True,
    #             key=f"visible_{ch_label}",
    #             label_visibility="collapsed",
    #         )
    #         # Range control
    #         range_col.segmented_control(
    #             options=[":heavy_minus_sign:", ":heavy_plus_sign:"],
    #             label=ch_label,
    #             selection_mode="single",
    #             key=f"control_{ch_label}",
    #             default=None,
    #             label_visibility="collapsed",           )

    # Variables with the current epoch and previous/next epochs.
    current_epoch = fig_config["current_epoch"]
    previous_epoch = current_epoch - 1
    next_epoch = current_epoch + 1
    previous_uncertain_epoch = find_closest_uncertain_period(data, current_epoch, direction=False)
    next_uncertain_epoch = find_closest_uncertain_period(data, current_epoch, direction=True)

    # Mechanism to grade the uncertaion periods.
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("Wake", key="wake", use_container_width=True,):
            st.session_state["scoring_manual"][current_epoch] = 0  # Wake
            st.session_state["scoring_manual_mask"][current_epoch] = True  # Mark as manually scored
            st.session_state["fig_config"]["current_epoch"] = next_epoch    # Update current epoch to next one
            logging.info(f"Wake button selected. Current epoch: {current_epoch}")
            st.rerun()  # Force rerun to update UI
    with col2:
        if st.button("REM", key="rem", use_container_width=True):
            st.session_state["scoring_manual"][current_epoch] = 1  # REM
            st.session_state["scoring_manual_mask"][current_epoch] = True  # Mark as manually scored
            st.session_state["fig_config"]["current_epoch"] = next_epoch    # Update current epoch to next one
            logging.info(f"REM button selected. Current epoch: {current_epoch}")
            st.rerun()  # Force rerun to update UI
    with col3:
        if st.button("N1", key="n1", use_container_width=True):
            st.session_state["scoring_manual"][current_epoch] = 2  # N1
            st.session_state["scoring_manual_mask"][current_epoch] = True  # Mark as manually scored
            st.session_state["fig_config"]["current_epoch"] = next_epoch    # Update current epoch to next one
            logging.info(f"N1 button selected. Current epoch: {current_epoch}")
            st.rerun()  # Force rerun to update UI
    with col4:
        if st.button("N2", key="n2", use_container_width=True):
            st.session_state["scoring_manual"][current_epoch] = 3  # N2
            st.session_state["scoring_manual_mask"][current_epoch] = True # Mark as manually scored
            st.session_state["fig_config"]["current_epoch"] = next_epoch    # Update current epoch to next one
            logging.info(f"N2 button selected. Current epoch: {current_epoch}")
            st.rerun()  # Force rerun to update UI
    with col5:
        if st.button("N3", key="n3", use_container_width=True):
            st.session_state["scoring_manual"][current_epoch] = 4  # N3
            st.session_state["scoring_manual_mask"][current_epoch] = True  # Mark as manually scored
            st.session_state["fig_config"]["current_epoch"] = next_epoch    # Update current epoch to next one
            logging.info(f"N3 button selected. Current epoch: {current_epoch}")
            st.rerun()  # Force rerun to update UI
    
    # Mechanism to navigate through recording.
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Rewind", key="rewind", use_container_width=True):
            st.session_state["fig_config"]["current_epoch"] = previous_uncertain_epoch
            logging.info(f"Rewind button selected. Current epoch: {previous_uncertain_epoch}")
            st.rerun()  # Force rerun to update UI
    with col2:
        if st.button("Back", key="back", use_container_width=True, disabled=(current_epoch == 0)):
            st.session_state["fig_config"]["current_epoch"] = previous_epoch
            logging.info(f"Back button selected. Current epoch: {previous_epoch}")
            st.rerun()  
    with col3:
        if st.button("Forward", key="forward", use_container_width=True):
            st.session_state["fig_config"]["current_epoch"] = next_epoch
            logging.info(f"Forward button selected. Current epoch: {next_epoch}")
            st.rerun()
    with col4:
        if st.button("Fast forward", key="fast_forward", use_container_width=True):
            st.session_state["fig_config"]["current_epoch"] = next_uncertain_epoch
            logging.info(f"Fast forward button selected. Current epoch: {next_uncertain_epoch}")
            st.rerun()

    # Add shortcuts for the buttons
    add_shortcuts(
        wake="0",
        rem="4",
        n1="1",
        n2="2",
        n3="3",
        rewind="arrowdown",
        back="arrowleft",
        forward="arrowright",
        fast_forward="arrowup",
    )

    # Handle manual scoring file management.
    # Save the manual scoring data to a file.
    np.save(CACHE_PATH + f"{subject_id}_scoring_manual.npy", st.session_state["scoring_manual"])
    with open(CACHE_PATH + f"{subject_id}_scoring_manual.npy", "rb") as f:
        st.download_button(
            label="Download manually graded data",
            data=f,
            file_name=f"{subject_id}_scoring_manual.npy",
            mime="application/octet-stream",
            use_container_width=True,
        )
    # # Button to upload files to the repository.
    # if st.button("Upload file to repository", use_container_width=True):
    #     logging.info("Upload file to repository button clicked.")
    #      upload_file_to_repository(CACHE_PATH + f"{subject_id}_scoring_manual.npy")

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
    st.session_state["initialized"] = True

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
        st.session_state["initialized"] = False
    # Variable to keep track of the subject id.
    if "subject_id" not in st.session_state:
        st.session_state["subject_id"] = None
    # Variable to keep track of the uploaded files.
    if "dataset_downloaded" not in st.session_state:
        st.session_state["dataset_downloaded"] = {
            "subject_id": None,     # Will populate with subject ID after data load.
            "scoring": None,        # Will populate with scoring data after data load.
            "raw_obj": None,        # Will populate with raw object after data load.
        }
    # Variable to keep track of processed data.
    if "dataset_processed" not in st.session_state:
        st.session_state["dataset_processed"] = {
            "subject_id": None,         # Will populate with subject ID after data load.
            "scoring_processed": {},    # Will populate with processed scoring data after data load.
            "processed_biosignals": {}, # Will populate with processed biosignals after data load.
        }
    # Variables to keep track of the manual scoring.
    if "dataset_rescored" not in st.session_state:
        st.session_state["dataset_rescored"] = {
            "subject_id": None,                     # Keep track of rescored subject ID.
            "scoring_manual": np.array([]),         # Initialize empty array for manual scoring.
            "scoring_manual_mask": np.array([]),    # Initialize empty array for keeping track of manual scoring.
        }
    # Variable to keep track of the current epoch.
    if "current_epoch" not in st.session_state:
        st.session_state["current_epoch"] = -1  # Initialize to -1 to indicate no epoch selected; will update after data load.
    # Variable to keep track of the current figure configuration.
    if "fig_config" not in st.session_state:
        st.session_state["fig_config"] = {
            "subject_id": None,         # Will populate with subject ID after data load.
            "svg_paths": {
                "scoring": f"{CACHE_PATH}/fig_scoring.svg",
            },            # Will populate with figure path after data load.
            "figures": {},              # Will populate with figure objects after data load.
            # "current_epoch": -1,        # Initialize to -1 to indicate no epoch selected; will update after data load.
            # "raw_obj_selection": None,  # Will populate with raw object selection after data load.
            # "fs": None,                 # Will populate with sampling frequency after data load.
            "signal_properties": {},    # Will populate with channel properties after data load.
        }

@st.cache_resource
def get_connection():
    # Configuration for RDR WebDAV access
    options = {
        'webdav_hostname': st.secrets["RDR_HOSTNAME"],
        'webdav_login':    st.secrets["RDR_USERNAME"],
        'webdav_password': st.secrets["RDR_PASSWORD"],
    }
    # Initialize the WebDAV client
    rdr_client = Client(options)
    if not rdr_client:
        logging.error("Connection error.")
        return None, None
    # Get file list from remote directory
    remote_path = f"{st.secrets['RDR_REMOTE_PATH']}/"
    filenames = rdr_client.list(remote_path)
    return rdr_client, filenames

def get_unique_subject_ids(filenames):
    """Extract unique subject IDs from filenames."""
    return sorted(set(["_".join(filename.split("_")[:2]) for filename in filenames]))

@st.cache_data
def download_dataset_from_repository(experiment_name: str):
    """Load data from the Radboud Data Repository."""
    # Get webDAV client and filenames
    client, filenames = get_connection()
    # Find valid files for the selected experiment
    experiment_files = [filename for filename in filenames if filename.startswith(experiment_name) and (filename.endswith(".npy") or filename.endswith(".edf"))]
    if not experiment_files:
        logging.error(f"No valid files found for experiment {experiment_name}.")
        st.sidebar.error("No valid files found for the selected experiment.")
        return None
    # Process uploaded files
    dataset = {}
    for filename in experiment_files:
        # Fetch files from Radboud Data Repository
        local_path = CACHE_PATH + filename
        remote_path = f"{st.secrets['RDR_REMOTE_PATH']}/{filename}"
        client.download(remote_path, local_path)
        # Read the file content according to its type
        if filename.endswith(".npy"):
            logging.info(f"Attempting to load scoring file: {filename}")
            dataset["scoring"] = np.load(local_path)
        elif filename.endswith(".edf"):
            logging.info(f"Attempting to load PSG file: {filename}")
            dataset["raw_obj"] = mne.io.read_raw_edf(local_path, preload=False, verbose=False)
        else:
            logging.warning(f"File {filename} is not a valid EDF or NPY file.")
    return dataset

def upload_file_to_repository(file_path: str = None):
    """Sidebar widget to upload a file to the external data repository."""
    client, _ = get_connection()
    if not client:
        st.sidebar.error("No connection to repository.")
        return

    # Define remote path
    remote_path = f"{st.secrets['RDR_REMOTE_PATH']}/{st.session_state['subject_id']}_scoring_manual.npy"

    # Upload to WebDAV
    try:
        client.upload_sync(remote_path=remote_path, local_path=file_path)
        st.sidebar.success(f"Uploaded {file_path} to repository.")
        logging.info(f"Uploaded {file_path} to repository.")
        st.success(f"File {file_path} uploaded successfully to the repository.")
    except Exception as e:
        st.error(f"Upload failed: {e}")
        logging.error(f"Upload failed: {e}")


# def sidebar_control_panel():


def analyze_uncertain_periods(confidence_data: np.ndarray) -> Dict:
    """Analyze periods of low confidence."""
    # Get the time in hours
    time_hrs = np.arange(confidence_data.shape[0]) * 30 / 3600  # 30 seconds per epoch

    max_confidences = np.max(confidence_data, axis=1)
    max_possible_conf = np.ceil(np.max(max_confidences))  # Round up to nearest whole number
    relative_threshold = 0.67  # equivalent to 4/6
    confidence_threshold = max_possible_conf * relative_threshold
    
    # Find regions of low confidence
    low_conf = max_confidences < confidence_threshold
    
    # Parameters for uncertain periods
    epoch_length = 30  # seconds
    min_duration = 3 * 60  # 3 minutes in seconds
    min_epochs = int(min_duration / epoch_length)
    
    # Find continuous regions
    labeled_regions, num_regions = ndimage.label(low_conf)
    
    # Analyze each region
    uncertain_periods = []
    for region in range(1, num_regions + 1):
        region_indices = np.where(labeled_regions == region)[0]
        duration = len(region_indices) * epoch_length  # duration in seconds
        
        if len(region_indices) >= min_epochs:
            uncertain_periods.append({
                'start_hour': time_hrs[region_indices[0]],
                'end_hour': time_hrs[region_indices[-1]],  # Is this the start of the endth hour?
                'duration_mins': duration / 60,
                'n_epochs': len(region_indices)
            })
    
    return {
        'n_uncertain_periods': len(uncertain_periods),
        'uncertain_periods': uncertain_periods,
        'total_uncertain_mins': sum(p['duration_mins'] for p in uncertain_periods),
        'confidence_threshold': confidence_threshold,
        'max_possible_conf': max_possible_conf,
        "scoring_naive": np.argmax(confidence_data, axis=1),
        "time_hrs": time_hrs,  # 30 seconds per epoch
    }

def find_closest_uncertain_period(_data: Dict, current_epoch: int, direction: bool) -> int:
    """Find the closest uncertain period to the current epoch."""
    # Get the current time in hours
    current_hour = current_epoch * 30 / 3600  # 30 seconds per epoch
    # Get the uncertain periods from scoring data
    uncertain_periods = _data["scoring_processed"]["uncertain_periods"]
    uncertain_periods_start_hrs = [period["start_hour"] for period in uncertain_periods]
    if direction:
        # If direction is True, find the next uncertain period
        uncertain_periods_start_hrs = [hour for hour in uncertain_periods_start_hrs if hour > current_hour]
        closest_period = uncertain_periods_start_hrs[0] if uncertain_periods_start_hrs else None
    else:
        # If direction is False, find the previous uncertain period
        uncertain_periods_start_hrs = [hour for hour in uncertain_periods_start_hrs if hour < current_hour]
        closest_period = uncertain_periods_start_hrs[-1] if uncertain_periods_start_hrs else None
    if closest_period is None:
        # If no uncertain period is found in the specified direction, log a warning and return the current epoch
        if direction:
            logging.warning("No uncertain period found in the specified direction (forward).")
        else:   
            logging.warning("No uncertain period found in the specified direction (rewind).")
        return current_epoch  # Return the current epoch if no uncertain period is found\
    closest_epoch = int(closest_period * 3600 / 30)  # Convert hours to epochs
    return closest_epoch

def get_sorted_biosignals(obj: mne.io.Raw) -> tuple:
    ''' Reorder channels to EOG x2, front EEG, ... , back EEG, EMG x2 '''
    # Load channel names
    current_order = obj.ch_names
    desired_order = []
    # Add EOG channels
    desired_order.append("EOG")
    # Add EEG channels
    for label in current_order:
        if "EMG" in label or "EOG" in label:
            continue
        else:
            desired_order.append(label)
    # Add both EMG channels
    desired_order.append("EMG1")
    desired_order.append("EMG2")
    # Reorder mne raw obj
    obj = obj.pick(desired_order)
    # Get data
    signals, time = obj[:]
    ch_labels = obj.ch_names
    # Duplicate EOG
    signals = np.vstack((signals[0], signals))
    ch_labels = ["EOG"] + ch_labels
    # Return
    return signals, time, ch_labels

def process_biosignals(raw_obj: mne.io.Raw, n_epoch: int, downsample_ratio: int = 5):
    """Process biosignals for visualization."""
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

@st.cache_data
def process_scoring_data(_data: Dict, subject_id: int) -> Dict:
    """Process scoring data to extract uncertain periods."""
    # Analyze uncertain periods in the scoring data
    _data["scoring_processed"] = analyze_uncertain_periods(_data["scoring"])
    return _data

@st.cache_data
def process_series_data(_data: Dict, subject_id: int, n_epoch:int):
    # Process U-Sleep scoring data
    # Process biosignals for visualization
    _data["processed_biosignals"] = process_biosignals(_data["raw_obj"], n_epoch, downsample_ratio=10)
    # Return processed data
    return _data

def plot_masked_regions(ax, mask, time, color="green", alpha=0.3):
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

def draw_figure(data, n_epoch: int = 0, auto_scaling: str = "STANDARD"):
    '''
    INPUTS
    period_scoring (int) - scoring sampling period in seconds.
    '''
    # Create the figure and GridSpec layout
    fig = plt.figure(figsize=(20, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 5])  # 1 unit for top, 5 for bottom

    # Create the top and bottom axes
    ax_top = fig.add_subplot(gs[0])
    ax_bottom = fig.add_subplot(gs[1])

    # Top plot. Algorithm scoring output.
    scoring_data = data["scoring_processed"]
    scoring_naive = scoring_data["scoring_naive"]
    time = scoring_data["time_hrs"]
    ax_top.plot(time, scoring_naive, linewidth=1, color="grey")
    # Highlight uncertain periods.
    uncertain_periods = scoring_data["uncertain_periods"]
    for uncertain_period in uncertain_periods:
        start_hour = uncertain_period["start_hour"]
        end_hour = uncertain_period["end_hour"] + 30 / 3600
        ax_top.fill_betweenx(y=[0, 5], x1=start_hour, x2=end_hour, color="red", alpha=0.3)
    # Highlight manually graded periods.
    manual_scoring_mask = st.session_state["scoring_manual_mask"]
    if np.any(manual_scoring_mask):
        plot_masked_regions(ax_top, manual_scoring_mask, time, color="green", alpha=0.3)
    # Draw line for the current epoch on the top plot.
    current_epoch = data["scoring_processed"]["time_hrs"][n_epoch]
    ax_top.axvline(x=current_epoch, color="blue", linestyle="--", linewidth=1)
    # Axes formatting.
    ax_top.set_yticks(ticks=range(5), labels=SLEEP_STAGE_LABELS)
    ax_top.set_ylim(4.25,-0.25)
    ax_top.set_xlabel("Time (hrs)")
    ax_top.set_ylabel("Sleep stages")
    plt.tight_layout()

    # Bottom plot. Raw data display.
    # Get the processed biosignals
    raw_obj = data["processed_biosignals"]["raw_obj_cropped"]
    fs = data["processed_biosignals"]["fs"]
    signals, time, ch_labels = get_sorted_biosignals(raw_obj)
    # Plot the signals
    for idx, (signal, ch_label) in enumerate(zip(signals, ch_labels)):
        # Autoscaling
        if auto_scaling == "MINMAX":
            c_minmax = signal.max() - signal.min()
            signal /= c_minmax
        elif auto_scaling == "RMS":
            c_rms = np.sqrt(np.mean(signal**2))
            signal /= c_rms
        elif auto_scaling == "STANDARD":
            if ch_label.startswith("EOG"):
                scale_val = 150
            elif ch_label.startswith("EMG"):
                scale_val = 50
            else:
                scale_val = 30
            c_range = 2 * scale_val 
            signal *= 1e6 # Convert to microvolts
            signal /= c_range
        ax_bottom.plot(time, signal + idx, linewidth=0.5)
    # Set y-ticks and labels
    ax_bottom.set_yticks(range(signals.shape[0]), ch_labels)
    ax_bottom.set_ylim(signals.shape[0], -1)
    ax_bottom.set_xlabel("Time (s)")
    return fig
    
# def callback_counter(current_epoch: int):
#     """Callback function for button clicks."""
#     # Update the current epoch in session state
#     st.session_state["fig_config"]["current_epoch"] = current_epoch
#     # Update the figure in the Streamlit app
#     # st.session_state["fig"] = fig
#     # st.pyplot(fig)

if __name__ == "__main__":
    main()

            