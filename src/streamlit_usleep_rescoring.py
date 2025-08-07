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
from io import BytesIO
import numpy as np
import mne
import tempfile
from scipy import ndimage
from scipy.signal import decimate
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging

LOG_LEVEL = logging.INFO

CACHE_PATH = "./cache/"

SLEEP_STAGE_LABELS = ["Wake", "REM", "N1", "N2", "N3"] 
FS_PSG = 500 # get from edf file instead.  

def main():
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
    st.session_state["scoring_manual"] = data["scoring_processed"]["scoring_naive"] # Initialize manual scoring with naive scoring.
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

    # Mechanism to grade the uncertaion periods.
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.button("Wake", key="wake", use_container_width=True,)
    with col2:
        st.button("REM", key="rem", use_container_width=True)
    with col3:
        st.button("N1", key="n1", use_container_width=True)
    with col4:
        st.button("N2", key="n2", use_container_width=True)
    with col5:
        st.button("N3", key="n3", use_container_width=True)
    
    # Mechanism to navigate through recording.
    current_epoch = fig_config["current_epoch"]
    previous_epoch = current_epoch - 1
    next_epoch = current_epoch + 1
    previous_uncertain_epoch = find_closest_uncertain_period(data, current_epoch, direction=False)
    next_uncertain_epoch = find_closest_uncertain_period(data, current_epoch, direction=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Rewind", key="rewind", use_container_width=True, 
                     on_click=callback_counter, args=(previous_uncertain_epoch, )):
            logging.info(f"Rewind button selected. Current epoch: {previous_uncertain_epoch}")
    with col2:
        if st.button("Back", key="back", use_container_width=True, 
                     on_click=callback_counter, args=(previous_epoch, ), 
                     disabled=(current_epoch == 0)):
            logging.info(f"Back button selected. Current epoch: {previous_epoch}")
    with col3:
        if st.button("Forward", key="forward", use_container_width=True, 
                     on_click=callback_counter, args=(next_epoch, )):
            logging.info(f"Forward button selected. Current epoch: {next_epoch}")
    with col4:
        if st.button("Fast forward", key="fast_forward", use_container_width=True, 
                     on_click=callback_counter, args=(next_uncertain_epoch, )):
            logging.info(f"Fast forward button selected. Current epoch: {next_uncertain_epoch}")
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

    # wake.button("Wake", key="wake", use_container_width=True)
    # rem.button("REM", key="rem", use_container_width=True)
    # n1.button("N1", key="n1", use_container_width=True)
    # n2.button("N2", key="n2", use_container_width=True)
    # n3.button("N3", key="n3", use_container_width=True)

    # Mechanism to navigate through recording.
    # rewind, back, forward, fast_forward = st.columns(4)
    # current_epoch = fig_config["current_epoch"]
    # with back:
    #     if shortcut_button(label="Back", shortcut="arrowleft", hint=False, key="back",
    #                        use_container_width=True, on_click=callback_counter, 
    #                        args=(current_epoch - 1, ), disabled=(current_epoch == 0)):
    #         logging.info(f"Back button clicked. Current epoch: {current_epoch - 1}")
    # with forward:
    #     if shortcut_button(label="Forward", shortcut="arrowright", hint=False, key="forward",
    #                        use_container_width=True, on_click=callback_counter, 
    #                        args=(current_epoch + 1, )):#, disabled=(current_epoch >= data["scoring_processed"]["n_uncertain_periods"])):
    #         logging.info(f"Forward button clicked. Current epoch: {current_epoch + 1}")
    # rewind.button("Rewind", key="rewind", use_container_width=True, on_click=callback_counter, 
    #             args=(find_closest_uncertain_period(data, current_epoch, direction=False), ))
    # back.button("Back", key="back", use_container_width=True, on_click=callback_counter, 
    #             args=(current_epoch - 1, ), disabled=(current_epoch == 0))
    # forward.button("Forward", key="forward", use_container_width=True, on_click=callback_counter, 
    #             args=(current_epoch + 1, ))#, disabled=(current_epoch >= data[""])
    # fast_forward.button("Fast forward", key="fast_forward", use_container_width=True, on_click=callback_counter, 
    #             args=(find_closest_uncertain_period(data, current_epoch, direction=True), ))

    # Add shortcuts for navigation
    # add_shortcuts(
    #     back="arrowleft",
    #     forward="arrowright",
    #     rewind="arrowdown",
    #     fast_forward="arrowup",
    # )


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
    if "subject_id" not in st.session_state:
        st.session_state["subject_id"] = None
    if not os.path.exists(CACHE_PATH):
        os.makedirs(CACHE_PATH)
    if "fig_config" not in st.session_state:
        st.session_state["fig_config"] = {
            "current_epoch": -1,        # Initialize to -1 to indicate no epoch selected; will update after data load.
            "raw_obj_selection": None,  # Will populate with raw object selection after data load.
            "fs": None,                 # Will populate with sampling frequency after data load.
            "signal_properties": {},    # Will populate with channel properties after data load.
        }
    if "scoring_manual" not in st.session_state:
        st.session_state["scoring_manual"] = np.array([])  # Initialize empty array for manual scoring.

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
def load_data_from_connection(experiment_name: str):
    """Load data from the Radboud Data Repository."""
    # Get the S3 client and bucket name
    client, filenames = get_connection()
    # Find valid files for the selected experiment
    experiment_files = [filename for filename in filenames if filename.startswith(experiment_name) and (filename.endswith(".npy") or filename.endswith(".edf"))]
    if not experiment_files:
        logging.error(f"No valid files found for experiment {experiment_name}.")
        st.sidebar.error("No valid files found for the selected experiment.")
        return None
    else:
        logging.info(f"Found {len(experiment_files)} files for experiment {experiment_name}.")
        logging.info(f"Files: {experiment_files}")
        st.sidebar.info(f"Found {len(experiment_files)} files for experiment {experiment_name}.")
        st.sidebar.info(f"Files: {experiment_files}")
    # Process uploaded files
    data = {}
    for filename in experiment_files:
        # Fetch the file from S3
        local_path = CACHE_PATH + filename
        remote_path = f"{st.secrets['RDR_REMOTE_PATH']}/{filename}"
        client.download(remote_path, local_path)
        # Read the file content according to its type
        if filename.endswith(".npy"):
            logging.info(f"Attempting to load scoring file: {filename}")
            data["scoring"] = np.load(local_path)
        elif filename.endswith(".edf"):
            logging.info(f"Attempting to load PSG file: {filename}")
            data["raw_obj"] = mne.io.read_raw_edf(local_path, preload=False, verbose=False)
        else:
            logging.warning(f"File {filename} is not a valid EDF or NPY file.")
    return data

def sidebar_import_data():
    """Sidebar for importing data."""
    # Initialize AWS client
    client, filenames = get_connection()
    if not client:
        logging.error("Failed to connect to Radboud Data Repository.")
        st.sidebar.error("Error with connection.")
        return None
    logging.info("Connected established.")
    # Filter for subject ids.
    experiment_ids = get_unique_subject_ids(filenames)
    # Populate sidebar with file upload options
    with st.sidebar:
        st.header("Import Data")
        experiment_choice = st.selectbox("Select experiment", experiment_ids, index=None)
    if experiment_choice:
        logging.info(f"Selected folder: {experiment_choice}")
        # Load data from the selected folder
        data = load_data_from_connection(experiment_choice)
        if not data:
            logging.error("No files found in the selected folder.")
            st.sidebar.error("Error connecting to RDR.")
            return None
        # Check if files are valid
        if "raw_obj" not in data:
            logging.error("No raw data uploaded.")
            st.sidebar.error("Error loading data from RDR.")
            return None
        elif "scoring" not in data:
            logging.error("No raw data uploaded.")
            st.sidebar.error("Error loading data from RDR.")
            return None
        # Return data
        st.session_state["subject_id"] = experiment_choice
        logging.info(f"Data loaded successfully for subject {experiment_choice}.")
        return data
    else:
        st.sidebar.warning("Please select a folder.")
        return None

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

def draw_figure(data, n_epoch: int = 0, auto_scaling: str = "STANDARD", scale_val = 30):
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
    for idx, signal in enumerate(signals):
        logging.info(f"Range of signal {idx}: {np.min(signal)} to {np.max(signal)} {np.std(signal)}")
        # Autoscaling
        if auto_scaling == "MINMAX":
            c_minmax = signal.max() - signal.min()
            signal /= c_minmax
        elif auto_scaling == "RMS":
            c_rms = np.sqrt(np.mean(signal**2))
            signal /= c_rms
        elif auto_scaling == "STANDARD":
            c_range = 2 * scale_val 
            signal *= 1e6 # Convert to microvolts
            signal /= c_range
        ax_bottom.plot(time, signal + idx, linewidth=0.5)
    # Set y-ticks and labels
    ax_bottom.set_yticks(range(signals.shape[0]), ch_labels)
    ax_bottom.set_ylim(signals.shape[0], -1)
    ax_bottom.set_xlabel("Time (s)")
    return fig
    
def callback_counter(current_epoch: int):
    """Callback function for button clicks."""
    # Update the current epoch in session state
    st.session_state["fig_config"]["current_epoch"] = current_epoch
    # Update the figure in the Streamlit app
    # st.session_state["fig"] = fig
    # st.pyplot(fig)

if __name__ == "__main__":
    # Initialize logging
    init_logging()
    # Run the main function
    main()

            