''' U-Sleep Rescoring GUI.
This tool can allow an expert to rescore low confidence time periods from an
autonomous scoring neural network. In low condifdence time periods, this tool will 
display the 30 seconds of raw data for the experiment and allow the expert to input
the correct sleep stage.

TODO
- Validate analyze_uncertain_periods logic with Sarah.
- Reorder biosignal channels [EOG x2, front EEG, ... , back EEG, EMG x2]
- Put file upload information in a container.
- Elaborate on file checks.
- Add a way to get context
- Mark rescoring periods on the scoring plot
- Add a stage 2 to find start of wake cycles.
- Autoscaling; change scale of individual channels
- Disable channels
- Score artifact
- EOG twice
- Wider


'''

import streamlit as st
import boto3
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

SELECT_AUTOSCALING = "MINMAX"
SELECT_N_UNCERTAIN_PERIOD = 5
SELECT_N_EPOCH = 3

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
    # Create mechanism to import files.
    data = sidebar_import_data()
    if not data:
        logging.error("No data loaded from S3.")
        st.warning("Please select files for upload to continue with rescoring.")
        return
    # Process uploaded files.
    subject_id = st.session_state["subject_id"]
    n_epoch = st.session_state["current_epoch"]
    data = process_data(data, subject_id, n_epoch)
    # Plot the uploaded files
    fig = draw_figure(data, n_epoch=st.session_state["current_epoch"])
    fig.savefig("usleep_rescoring.svg", bbox_inches='tight')
    st.image("usleep_rescoring.svg", caption="U-Sleep Rescoring Tool", use_container_width=False)
    # st.pyplot(fig, clear_figure=True, use_container_width=True)
    # Buttons to navigate through the data
    rewind, back, forward, fast_forward = st.columns(4)
    current_epoch = st.session_state["current_epoch"]
    uncertain_epochs = data["scoring_processed"]["n_uncertain_periods"]

    rewind.button("Rewind", key="rewind", use_container_width=True, on_click=callback_counter, args=(-2), disabled=True)
    back.button("Back", key="back", use_container_width=True, on_click=callback_counter, args=(-1,), disabled=(current_epoch == 0))
    forward.button("Forward", key="forward", use_container_width=True, on_click=callback_counter, args=(+1,))
    fast_forward.button("Fast forward", key="fast_forward", use_container_width=True, on_click=callback_counter, args=(+2), disabled=True)

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
    if "current_epoch" not in st.session_state:
        st.session_state["current_epoch"] = 0
    
@st.cache_resource
def get_s3_connection():
    """Initialize AWS session state."""
    # Load credentials from secrets.toml
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        region_name=st.secrets["AWS_DEFAULT_REGION"]
    )
    bucket_name = st.secrets["AWS_S3_BUCKET_NAME"]
    # List directories in the bucket
    response = s3_client.list_objects_v2(Bucket=bucket_name, Delimiter="/")
    bucket_directories = [prefix["Prefix"] for prefix in response.get("CommonPrefixes", [])]
    if not bucket_directories:
        logging.error("Bucket is empty.")
        return None, None, None
    return s3_client, bucket_name, bucket_directories

@st.cache_data
def load_data_from_s3(folder_name):
    """Load data from S3 bucket."""
    # Get the S3 client and bucket name
    s3_client, bucket_name, _ = get_s3_connection()
    # List files in the selected folder
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)
    files = [content["Key"] for content in response.get("Contents", [])]
    # Process uploaded files
    data = {}
    for file_key in files:
        # Fetch the file from S3
        obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        # Read the file content according to its type
        if file_key.endswith(".npy"):
            logging.info(f"Attempting to load scoring file: {file_key}")
            data["scoring"] = np.load(BytesIO(obj["Body"].read()))
        elif file_key.endswith(".edf"):
            logging.info(f"Attempting to load PSG file: {file_key}")
            # Read the EDF file using MNE
            with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
                tmp_file.write(obj["Body"].read())
                tmp_file.flush()  # Ensure the file is written
                # Read the EDF file using MNE
                data["raw_obj"] = mne.io.read_raw_edf(tmp_file.name, preload=False, verbose=False)
        else:
            logging.warning(f"File {file_key} is not a valid EDF or NPY file.")
    return data

def sidebar_import_data():
    """Sidebar for importing data."""
    # Initialize AWS client
    s3_client, _, bucket_directories = get_s3_connection()
    if not s3_client:
        logging.error("Failed to connect to AWS S3.")
        st.sidebar.error("Error connecting to AWS.")
        return None
    logging.info("Connected to AWS S3 successfully.")

    # Populate sidebar with file upload options
    with st.sidebar:
        st.header("Import Data")
        folder_choice = st.selectbox("Select folder", bucket_directories, index=None)
    
    if folder_choice:
        logging.info(f"Selected folder: {folder_choice}")
        # Load data from the selected folder
        data = load_data_from_s3(folder_choice)
        if not data:
            logging.error("No files found in the selected folder.")
            st.sidebar.error("Error connecting to AWS.")
            return None
        # Check if files are valid
        if "raw_obj" not in data:
            logging.error("No raw data uploaded.")
            st.sidebar.error("Error loading data from AWS.")
            return None
        elif "scoring" not in data:
            logging.error("No raw data uploaded.")
            st.sidebar.error("Error loading data from AWS.")
            return None
        # Return data
        subject_id = folder_choice.split("/")[-2]  # Assuming folder structure is like 'bucket/subject_id/'
        st.session_state["subject_id"] = subject_id
        logging.info(f"Data loaded successfully for subject {subject_id}.")
        return data
    else:
        st.sidebar.warning("Please select a folder.")
        return None

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
    # # Downsample the signals
    # signals_downsampled = decimate(signals, downsample_ratio, axis=1)
    # signals_downsampled = decimate(signals_downsampled, downsample_ratio, axis=1)
    # time_downsampled = np.arange(signals_downsampled.shape[1]) / (fs_biosignals / downsample_ratio)
    # Create a dictionary to store the processed data
    processed_data = {
        "raw_obj_cropped": raw_selection,
        "fs": fs,
    }
    return processed_data

def process_data(data, subject_id, n_epoch:int):
    # Process U-Sleep scoring data
    # data = _data.copy()
    if "scoring_processed" not in data:
        data["scoring_processed"] = analyze_uncertain_periods(data["scoring"])
    else:
        logging.warning("Scoring data already processed. Skipping reprocessing.")
    # Process biosignals for visualization
    if "processed_biosignals" not in data:
        data["processed_biosignals"] = process_biosignals(data["raw_obj"], n_epoch, downsample_ratio=10)
    else:
        logging.warning("Biosignals data already processed. Skipping reprocessing.")
    # Return processed data
    return data

def draw_figure(data, n_epoch: int = 0, auto_scaling: str = "MINMAX"):
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
        # Autoscaling
        if auto_scaling == "MINMAX":
            c_minmax = signal.max() - signal.min()
            signal /= c_minmax
        elif auto_scaling == "RMS":
            c_rms = np.sqrt(np.mean(signal**2))
            signal /= c_rms
        ax_bottom.plot(time, signal + idx, linewidth=0.5)
    # Set y-ticks and labels
    ax_bottom.set_yticks(range(signals.shape[0]), ch_labels)
    ax_bottom.set_ylim(signals.shape[0], -1)
    ax_bottom.set_xlabel("Time (s)")
    return fig
    
def callback_counter(action_type: int):
    """Callback function for button clicks."""
    # Get the current epoch from session state
    if action_type == -1: # Rewind
        current_epoch = st.session_state["current_epoch"] - 1
    elif action_type == +1: # Forward
        current_epoch = st.session_state["current_epoch"] + 1
    # Update the current epoch in session state
    st.session_state["current_epoch"] = current_epoch
    # Update the figure in the Streamlit app
    # st.session_state["fig"] = fig
    # st.pyplot(fig)

if __name__ == "__main__":
    # Initialize logging
    init_logging()
    # Run the main function
    main()

            