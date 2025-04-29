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
import numpy as np
import mne
import pyedflib
import tempfile
from scipy import ndimage
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
    # Create mechanism to import files.
    data = sidebar_import_data()
    # Check if data is None
    if data is None:
        st.warning("Please upload files to continue with rescoring.")
    else:
        # Process uploaded files
        data["scoring_processed"] = analyze_uncertain_periods(data["scoring"])
        # Plot the uploaded files
        fig = draw_figure_init(data)
        # # Process uploaded files
        # uncertain_info = analyze_uncertain_periods(scoring, scoring_time_hrs)
        # naive_scoring = np.argmax(scoring, axis=1)
        # # Look through uncertain periods
        # for uncertain_period in uncertain_info["uncertain_periods"]:
        #     slice_start = int(uncertain_period["start_hour"]*3600)*FS_PSG
        #     slice_stop = int(slice_start + 30)*FS_PSG
        #     visible_time_slice = np.s_[slice_start:slice_stop]
        #     break
        # # Plot figures
        # fig = draw_figure(data_scoring=naive_scoring, data_biosignals=raw_obj,
        #                 time_slice=visible_time_slice)
        st.pyplot(fig)
        # Buttons to navigate through the data
        rewind, back, forward, fast_forward = st.columns(4)
        rewind.button("Rewind", key="rewind", use_container_width=True)
        back.button("Back", key="back", use_container_width=True)
        forward.button("Forward", key="forward", use_container_width=True)
        fast_forward.button("Fast forward", key="fast_forward", use_container_width=True)

def logging_init():
    """Initialize logging."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    logging.info("Logging initialized.")

def sidebar_import_data():
    """Sidebar for importing data."""
    # Populate sidebar with file upload options
    with st.sidebar:
        st.header("Import Data")
        st.write("The raw data should be in EDF format, and the U-Sleep scoring should be in NPY format.")
        st.write("Please ensure that both files are from the same experiment.")
        uploaded_files = st.file_uploader(label="Upload raw data and U-Sleep scoring.",
            type=[".npy", ".edf"],
            accept_multiple_files=True,
            label_visibility="visible",
        )
    # Check if files are uploaded
    if uploaded_files == []:
        return None
    else:
        logging.info("Files uploaded.")
    # Import uploaded files
    unpacked_data = {
        "raw_obj": None,
        "scoring": None,
    }
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        if filename.endswith(".npy"):
            logging.debug(f"Found scoring file: {filename}")
            unpacked_data["scoring"] = np.load(uploaded_file)
            # Every data point corresponds to a 30 second epoch; convert to hours.
            scoring_time_hrs = np.arange(unpacked_data["scoring"].shape[0]) * 30 / 3600 
            scoring_duration = scoring_time_hrs[-1]
            logging.info(f"Loaded scoring file labeled {filename} with " + \
                    f"{scoring_duration:.0f} hours and {scoring_duration%1*60:.0f} minutes.")
        elif filename.endswith(".edf"):
            logging.info(f"Found raw data file: {filename}")
            unpacked_data["raw_obj"] = import_edf_file(uploaded_file)

            ch_labels = unpacked_data["raw_obj"].ch_names
            _, time_sec = unpacked_data["raw_obj"][:]
            logging.info(f"Loaded PSG file with the following signals: {ch_labels} and a duration of " + \
                    f"{time_sec[-1]/3600:.0f} hours and {time_sec[-1]/3600%1*60:.0f} minutes.")
    # Check if both files are uploaded
    if unpacked_data is None:
        logging.warning("No files uploaded.")
        st.sidebar.warning("Please upload files to continue with rescoring.")
        return None
    elif unpacked_data["raw_obj"] is None:
        logging.warning("No raw data uploaded.")
        st.sidebar.warning("Please upload PSG data.")
        return None
    elif unpacked_data["scoring"] is None:
        logging.warning("No scoring data uploaded.")
        st.sidebar.warning("Please scoring file.")
        return None
    else:
        # Display data
        st.sidebar.success("Files uploaded successfully.")
        return unpacked_data

def import_edf_file(uploaded_file):
    """Import EDF file."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as temp_file:
        temp_file.write(uploaded_file.read())  # Write the uploaded file content
        temp_file_path = temp_file.name  # Get the file path

    # Read the EDF file using MNE
    raw_obj = mne.io.read_raw_edf(temp_file_path, preload=True)
    return raw_obj

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

def draw_figure_init(data_dict: Dict):
    '''
    INPUTS
    period_scoring (int) - scoring sampling period in seconds.
    '''
    # Create the figure and GridSpec layout
    fig = plt.figure(figsize=(18, 7), dpi=300)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 5])  # 1 unit for top, 5 for bottom

    # Create the top and bottom axes
    ax_top = fig.add_subplot(gs[0])
    ax_bottom = fig.add_subplot(gs[1])

    # Top plot. Algorithm scoring output.
    scoring_data = data_dict["scoring_processed"]
    scoring_naive = scoring_data["scoring_naive"]
    time = scoring_data["time_hrs"]
    ax_top.plot(time, scoring_naive, linewidth=1, color="grey")
    # Highlight uncertain periods.
    uncertain_periods = scoring_data["uncertain_periods"]
    for uncertain_period in uncertain_periods:
        start_hour = uncertain_period["start_hour"]
        end_hour = uncertain_period["end_hour"] + 30 / 3600
        ax_top.fill_betweenx(y=[0, 5], x1=start_hour, x2=end_hour, color="red", alpha=0.3)
    
    ax_top.set_yticks(ticks=range(5), labels=SLEEP_STAGE_LABELS)
    ax_top.set_ylim(4.25,-0.25)
    ax_top.set_xlabel("Time (hrs)")
    ax_top.set_ylabel("Sleep stages")

    # # Bottom plot. Raw data display.
    # # data_biosignals = get_sorted_biosignals(data_biosignals)
    # # signals, time = data_biosignals[:]
    # # ch_labels = data_biosignals.ch_names
    # signals, time, ch_labels = get_sorted_biosignals(data_biosignals)
    # for idx, signal in enumerate(signals[:,time_slice]):
    #     # autoscaling 
    #     if SELECT_AUTOSCALING == "MINMAX":
    #         c_minmax = signal.max() - signal.min()
    #         signal /= c_minmax
    #     elif SELECT_AUTOSCALING == "RMS":
    #         c_rms = np.sqrt(np.mean(signal**2))
    #         signal /= c_rms
    #     ax_bottom.plot(time[time_slice], signal + idx, linewidth=0.75)  # Blue diagonal line
    # ax_bottom.set_yticks(range(signals.shape[0]), ch_labels);
    # ax_bottom.set_ylim(signals.shape[0], -1)
    # ax_bottom.set_xlabel("Time (s)")

    # Adjust layout
    plt.tight_layout()
    
    return fig

def draw_figure_update(fig, slice_window):
    axs = fig.get_axes()
    ax_top = axs[0]
    ax_top.fill_betweenx(y=[0, 5], x1=slice_window.start/3600/30, x2=slice_window.stop/3600/30, color="red", alpha=0.3)
    st.pyplot(fig)

def get_sorted_biosignals(mne_raw_obj):
    ''' Reorder channels to EOG x2, front EEG, ... , back EEG, EMG x2 '''
    # Load channel names
    current_order = mne_raw_obj.ch_names
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
    mne_raw_obj = mne_raw_obj.pick(desired_order)
    # Get data
    signals, time = mne_raw_obj[:]
    ch_labels = mne_raw_obj.ch_names
    # Duplicate EOG
    signals = np.vstack((signals[0], signals))
    ch_labels = ["EOG"] + ch_labels
    # Return
    return signals, time, ch_labels

if __name__ == "__main__":
    # Initialize logging
    logging_init()
    # Run the main function
    main()


# slice_start = int(uncertain_info["uncertain_periods"][SELECT_N_UNCERTAIN_PERIOD]["start_hour"]*3600 + SELECT_N_EPOCH*30)*500
# slice_stop = slice_start + 30*500
# visible_time_slice = np.s_[slice_start:slice_stop]

# plt.figure(figsize=(15,6), dpi=300)
# for idx, signal in enumerate(data):
#     # (x,y) correspond to the visibile signal for plotting.
#     x = time_sec[visible_time_slice]    
#     y = signal[visible_time_slice]

#     # autoscaling 
#     y_rms = np.sqrt(np.mean(y**2))

#     if SELECT_AUTOSCALING == "MINMAX":
#         c_minmax = y.max() - y.min()
#         y /= c_minmax
#     elif SELECT_AUTOSCALING == "RMS":
#         c_rms = np.sqrt(np.mean(y**2))
#         y /= c_rms
    
#     plt.plot(x, y + idx, linewidth=0.75)
# plt.yticks(range(data.shape[0]), ch_labels);
            