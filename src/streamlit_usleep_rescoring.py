''' U-Sleep Rescoring GUI.
This tool can allow an expert to rescore low confidence time periods from an
autonomous scoring neural network. In low condifdence time periods, this tool will 
display the 30 seconds of raw data for the experiment and allow the expert to input
the correct sleep stage.

TODO
- Confirm [0,1,2,3,4] -> [Wake REM N1 N2 N3]
- Validate analyze_uncertain_periods logic with Sarah.
- Reorder biosignal channels [EOG x2, front EEG, ... , back EEG, EMG x2]
- Put file upload information in a container.
- Elaborate on file checks.


'''

import streamlit as st
import numpy as np
import mne
import tempfile
from scipy import ndimage
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

SELECT_AUTOSCALING = "MINMAX"
SELECT_N_UNCERTAIN_PERIOD = 5
SELECT_N_EPOCH = 3

SLEEP_STAGE_LABELS = ["Wake", "REM", "N1", "N2", "N3"] 
FS_PSG = 500 # get from edf file instead.  

def main():
    print(*reversed(SLEEP_STAGE_LABELS))
    uploaded_files = st.file_uploader(label="Upload raw data and U-Sleep scoring.",
        type=[".npy", ".edf"],
        accept_multiple_files=True,
        label_visibility="visible",
    )
    print(uploaded_files is None)

    if len(uploaded_files) != 2:
        print("test")
        st.warning("Please upload files to continue with rescoring.")
    else:
        # Import uploaded files
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            if filename.endswith(".npy"):
                scoring = np.load(uploaded_file)
                scoring_time_hrs = np.arange(scoring.shape[0]) * 30 / 3600  # Every data point corresponds to a 30 second epoch; convert to hours.
                scoring_duration = scoring_time_hrs[-1]
                st.write(f"Found scoring file labeled {filename} with \
                        {scoring_duration:.0f} hours and {scoring_duration%1*60:.0f} minutes.")
            elif filename.endswith(".edf"):
                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as temp_file:
                    temp_file.write(uploaded_file.read())  # Write the uploaded file content
                    temp_file_path = temp_file.name  # Get the file path

                # Read the EDF file using MNE
                raw_obj = mne.io.read_raw_edf(temp_file_path, preload=True)

                ch_labels = raw_obj.ch_names
                _, time_sec = raw_obj[:]
                st.write(f"Found PSG file with the following signals: {ch_labels} and a duration of \
                        {time_sec[-1]/3600:.0f} hours and {time_sec[-1]/3600%1*60:.0f} minutes.")
        # Process uploaded files
        uncertain_info = analyze_uncertain_periods(scoring, scoring_time_hrs)
        naive_scoring = np.argmax(scoring, axis=1)
        # Look through uncertain periods
        for uncertain_period in uncertain_info["uncertain_periods"]:
            slice_start = int(uncertain_period["start_hour"]*3600)*FS_PSG
            slice_stop = int(slice_start + 30)*FS_PSG
            visible_time_slice = np.s_[slice_start:slice_stop]
            break
        # Plot figures
        fig = draw_figure(data_scoring=naive_scoring, data_biosignals=raw_obj,
                        time_slice=visible_time_slice)
        st.pyplot(fig)


def analyze_uncertain_periods(confidence_data: np.ndarray, time_hrs: np.ndarray) -> Dict:
    """Analyze periods of low confidence."""
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
        'max_possible_conf': max_possible_conf
    }

def draw_figure(data_scoring, data_biosignals, time_slice, period_scoring=30):
    '''
    INPUTS
    period_scoring (int) - scoring sampling period in seconds.
    '''
    # Create the figure and GridSpec layout
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 5])  # 1 unit for top, 5 for bottom

    # Create the top and bottom axes
    ax_top = fig.add_subplot(gs[0])
    ax_bottom = fig.add_subplot(gs[1])

    # Top plot. Algorithm scoring output.
    time_scoring = np.arange(len(data_scoring))*period_scoring/3600
    ax_top.plot(time_scoring, data_scoring, linewidth=0.5, color="grey")
    ax_top.set_yticks(ticks=range(5), labels=SLEEP_STAGE_LABELS)
    ax_top.set_ylim(4.25,-0.25)
    ax_top.set_xlabel("Time (hrs)")
    ax_top.set_ylabel("Sleep stages")

    # Bottom plot. Raw data display.
    signals, time = data_biosignals[:]
    ch_labels = data_biosignals.ch_names
    for idx, signal in enumerate(signals[:,time_slice]):
        # autoscaling 
        if SELECT_AUTOSCALING == "MINMAX":
            c_minmax = signal.max() - signal.min()
            signal /= c_minmax
        elif SELECT_AUTOSCALING == "RMS":
            c_rms = np.sqrt(np.mean(signal**2))
            signal /= c_rms
        ax_bottom.plot(time[time_slice], signal + idx, linewidth=0.75)  # Blue diagonal line
    ax_bottom.set_yticks(range(signals.shape[0]), ch_labels);

    # Adjust layout
    plt.tight_layout()
    
    return fig

if __name__ == "__main__":
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
            