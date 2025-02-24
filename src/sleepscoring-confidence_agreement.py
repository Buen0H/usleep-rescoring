# Part 1: Imports, Constants, and Core Functions
import os
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from scipy import ndimage
import mne
import csv
import json

# Constants
STAGE_LABELS = ["Wake", "N1", "N2", "N3", "REM"]
STAGE_MAPPING = {
    'Wake': 0,
    'N1': -2,
    'N2': -3,
    'N3': -4,
    'REM': -1
}

class SleepDataLoader:
    def __init__(self):
        """Initialize the data loader with the project directories."""
        self.manual_scoring_dir = "/Volumes/project/3013101.01/preprocessed/manual_sleepscoring/scoring_sarah"
        self.confidence_dir = "/Volumes/project/3013101.01/preprocessed/manual_sleepscoring/confidence_outputs"
        self.edf_dir = "/Volumes/project/3013101.01/preprocessed/manual_sleepscoring/scoring_tania"
        
        # Verify directories exist
        self._verify_directories()
    
    def _verify_directories(self):
        """Verify all required directories exist."""
        for dir_path in [self.manual_scoring_dir, self.confidence_dir, self.edf_dir]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    def find_matching_files(self, folder_name: str) -> Dict[str, str]:
        """Find matching files for a participant folder."""
        paths = {
            'manual_scoring': None,
            'confidence': None,
            'edf': None
        }
        
        folder_parts = folder_name.split(' ')[0]
        participant_match = re.match(r"(P\d+)(_ses\d+)?", folder_parts)
        if not participant_match:
            print(f"Could not parse participant info from folder: {folder_name}")
            return paths
            
        participant_id = participant_match.group(1)
        session = participant_match.group(2) or ""
        
        # Find scoring file
        scoring_file = os.path.join(self.manual_scoring_dir, folder_name, "Sleep profile.txt")
        if os.path.exists(scoring_file):
            paths['manual_scoring'] = scoring_file
        
        # Find confidence file
        conf_filename = f"{participant_id}{session}_sleepscoring_manual_confidence.npy"
        conf_path = os.path.join(self.confidence_dir, conf_filename)
        if os.path.exists(conf_path):
            paths['confidence'] = conf_path
        else:
            print(f"Confidence file not found: {conf_filename}")
            
        # Find EDF file
        edf_filename = f"{participant_id}{session}_sleepscoring_manual.edf"
        edf_path = os.path.join(self.edf_dir, edf_filename)
        if os.path.exists(edf_path):
            paths['edf'] = edf_path
        
        return paths

    def load_manual_scoring(self, scoring_file: str) -> List[int]:
        """Load manual sleep scoring data, skipping first epoch."""
        try:
            with open(scoring_file, "r") as f:
                sleep_stages = []
                for line in f:
                    if ";" in line:
                        stage = line.strip().split(";")[1].strip()
                        sleep_stages.append(STAGE_MAPPING.get(stage, -99))
                return sleep_stages[1:] if len(sleep_stages) > 1 else []
        except Exception as e:
            print(f"Error reading manual scoring file {scoring_file}: {e}")
            return []
    
    def load_confidence_data(self, confidence_file: str) -> np.ndarray:
        """Load confidence scores from numpy file."""
        try:
            return np.load(confidence_file)
        except Exception as e:
            print(f"Error loading confidence file {confidence_file}: {e}")
            return np.array([])

def get_awakening_times(edf_file: str) -> List[float]:
    """Extract awakening times from EDF annotations without loading the full EDF.
    Includes detailed debugging information about annotations.
    """
    try:
        print(f"\n{'='*50}")
        print(f"Analyzing annotations for file: {os.path.basename(edf_file)}")
        print(f"{'='*50}")
        
        # Try different encodings with detailed error reporting
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        annotations = None
        successful_encoding = None
        
        for encoding in encodings:
            try:
                annotations = mne.read_annotations(edf_file, encoding=encoding)
                successful_encoding = encoding
                print(f"\nSuccessfully read annotations with {encoding} encoding")
                break
            except UnicodeDecodeError as e:
                print(f"\nFailed with {encoding} encoding:")
                print(f"Error: {str(e)}")
                continue
        
        if annotations is None:
            print("\nFailed to read annotations with any encoding")
            return []
        
        print("\nAnnotation Analysis:")
        print("-" * 30)
        
        # Analyze all annotations
        print("\nAll annotations found:")
        print(f"Total number of annotations: {len(annotations)}")
        print("\nAnnotation details:")
        print(f"{'Description':<40} | {'Onset (s)':<12} | {'Onset (h)':<12} | {'Duration':<10}")
        print("-" * 80)
        
        for i, annot in enumerate(annotations):
            try:
                desc = str(annot['description'])
                onset_sec = annot['onset']
                onset_hour = onset_sec / 3600
                duration = annot['duration']
                
                print(f"{desc:<40} | {onset_sec:<12.2f} | {onset_hour:<12.2f} | {duration:<10.2f}")
                
            except Exception as e:
                print(f"Error processing annotation {i}: {str(e)}")
                print(f"Raw annotation data: {annot}")
        
        # Look for timing-related annotations
        print("\nSearching for timing markers:")
        time_markers = {
            'lights_off': None,
            'lights_on': None,
            'recording_start': None,
            'first_epoch': None,
            'last_epoch': None
        }
        
        for annot in annotations:
            try:
                desc = annot['description'].lower()
                if 'lights_off' in desc:
                    time_markers['lights_off'] = annot['onset']
                elif 'lights_on' in desc:
                    time_markers['lights_on'] = annot['onset']
                elif 'recording' in desc and 'start' in desc:
                    time_markers['recording_start'] = annot['onset']
                # Add any other relevant markers you find in the data
            except Exception as e:
                continue
        
        print("\nTiming markers found:")
        for marker, time in time_markers.items():
            if time is not None:
                print(f"{marker}: {time/3600:.2f} hours from file start")
        
        # Find awakening events
        awakening_pattern = re.compile(r'a_(NREM|REM)_\d+_s')
        awakening_times = []
        
        print("\nAwakening events found:")
        for annot in annotations:
            try:
                if awakening_pattern.match(annot['description']):
                    onset_sec = annot['onset']
                    # Store raw time for now (no adjustment)
                    awakening_times.append(onset_sec / 3600)
                    print(f"Found: {annot['description']}")
                    print(f"Time from file start: {onset_sec/3600:.2f} hours")
                    if time_markers['lights_off'] is not None:
                        rel_time = (onset_sec - time_markers['lights_off']) / 3600
                        print(f"Time from lights off: {rel_time:.2f} hours")
                    print("-" * 20)
            except Exception as e:
                print(f"Error processing potential awakening: {str(e)}")
                continue
        
        print("\nSummary:")
        print(f"Found {len(awakening_times)} awakening events")
        if time_markers['lights_off'] is not None:
            print("Times will be adjusted relative to lights off trigger")
            awakening_times = [(t * 3600 - time_markers['lights_off']) / 3600 for t in awakening_times]
        else:
            print("No lights off trigger found - times will be relative to file start")
            
        print(f"{'='*50}\n")
        
        return sorted(awakening_times)
        
    except Exception as e:
        print(f"\nError in main annotation processing:")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return []# Part 1: Imports, Constants, and Core Functions

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
                'end_hour': time_hrs[region_indices[-1]],
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

def plot_confusion_matrix(conf_matrix: np.ndarray, output_path: str, title: str = "Confusion Matrix"):
    """Plot confusion matrix using seaborn."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=STAGE_LABELS,
        yticklabels=STAGE_LABELS
    )
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_hypnogram_comparison(manual_scores: List[int], confidence_data: np.ndarray, 
                            output_path: str, participant_id: str, session: str,
                            awakening_times: List[float] = None) -> Dict:
    """Create a three-panel plot showing hypnogram comparison and return uncertainty stats."""
    # Convert manual scores to 0-4 range for plotting (reversed order: Wake, REM, N1, N2, N3)
    stage_conversion_reverse = {0: 4, -1: 3, -2: 2, -3: 1, -4: 0}  # Reversed order
    manual_plot = np.array([stage_conversion_reverse.get(score, -1) for score in manual_scores])
    
    # Get automatic scoring and calculate confidence measures
    auto_scores_orig = np.argmax(confidence_data, axis=1)
    auto_conversion = {0: 4, 4: 3, 1: 2, 2: 1, 3: 0}
    auto_scores = np.array([auto_conversion[score] for score in auto_scores_orig])
    
    # Create time array
    epoch_length = 30
    duration_hours = len(manual_scores) * epoch_length / 3600
    time_hrs = np.linspace(0, duration_hours, len(manual_scores))
    
    # Analyze uncertain periods
    uncertainty_stats = analyze_uncertain_periods(confidence_data, time_hrs)
    
    # Create the figure
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.5])
    
    # Plot 1: Hypnogram Comparison
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time_hrs, manual_plot, 'r-', label='Manual', linewidth=1)
    ax1.plot(time_hrs, auto_scores, 'k-', label='Automatic', linewidth=1)
    
    # Add awakening markers if provided
    if awakening_times:
        for t in awakening_times:
            ax1.axvline(x=t, color='blue', linestyle='--', alpha=0.5)
    
    ax1.set_yticks(range(5))
    ax1.set_yticklabels(['N3', 'N2', 'N1', 'REM', 'Wake'])
    ax1.set_ylabel('Sleep Stage')
    ax1.legend()
    ax1.set_title(f'Hypnogram Comparison - {participant_id} Session {session}')
    ax1.grid(True, which='major', axis='y', linestyle='--', alpha=0.3)
    
    # Plot 2: Hypnodensity with filled areas
    ax2 = fig.add_subplot(gs[1])
    colors = ['skyblue', 'orange', 'green', 'red', 'purple']
    labels = ['Wake', 'N1', 'N2', 'N3', 'REM']
    
    y_stack = np.zeros(len(time_hrs))
    for i, (color, label) in enumerate(zip(colors, labels)):
        values = confidence_data[:, i]
        ax2.fill_between(time_hrs, y_stack, y_stack + values, 
                        color=color, alpha=0.6, label=label)
        y_stack += values
    
    # Add awakening markers to hypnodensity plot
    if awakening_times:
        for t in awakening_times:
            ax2.axvline(x=t, color='blue', linestyle='--', alpha=0.5)
    
    ax2.set_ylabel('Stage Probability')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 3: Confidence with highlighting
    ax3 = fig.add_subplot(gs[2])
    max_confidences = np.max(confidence_data, axis=1)
    ax3.plot(time_hrs, max_confidences, 'k-', label='Maximum Confidence')
    
    # Highlight uncertain periods
    for period in uncertainty_stats['uncertain_periods']:
        ax3.axvspan(period['start_hour'], period['end_hour'],
                    color='red', alpha=0.2)
    
    # Add awakening markers to confidence plot
    if awakening_times:
        for t in awakening_times:
            ax3.axvline(x=t, color='blue', linestyle='--', alpha=0.5)
    
    ax3.set_ylabel('Confidence')
    ax3.set_xlabel('Time (hours)')
    ax3.axhline(y=uncertainty_stats['confidence_threshold'], color='r', 
                linestyle='--', alpha=0.5,
                label=f'Threshold ({uncertainty_stats["confidence_threshold"]:.2f} of {uncertainty_stats["max_possible_conf"]:.1f})')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return uncertainty_stats

def analyze_confidence(confidence_data: np.ndarray) -> Dict:
    """Analyze confidence scores from the automatic scoring."""
    # Get maximum confidence per epoch
    max_per_epoch = np.max(confidence_data, axis=1)
    
    # Count epochs with multiple significant stages (confidence > 1)
    significant_counts = np.sum(confidence_data > 1, axis=1)
    unsure_epochs = significant_counts > 1
    
    # Calculate stage contributions for unsure epochs
    stage_contributions = np.sum((confidence_data > 1)[unsure_epochs], axis=0)
    total_unsure_epochs = np.sum(unsure_epochs)
    
    # Calculate stage percentages
    stage_percentages = (stage_contributions / total_unsure_epochs * 100 
                        if total_unsure_epochs > 0 else np.zeros(confidence_data.shape[1]))
    
    return {
        "mean_confidence": float(np.mean(max_per_epoch)),
        "std_dev_confidence": float(np.std(max_per_epoch)),
        "max_confidence": float(np.max(max_per_epoch)),
        "unsure_epochs": int(total_unsure_epochs),
        "total_epochs": int(len(max_per_epoch)),
        "unsure_percentage": float((total_unsure_epochs / len(max_per_epoch)) * 100),
        "stage_contributions": stage_contributions.tolist(),
        "stage_percentages": stage_percentages.tolist()
    }

def calculate_agreement_metrics(manual_scores: List[int], confidence_data: np.ndarray) -> Dict:
    """Calculate agreement metrics between manual scoring and automatic scoring."""
    # Get predicted stages from confidence data
    predicted_stages = np.argmax(confidence_data, axis=1)
    
    # Convert predicted stages to same mapping as manual scores
    stage_conversion = {
        0: 0,    # Wake -> Wake
        1: -2,   # N1 -> -2
        2: -3,   # N2 -> -3
        3: -4,   # N3 -> -4
        4: -1    # REM -> -1
    }
    predicted_stages = np.array([stage_conversion[stage] for stage in predicted_stages])
    
    # Calculate basic agreement
    agreement_rate = np.mean(predicted_stages == manual_scores) * 100
    
    # Calculate Cohen's Kappa
    kappa = cohen_kappa_score(manual_scores, predicted_stages)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(
        manual_scores, 
        predicted_stages,
        labels=[0, -2, -3, -4, -1]  # Order: Wake, N1, N2, N3, REM
    )
    
    # Calculate per-stage agreement
    stage_agreements = {}
    for stage_name, stage_value in [("Wake", 0), ("N1", -2), ("N2", -3), ("N3", -4), ("REM", -1)]:
        stage_indices = np.where(np.array(manual_scores) == stage_value)[0]
        if len(stage_indices) > 0:
            stage_agreement = np.mean(predicted_stages[stage_indices] == stage_value) * 100
            stage_agreements[f"{stage_name}_agreement"] = stage_agreement
        else:
            stage_agreements[f"{stage_name}_agreement"] = 0.0
    
    return {
        "overall_agreement": agreement_rate,
        "kappa": kappa,
        "confusion_matrix": conf_matrix,
        **stage_agreements
    }

def main():
    # Initialize the data loader
    loader = SleepDataLoader()
    
    # Set up output files
    output_dir = "/Volumes/project/3013101.01/preprocessed/manual_sleepscoring"
    agreement_file = os.path.join(output_dir, "agreement_results.csv")
    uncertainty_file = os.path.join(output_dir, "uncertainty_results.csv")
    
    # Define CSV headers
    agreement_fields = [
        "participant_id", "session", "overall_agreement", "kappa",
        "Wake_agreement", "N1_agreement", "N2_agreement", "N3_agreement", "REM_agreement",
        "total_epochs", "mean_confidence", "std_dev_confidence", "max_confidence",
        "unsure_epochs", "unsure_percentage"
    ] + [f"{stage}_contribution" for stage in STAGE_LABELS]
    
    uncertainty_fields = [
        "participant_id", "session", "n_uncertain_periods", "total_uncertain_mins",
        "confidence_threshold", "max_possible_conf", "n_awakenings", "awakening_times",
        "uncertain_periods"
    ]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize overall confusion matrix
    overall_conf_matrix = None
    
    # Get list of participant folders
    participant_folders = [
        f for f in os.listdir(loader.manual_scoring_dir)
        if os.path.isdir(os.path.join(loader.manual_scoring_dir, f))
    ]
    
    print(f"Found {len(participant_folders)} participant folders")
    
    # Open CSV files for writing
    with open(agreement_file, 'w', newline='') as agr_file, \
         open(uncertainty_file, 'w', newline='') as unc_file:
        
        agr_writer = csv.DictWriter(agr_file, fieldnames=agreement_fields)
        unc_writer = csv.DictWriter(unc_file, fieldnames=uncertainty_fields)
        
        agr_writer.writeheader()
        unc_writer.writeheader()
        
        # Process each participant
        for folder in participant_folders:
            print(f"\nProcessing folder: {folder}")
            
            # Get matching files
            files = loader.find_matching_files(folder)
            
            # Check if required files exist
            if not files['manual_scoring'] or not files['confidence']:
                print(f"Missing required files for {folder}, skipping")
                continue
            
            try:
                # Extract participant info from folder name
                folder_parts = folder.split(' ')[0]
                participant_match = re.match(r"(P\d+)(_ses\d+)?", folder_parts)
                participant_id = participant_match.group(1)
                session = participant_match.group(2)[4:] if participant_match.group(2) else "1"
                
                # Load data
                manual_scores = loader.load_manual_scoring(files['manual_scoring'])
                confidence_data = loader.load_confidence_data(files['confidence'])
                
                # Get awakening times if EDF file exists
                awakening_times = []
                if files['edf']:
                    print("Reading awakening times from EDF...")
                    awakening_times = get_awakening_times(files['edf'])
                    print(f"Found {len(awakening_times)} awakenings")
                
                if len(manual_scores) == 0 or confidence_data.size == 0:
                    print(f"Empty data found for {folder}, skipping")
                    continue
                
                # Verify data alignment
                min_length = min(len(manual_scores), len(confidence_data))
                manual_scores = manual_scores[:min_length]
                confidence_data = confidence_data[:min_length]
                
                # Calculate metrics
                metrics = calculate_agreement_metrics(manual_scores, confidence_data)
                confidence_metrics = analyze_confidence(confidence_data)
                
                # Update overall confusion matrix
                if overall_conf_matrix is None:
                    overall_conf_matrix = metrics["confusion_matrix"]
                else:
                    overall_conf_matrix += metrics["confusion_matrix"]
                
                # Plot confusion matrix and hypnogram
                conf_plot_path = os.path.join(loader.manual_scoring_dir, folder, 
                                            f"{participant_id}_ses{session}_confusion.png")
                hypno_plot_path = os.path.join(loader.manual_scoring_dir, folder, 
                                             f"{participant_id}_ses{session}_hypnogram.png")
                
                plot_confusion_matrix(
                    metrics["confusion_matrix"],
                    conf_plot_path,
                    f"Confusion Matrix - {participant_id} Session {session}"
                )
                
                # Create hypnogram and get uncertainty stats
                uncertainty_stats = plot_hypnogram_comparison(
                    manual_scores,
                    confidence_data,
                    hypno_plot_path,
                    participant_id,
                    session,
                    awakening_times  # Pass awakening times to plotting function
                )
                
                # Prepare agreement row
                agreement_row = {
                    "participant_id": participant_id,
                    "session": session,
                    "overall_agreement": round(metrics["overall_agreement"], 2),
                    "kappa": round(metrics["kappa"], 3),
                    "Wake_agreement": round(metrics["Wake_agreement"], 2),
                    "N1_agreement": round(metrics["N1_agreement"], 2),
                    "N2_agreement": round(metrics["N2_agreement"], 2),
                    "N3_agreement": round(metrics["N3_agreement"], 2),
                    "REM_agreement": round(metrics["REM_agreement"], 2),
                    "total_epochs": min_length,
                    "mean_confidence": round(confidence_metrics["mean_confidence"], 3),
                    "std_dev_confidence": round(confidence_metrics["std_dev_confidence"], 3),
                    "max_confidence": round(confidence_metrics["max_confidence"], 3),
                    "unsure_epochs": confidence_metrics["unsure_epochs"],
                    "unsure_percentage": round(confidence_metrics["unsure_percentage"], 2)
                }
                
                # Add stage contributions
                for stage, contrib in zip(STAGE_LABELS, confidence_metrics["stage_contributions"]):
                    agreement_row[f"{stage}_contribution"] = round(contrib, 2)
                
                # Prepare uncertainty row with awakening information
                uncertainty_row = {
                    "participant_id": participant_id,
                    "session": session,
                    "n_uncertain_periods": uncertainty_stats["n_uncertain_periods"],
                    "total_uncertain_mins": round(uncertainty_stats["total_uncertain_mins"], 2),
                    "confidence_threshold": round(uncertainty_stats["confidence_threshold"], 3),
                    "max_possible_conf": uncertainty_stats["max_possible_conf"],
                    "n_awakenings": len(awakening_times),
                    "awakening_times": json.dumps([round(t, 3) for t in awakening_times]),
                    "uncertain_periods": json.dumps(uncertainty_stats["uncertain_periods"])
                }
                
                # Write to CSV files
                agr_writer.writerow(agreement_row)
                unc_writer.writerow(uncertainty_row)
                
                print(f"Processed {participant_id} session {session}: "
                      f"Agreement = {agreement_row['overall_agreement']}%, "
                      f"Kappa = {agreement_row['kappa']}, "
                      f"Uncertain periods = {uncertainty_row['n_uncertain_periods']}, "
                      f"Awakenings = {uncertainty_row['n_awakenings']}")
                
            except Exception as e:
                print(f"Error processing folder {folder}: {e}")
                continue
    
    # Plot overall confusion matrix
    if overall_conf_matrix is not None:
        overall_plot_path = os.path.join(output_dir, "overall_confusion_matrix.png")
        plot_confusion_matrix(
            overall_conf_matrix,
            overall_plot_path,
            "Overall Confusion Matrix"
        )

if __name__ == "__main__":
    main()