import numpy as np
from scipy import ndimage
from typing import Dict

def analyze_uncertain_periods(confidence_data: np.ndarray) -> Dict:
    """Analyze periods of low confidence.
    Given a 2D numpy array of shape (n_epochs, n_classes) representing the confidence scores
    for each epoch and class, identify periods of low confidence and return uncertain periods.
    """
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
