import numpy as np
import pandas as pd


def movmean(data, window_size):
    """
    Apply moving average filter to data (similar to MATLAB's movmean)

    Args:
        data: Input data array/list
        window_size: Size of the moving window

    Returns:
        Smoothed data array
    """
    if len(data) < window_size:
        # If data is shorter than window, return original data
        return np.array(data)

    # Convert to numpy array if not already
    data = np.array(data)

    # Use pandas rolling mean for efficient computation
    df = pd.DataFrame(data)
    smoothed = df.rolling(window=window_size, center=True, min_periods=1).mean().values.flatten()

    return smoothed


def movmean_numpy(data, window_size):
    """
    Alternative implementation using numpy only
    """
    data = np.array(data)
    if len(data) < window_size:
        return data

    # Pad the data to handle edges
    pad_width = window_size // 2
    padded_data = np.pad(data, pad_width, mode='edge')

    # Apply moving average
    smoothed = np.convolve(padded_data, np.ones(window_size) / window_size, mode='same')

    # Remove padding
    return smoothed[pad_width:pad_width + len(data)]