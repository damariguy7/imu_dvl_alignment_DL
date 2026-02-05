import numpy as np


def calc_squared_err_angles(a, b):
    squared_diff = np.zeros(3)

    for i in range(3):
        # Normalize both angles to [-180, 180]
        a_norm = ((a[i] + 180) % 360) - 180
        b_norm = ((b[i] + 180) % 360) - 180

        # Calculate the absolute difference
        diff = abs(a_norm - b_norm)

        # Handle wrap-around case
        if diff > 180:
            diff = 360 - diff

        squared_diff[i] = diff ** 2

    # mean_squared_err = np.mean(squared_diff)
    squared_err = squared_diff

    # rmse = np.sqrt(mean_squared_err)

    return squared_err

def calc_err_angles(a, b):
    angle_diff = np.zeros(3)

    for i in range(3):
        # Normalize both angles to [-180, 180]
        a_norm = ((a[i] + 180) % 360) - 180
        b_norm = ((b[i] + 180) % 360) - 180

        # Calculate the absolute difference
        diff = abs(a_norm - b_norm)

        # Handle wrap-around case
        if diff > 180:
            diff = 360 - diff

        angle_diff[i] = diff

    return angle_diff


def squared_angle_difference(a, b):
    return (min((a - b) % 360, (b - a) % 360)) ** 2
