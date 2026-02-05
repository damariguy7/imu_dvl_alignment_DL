import numpy as np


def euler_angles_to_rotation_matrix(roll_rads, pitch_rads, yaw_rads):
    """
    Convert Euler angles (roll, pitch, yaw) to a rotation matrix.
    ZYX convention: Rotate about z-axis, then y-axis, then x-axis.

    Parameters:
        roll_rads: Roll angle in radians
        pitch_rads: Pitch angle in radians
        yaw_rads: Yaw angle in radians

    Returns:
        R: 3x3 rotation matrix
    """
    # Compute sines and cosines
    c_roll = np.cos(roll_rads)
    s_roll = np.sin(roll_rads)
    c_pitch = np.cos(pitch_rads)
    s_pitch = np.sin(pitch_rads)
    c_yaw = np.cos(yaw_rads)
    s_yaw = np.sin(yaw_rads)

    # Compute rotation matrix
    Rz = np.array([[c_yaw, s_yaw, 0],
                   [-s_yaw, c_yaw, 0],
                   [0, 0, 1]])

    Ry = np.array([[c_pitch, 0, -s_pitch],
                   [0, 1, 0],
                   [s_pitch, 0, c_pitch]])

    Rx = np.array([[1, 0, 0],
                   [0, c_roll, s_roll],
                   [0, -s_roll, c_roll]])

    # Combine rotation matrices
    R = np.dot(Rx, np.dot(Ry, Rz))

    return R
