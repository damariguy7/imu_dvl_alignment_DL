import numpy as np


def run_svd_solution_for_wahba_problem(velocity_ins_sampled, velocity_dvl_sampled):
    """
    Solve the Wahba problem using Singular Value Decomposition (SVD).

    Parameters:
        v_imu_sampled: Sampled INS velocity vector.
        v_dvl_sampled: Sampled DVL velocity vector.

    Returns:
        rotation_matrix_wahba: Optimal rotation matrix.
    """

    # Compute the cross-correlation matrix
    w = np.dot(velocity_ins_sampled, velocity_dvl_sampled.T)

    # Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(w)

    # Compute the optimal rotation matrix
    rotation_matrix_wahba = np.dot(Vt.T, U.T)

    curr_euler_angles_rads = rotation_matrix_to_euler_zyx(rotation_matrix_wahba)

    # Convert angles to degrees
    return curr_euler_angles_rads


def rotation_matrix_to_euler_zyx(R):
    """
    Extract Euler angles from a rotation matrix using ZYX convention.

    Args:
        R (numpy.ndarray): 3x3 rotation matrix.

    Returns:
        numpy.ndarray: Euler angles [yaw, pitch, roll] in radians.
    """

    # R = R.T

    # Extracting individual elements from the rotation matrix
    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r23, r33 = R[1, 2], R[2, 2]

    # Computing roll (around X axis)
    roll = np.arctan2(r23, r33)
    roll = np.arctan2(r23, r33)

    # Computing pitch (around Y axis)
    pitch = np.arcsin(-r13)

    # Computing yaw (around Z axis)
    yaw = np.arctan2(r12, r11)

    return np.array([roll, pitch, yaw])