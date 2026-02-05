import numpy as np


def rotation_matrix_log_so3(R):
    """
    Compute the SO(3) logarithm map of a rotation matrix.

    This converts a rotation matrix to its axis-angle representation in so(3).
    The result is a 3D vector where the magnitude is the rotation angle (in radians)
    and the direction is the rotation axis.

    Parameters:
        R: 3x3 rotation matrix

    Returns:
        omega: 3D vector in so(3) (axis-angle representation)
    """
    # Compute the trace to find the rotation angle
    trace_R = np.trace(R)

    # Handle the identity case (no rotation)
    if np.abs(trace_R - 3.0) < 1e-6:
        return np.zeros(3)

    # Compute rotation angle
    # trace(R) = 1 + 2*cos(theta)
    theta = np.arccos(np.clip((trace_R - 1.0) / 2.0, -1.0, 1.0))

    # Handle small angle case (near identity)
    if theta < 1e-6:
        return np.zeros(3)

    # Handle the pi rotation case (trace = -1)
    if np.abs(theta - np.pi) < 1e-6:
        # Find the axis by looking at the diagonal elements
        # The axis is along the column with the largest diagonal element
        diag_elements = np.diag(R)
        k = np.argmax(diag_elements)

        # Extract the axis
        axis = np.zeros(3)
        axis[k] = np.sqrt((R[k, k] + 1.0) / 2.0)

        for i in range(3):
            if i != k:
                axis[i] = R[i, k] / (2.0 * axis[k])

        omega = theta * axis
        return omega

    # General case: compute the skew-symmetric part
    # log(R) = (theta / (2*sin(theta))) * (R - R^T)
    skew_symmetric = (R - R.T) * theta / (2.0 * np.sin(theta))

    # Extract the vector from skew-symmetric matrix
    # [0  -z   y]
    # [z   0  -x]
    # [-y  x   0]
    omega = np.array([
        skew_symmetric[2, 1],
        skew_symmetric[0, 2],
        skew_symmetric[1, 0]
    ])

    return omega
