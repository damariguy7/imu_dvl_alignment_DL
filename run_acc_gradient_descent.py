def run_acc_gradient_descent(v_imu, v_dvl, omega_skew_imu, a_imu, max_iterations=1000, learning_rate=0.0001,
                             tolerance=1e-6):
    """
    Implement acceleration-based gradient descent optimization to find optimal rotation matrix.

    Args:
        v_imu: IMU velocity data (3xN array)
        v_dvl: DVL velocity data (3xN array)
        omega_skew_imu: Skew-symmetric matrices of angular velocities (3x3xN array)
        a_imu: IMU acceleration data (3xN array)
        max_iterations: Maximum number of iterations for gradient descent
        learning_rate: Learning rate for gradient descent
        tolerance: Convergence tolerance

    Returns:
        R: Optimal rotation matrix
        euler_angles_deg: Euler angles in degrees [roll, pitch, yaw]
    """

    def skew_to_vec(S):
        """Convert 3x3 skew symmetric matrix to 3x1 vector"""
        return np.array([S[2, 1], S[0, 2], S[1, 0]])

    def rotation_matrix_to_euler_xyz(R):
        """
        Convert rotation matrix to XYZ (roll, pitch, yaw) Euler angles

        Args:
            R: 3x3 rotation matrix

        Returns:
            np.array: [roll, pitch, yaw] in radians
        """
        pitch = np.arcsin(-R[0, 2])

        if np.cos(pitch) > 1e-6:
            roll = np.arctan2(R[1, 2], R[2, 2])
            yaw = np.arctan2(R[0, 1], R[0, 0])
        else:
            # Gimbal lock case
            roll = 0
            yaw = np.arctan2(-R[1, 0], R[1, 1])

        return np.array([roll, pitch, yaw])

    def euler_xyz_to_rotation_matrix(angles):
        """
        Convert XYZ (roll, pitch, yaw) Euler angles to rotation matrix

        Args:
            angles: [roll, pitch, yaw] in radians

        Returns:
            R: 3x3 rotation matrix
        """
        roll, pitch, yaw = angles

        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])

        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])

        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])

        return Rx @ Ry @ Rz

    def compute_error(R):
        """Compute sum of squared errors"""
        total_error = 0
        for i in range(v_imu.shape[1]):
            # Predicted acceleration using current rotation estimate
            a_pred = omega_skew_imu[:, :, i] @ R @ v_dvl[:, i] + R @ np.gradient(v_dvl[:, i])

            # Error between predicted and measured acceleration
            error = a_imu[:, i] - a_pred
            total_error += np.sum(error ** 2)

        return total_error

    def compute_gradient(R):
        """Compute gradient of error with respect to rotation matrix"""
        gradient = np.zeros((3, 3))

        for i in range(v_imu.shape[1]):
            omega_skew = omega_skew_imu[:, :, i]
            v_dvl_i = v_dvl[:, i]
            v_dvl_dot = np.gradient(v_dvl_i)

            # Predicted acceleration
            a_pred = omega_skew @ R @ v_dvl_i + R @ v_dvl_dot

            # Error
            error = a_imu[:, i] - a_pred

            # Gradient contribution from this sample
            gradient -= 2 * (np.outer(error, v_dvl_i) @ omega_skew.T +
                             np.outer(error, v_dvl_dot))

        return gradient

    # Initialize with identity rotation
    euler_angles_rads = np.zeros(3)
    R = euler_xyz_to_rotation_matrix(euler_angles_rads)

    prev_error = float('inf')

    for iteration in range(max_iterations):
        # Compute current error
        current_error = compute_error(R)

        # Check convergence
        if abs(current_error - prev_error) < tolerance:
            break

        # Compute gradient
        gradient = compute_gradient(R)

        # Update rotation matrix using gradient descent on SO(3)
        # Project gradient onto tangent space of SO(3)
        A = gradient @ R.T - R @ gradient.T
        omega = skew_to_vec(A)

        # Update Euler angles
        euler_angles_rads -= learning_rate * omega

        # Convert back to rotation matrix ensuring SO(3) constraint
        R = euler_xyz_to_rotation_matrix(euler_angles_rads)

        prev_error = current_error

    # Convert final result to degrees
    euler_angles_deg = np.degrees(euler_angles_rads)

    return euler_angles_deg