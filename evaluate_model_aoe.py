import numpy as np
import torch
from euler_angles_to_rotation_matrix import euler_angles_to_rotation_matrix
from rotation_matrix_log_so3 import rotation_matrix_log_so3


def evaluate_model_aoe(model, test_loader, device):
    """
    Evaluate model using Absolute Orientation Error (AOE) criterion.

    AOE = sqrt(mean(||log(R_gt^T @ R_est)||^2))

    This properly accounts for the manifold structure of SO(3).

    Parameters:
        model: Trained model
        test_loader: DataLoader with test data
        device: Computing device (CPU/GPU)

    Returns:
        aoe: Absolute Orientation Error in radians
        aoe_degrees: AOE converted to degrees for interpretability
    """
    model.eval()
    squared_error_list = []



    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Move to CPU immediately and detach
            outputs_cpu = outputs.detach().cpu().numpy()
            targets_cpu = targets.detach().cpu().numpy()

            # Calculate AOE for each sample in the batch
            for ii in range(len(targets_cpu)):
                squared_error_rads = calc_aoe_single_sample(targets_cpu[ii], outputs_cpu[ii])
                squared_error_degrees = np.degrees(squared_error_rads)
                squared_error_list.append(squared_error_degrees)

            # Clear GPU cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

    # Compute AOE: sqrt(mean(squared_errors))
    aoe = np.sqrt(np.mean(squared_error_list))
    aoe_degrees = aoe

    print(f'AOE: {aoe:.6f} radians ({aoe_degrees:.4f} degrees)')

    return aoe, aoe_degrees




def calc_aoe_single_sample(euler_gt, euler_est):
    """
    Calculate the Absolute Orientation Error (AOE) for a single sample.

    AOE measures the geodesic distance on SO(3) between two orientations.

    Parameters:
        euler_gt: Ground truth Euler angles [roll, pitch, yaw] in degrees
        euler_est: Estimated Euler angles [roll, pitch, yaw] in degrees

    Returns:
        squared_error: Squared rotation error (angle^2 in radians^2)
    """
    # Convert Euler angles from degrees to radians
    roll_gt, pitch_gt, yaw_gt = np.radians(euler_gt)
    roll_est, pitch_est, yaw_est = np.radians(euler_est)

    # Convert to rotation matrices
    R_gt = euler_angles_to_rotation_matrix(roll_gt, pitch_gt, yaw_gt)
    R_est = euler_angles_to_rotation_matrix(roll_est, pitch_est, yaw_est)

    # Compute rotation error matrix: R_error = R_gt^T @ R_est
    R_error = R_gt.T @ R_est

    # Use SO(3) logarithm to get axis-angle representation
    omega = rotation_matrix_log_so3(R_error)

    # The AOE for this sample is the squared norm of omega
    # ||omega||^2 = theta^2 (where theta is the rotation angle)
    squared_error = np.sum(omega ** 2)

    return squared_error