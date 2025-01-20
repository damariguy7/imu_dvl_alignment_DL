import math

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import logm, expm
import torch
from torch import nn

from scripts.models.resnet1d import resnet18_1d

class Resnet1chDnet(nn.Module):
    def __init__(self, in_channels=6, output_features=1):
        self.in_channels = in_channels
        super(Resnet1chDnet, self).__init__()

        self.model = resnet18_1d()

        self.model.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(512, output_features)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        return self.model(x)


# Define the CNN model
class IMUDVLCNN(nn.Module):
    def __init__(self, dropout_rate=0.2):  # Reduced dropout rate
        super(IMUDVLCNN, self).__init__()
        # Increase network capacity slightly
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)

        self.conv1 = nn.Conv1d(6, 128, kernel_size=5, padding=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=1)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=5, padding=1)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)

        # Residual connection for first block
        identity = self.conv1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x + identity  # Residual connection
        x = self.dropout(x)

        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Third block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Projection shortcut if dimensions change
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class SimplerIMUResNet(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(SimplerIMUResNet, self).__init__()

        # Initial convolution
        self.conv1 = nn.Conv1d(6, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet blocks with increasing channels
        self.res1_1 = ResBlock1D(64, 128)
        #self.res1_2 = ResBlock1D(64, 64)

        self.res2_1 = ResBlock1D(128, 256)
        #self.res2_2 = ResBlock1D(128, 128)

        self.res3_1 = ResBlock1D(256, 512)
        #self.res3_2 = ResBlock1D(256, 256)

        self.res4_1 = ResBlock1D(512, 1024)
        #self.res4_2 = ResBlock1D(512, 512)

        # Global pooling and final layers
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(1024, 4)

    def forward(self, x):
        # Input shape: (batch, time, features)
        x = x.permute(0, 2, 1)  # to (batch, features, time)

        # Initial convolution block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet blocks
        # Stage 1
        x = self.res1_1(x)
        #x = self.res1_2(x)

        # Stage 2
        x = self.res2_1(x)
        #x = self.res2_2(x)

        # Stage 3
        x = self.res3_1(x)
        #x = self.res3_2(x)

        # Stage 4
        x = self.res4_1(x)
        #x = self.res4_2(x)

        # Global pooling and prediction
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        #x = self.dropout(x)
        x = self.fc(x)

        return x


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, scheduler=None):  # Removed l2_lambda parameter
    model.to(device)
    best_val_loss = float('inf')
    patience = 1
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            # Removed L2 regularization calculation and addition

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            running_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Training Loss: {avg_train_loss:.4f}, '
              f'Validation Loss: {avg_val_loss:.4f}')

        if scheduler is not None:
            scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch + 1}")
                model.load_state_dict(best_model_state)  # Restore best model
                break
class AngularMSELoss(nn.Module):
    def __init__(self, period=360.0):
        """
        Custom loss function for angular values that handles wrapping.

        Args:
            period (float): The period of the angle in degrees (360 for full circle, 180 for half circle)
        """
        super(AngularMSELoss, self).__init__()
        self.period = period

    def forward(self, pred, target):
        """
        Calculate the MSE loss accounting for angular wrapping.

        Args:
            pred (torch.Tensor): Predicted angles in degrees (batch_size x 3 for roll, pitch, yaw)
            target (torch.Tensor): Target angles in degrees (batch_size x 3 for roll, pitch, yaw)

        Returns:
            torch.Tensor: Mean squared angular difference
        """
        # Calculate the absolute difference
        diff = pred - target

        # Handle wrapping for each angle separately
        wrapped_diff = torch.remainder(diff + self.period / 2, self.period) - self.period / 2

        # Calculate MSE on the wrapped differences
        return torch.mean(wrapped_diff ** 2)



def quaternion_to_euler(q):
    """Convert quaternions to Euler angles (roll, pitch, yaw).

    Args:
        q (torch.Tensor): Quaternions with shape (..., 4) where q = [w, x, y, z]

    Returns:
        torch.Tensor: Euler angles [roll, pitch, yaw] in degrees
    """
    # Extract quaternion components
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * torch.tensor(math.pi / 2, device=q.device),
        torch.asin(sinp)
    )

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    # Convert from radians to degrees
    roll_deg = roll * 180.0 / math.pi
    pitch_deg = pitch * 180.0 / math.pi
    yaw_deg = yaw * 180.0 / math.pi

    return torch.stack([roll_deg, pitch_deg, yaw_deg], dim=-1)


def single_quaternion_to_euler(q):
    """Convert a single quaternion to Euler angles (roll, pitch, yaw).

    Args:
        q (list or array): Single quaternion [w, x, y, z]

    Returns:
        tuple: Euler angles (roll, pitch, yaw) in degrees
    """
    # Extract quaternion components
    w, x, y, z = q[0], q[1], q[2], q[3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    # Check for gimbal lock, pitch = ±90°
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    # Convert from radians to degrees
    roll_deg = roll * 180.0 / math.pi
    pitch_deg = pitch * 180.0 / math.pi
    yaw_deg = yaw * 180.0 / math.pi

    return roll_deg, pitch_deg, yaw_deg

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    rmse_list = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # Convert quaternions to Euler angles
            outputs_euler = quaternion_to_euler(outputs)
            targets_euler = quaternion_to_euler(targets)

            for ii in range(len(inputs)):
                squared_err = squared_angular_difference(targets_euler[ii], outputs_euler[ii])
                rmse = calculate_rmse(squared_err)
                rmse_list.append(rmse)



            # loss = criterion(outputs, targets)
            # total_loss += loss.item()

            all_predictions.append(outputs_euler.cpu().numpy())
            all_targets.append(targets_euler.cpu().numpy())

            # squared_error_svd_baseline = squared_angular_difference(np.array(euler_angles_svd_degrees),
            #                                                         euler_body_dvl_gt)
            # squared_error_svd_baseline_list.append(squared_error_svd_baseline)

    avg_loss = total_loss / len(test_loader)
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    # Calculate RMSE for each Euler angle
    rmse_components = np.sqrt(np.mean((all_predictions - all_targets) ** 2, axis=0))

    # Calculate total RMSE
    total_rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))

    mean_rmse = np.mean(rmse_list)

    return mean_rmse, avg_loss, rmse_components, total_rmse



class IMUDVLWindowedDataset(Dataset):
    def __init__(self, series, window_size):
        self.imu_series = torch.FloatTensor(series[0])
        self.dvl_series = torch.FloatTensor(series[1])
        self.euler_body_dvl_series = torch.FloatTensor(series[2])
        self.window_size = window_size

    def __len__(self):
        return len(self.imu_series) - self.window_size

    def __getitem__(self, idx):
        #window = self.imu_series[idx:idx + self.window_size + 1]
        #return window[:-1], window[-1]

        imu_window = self.imu_series[idx:idx + self.window_size]
        dvl_window = self.dvl_series[idx:idx + self.window_size]
        euler_body_dvl_window = self.euler_body_dvl_series[idx:idx + self.window_size]

        # Combine IMU and DVL data
        input_data = torch.cat((imu_window, dvl_window), dim=1)

        # Return features (IMU and DVL data for the window) and labels (last IMU and DVL data points)
        return input_data, euler_body_dvl_window[0]



def windowed_dataset(series_list, window_size, batch_size, shuffle):
    """Generates dataset windows from multiple time series

    Args:
      series_list (list of arrays of float) - list of time series
      window_size (int) - the number of time steps to include in the feature
      batch_size (int) - the batch size
      shuffle (bool) - whether to shuffle the dataset

    Returns:
      dataloader (torch.utils.data.DataLoader) - DataLoader containing time windows from all series
    """

    datasets = [IMUDVLWindowedDataset(series, window_size) for series in series_list]
    if len(datasets) > 1:
        combined_dataset = ConcatDataset(datasets)
    else:
        combined_dataset = datasets[0]

    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True
    )

    return dataloader



def temporal_split(dataset, train_ratio=0.8):
    split_idx = int(len(dataset) * train_ratio)
    return dataset[:split_idx], dataset[split_idx:]


def save_model(model, window_size, base_path="models"):
    """
    Save the trained model to a file with window size in the filename.

    Args:
        model (torch.nn.Module): The trained model to save
        window_size (int): The window size used for training
        base_path (str): Base directory to save models
    """
    import os
    # # Create models directory if it doesn't exist
    # os.makedirs(base_path, exist_ok=True)


    # Create filename with window size
    filepath = os.path.join(base_path, f'imu_dvl_model_window_{window_size}.pth')

    # Save the model
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")
    return filepath


def vector_to_skew(vector):
    return np.array([
        [0, -vector[2], vector[1]],
        [vector[2], 0, -vector[0]],
        [-vector[1], vector[0], 0]
    ])


def skew_symetric(vectors):
    """
    Convert multiple 3D vectors to skew-symmetric matrices.

    Args:
        vectors: array-like, shape (3, N)

    Returns:
        Array of skew-symmetric matrices, shape (N, 3, 3)
    """
    if vectors.shape[0] != 3:
        raise ValueError("Input array must have shape (3, N)")

    N = vectors.shape[1]
    result = np.zeros((3, 3, N))

    for i in range(N):
        result[:,:,i] = vector_to_skew(vectors[:, i])

    return result


def run_acc_gradient_descent(v_imu, v_dvl, omega_skew_imu, a_imu, max_iterations=1000, learning_rate=0.01,
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
    euler_angles = np.zeros(3)
    R = euler_xyz_to_rotation_matrix(euler_angles)

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
        euler_angles -= learning_rate * omega

        # Convert back to rotation matrix ensuring SO(3) constraint
        R = euler_xyz_to_rotation_matrix(euler_angles)

        prev_error = current_error

    # Convert final result to degrees
    euler_angles_deg = np.degrees(euler_angles)

    return euler_angles_deg


def squared_angular_difference(a, b):

    squared_diff = np.zeros(3)


    for i in range(3):
        # Calculate both possible angular differences
        diff1 = (a[i] - b[i]) % 360
        diff2 = (b[i] - a[i]) % 360

        # Take the minimum of the two differences
        squared_diff[i] = (min(diff1, diff2))**2

    sum_diff = np.mean(squared_diff)
    return sum_diff


def squared_angle_difference(a, b):
    return (min((a - b) % 360, (b - a) % 360))**2


def euler_angles_to_rotation_matrix(roll_rads, pitch_rads, yaw_rads):
    # Convert Euler angles (roll, pitch, yaw) to a rotation matrix.
    # ZYX convention: Rotate about z-axis, then y-axis, then x-axis.

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


def calculate_rmse(squared_error):
    """
    Calculate Root Mean Square Error (RMSE) between ground truth and estimated values.

    Parameters:
        gt: Ground truth values.
        estimated: Estimated values.

    Returns:
        RMSE value.
    """

    # check = (gt - estimated)
    # summ = np.sum(check)

    # return np.sqrt((gt[0] - estimated[0]) ** 2 + (gt[1] - estimated[1]) ** 2 + (gt[2] - estimated[2]) ** 2)
    # result = np.sqrt(np.sum((gt - estimated) ** 2))
    # print(result)
    # return np.sqrt(np.sum((gt - estimated) ** 2))

    checj_it = np.sqrt(np.mean(squared_error))

    cg = np.sqrt(squared_error)

    return np.sqrt(squared_error)




def convert_deg_to_rads(roll_deg, pitch_deg, yaw_deg):
    roll_rads = np.radians(roll_deg)
    pitch_rads = np.radians(pitch_deg)
    yaw_rads = np.radians(yaw_deg)
    return roll_rads, pitch_rads, yaw_rads



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

    # Computing pitch (around Y axis)
    pitch = np.arcsin(-r13)

    # Computing yaw (around Z axis)
    yaw = np.arctan2(r12, r11)

    return np.array([roll, pitch, yaw])


def calculate_min_rmse(rmse_values):
    min_rmse = min(rmse_values)
    min_index = rmse_values.index(min_rmse)
    return min_rmse, min_index

def main(config):
    # Example usage

    # Main variables to run the simulation
    roll_gt_deg = config['roll_gt_deg']
    pitch_gt_deg = config['pitch_gt_deg']
    yaw_gt_deg = config['yaw_gt_deg']
    simulation_freq = 5
    single_dataset_len = 1612
    single_dataset_duration_sec = single_dataset_len/simulation_freq #should be same parameter value like in matlab simulation

    #single_dataset_duration_sec = 230 #should be same parameter value like in matlab simulation
    #single_dataset_len = single_dataset_duration_sec * simulation_freq
    data_path = config['data_path']
    simulated_data_file_name = config['simulated_data_file_name']
    real_data_file_name = config['real_data_file_name']
    trained_model_base_path = config['trained_model_path']
    window_sizes = [8]
    batch_size = 32
    validation_precentage = 20
    num_of_check_baseline_iterations = 1


    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert degrees to radians
    roll_gt_rads, pitch_gt_rads, yaw_gt_rads = convert_deg_to_rads(roll_gt_deg, pitch_gt_deg, yaw_gt_deg)

    # Convert Euler angles to rotation matrix
    rotation_matrix_ins_to_dvl = euler_angles_to_rotation_matrix(roll_gt_rads, pitch_gt_rads, yaw_gt_rads)

    # Read data from the .csv file
    simulated_data_pd = pd.read_csv(os.path.join(data_path, f'{simulated_data_file_name}'), header=None)

    ##prepare real data dataset
    real_data_pd = pd.read_csv(os.path.join(data_path, f'{real_data_file_name}'), header=None)
    v_imu_body_real_data_full = np.array(real_data_pd.iloc[:, 4:7].T)
    # v_imu_body_real_data_full = v_imu_body_real_data_full[:, 150:300]
    v_dvl_real_data_full = np.array(real_data_pd.iloc[:, 13:16].T)
    # v_dvl_real_data_full = v_dvl_real_data_full[:, 150:300]
    euler_body_dvl_real_data_full = np.array(real_data_pd.iloc[:, 16:19].T)
    # euler_body_dvl_real_data_full = euler_body_dvl_real_data_full[:, 150:300]
    real_dataset_len = len(v_imu_body_real_data_full[1])


    ##prepare real data for check
    # real_data_trajectory_index = config['real_data_trajectory_index']
    # real_imu_file_name = config['real_imu_file_name']
    # real_dvl_file_name = config['real_dvl_file_name']
    # real_gt_file_name = config['real_gt_file_name']
    # real_data_imu_pd = pd.read_csv(os.path.join(data_path, f'{real_imu_file_name}',f'{real_data_trajectory_index}'), header=None)
    # real_data_dvl_pd = pd.read_csv(os.path.join(data_path, f'{real_dvl_file_name}',f'{real_data_trajectory_index}'), header=None)
    # real_data_gt_pd = pd.read_csv(os.path.join(data_path, f'{real_gt_file_name}',f'{real_data_trajectory_index}'), header=None)


    ### Prepare the full dataset
    time = np.array(simulated_data_pd.iloc[:, 0].T)
    num_of_simulated_datasets = len(time) // single_dataset_len
    v_imu_body_full = np.array(simulated_data_pd.iloc[:, 1:4].T)
    v_dvl_full = np.array(simulated_data_pd.iloc[:, 4:7].T)
    v_dvl_body_full = np.dot(rotation_matrix_ins_to_dvl.T, v_dvl_full)
    v_gt_body_full = np.array(simulated_data_pd.iloc[:, 7:10].T)
    euler_body_dvl_full = np.array(simulated_data_pd.iloc[:, 16:20].T)
    a_imu_body_full = np.array(simulated_data_pd.iloc[:, 20:23].T)
    omega_imu_body_full = np.array(simulated_data_pd.iloc[:, 23:26].T)
    omega_skew_imu_body_full = skew_symetric(omega_imu_body_full)
    # euler_body_dvl_full = np.array(simulated_data_pd.iloc[:, 16:19].T)
    # a_imu_body_full = np.array(simulated_data_pd.iloc[:, 19:22].T)
    # omega_imu_body_full = np.array(simulated_data_pd.iloc[:, 22:25].T)
    # omega_skew_imu_body_full = skew_symetric(omega_imu_body_full)



    # split data into training and validation sets
    index_of_split_series = int(num_of_simulated_datasets * (validation_precentage / 100))


    current_time_test_list = []
    rmse_test_list = []
    rmse_roll_test_list = []
    rmse_pitch_test_list = []
    rmse_yaw_test_list = []
    v_imu_dvl_train_series_list = []
    v_imu_dvl_valid_series_list = []
    v_imu_dvl_test_series_list = []
    v_imu_dvl_test_real_data_list = []
    model_paths = []

    for i in range(0, index_of_split_series):
        v_imu_dvl_train_series_list.append(
            [v_imu_body_full[:, i * single_dataset_len:i * single_dataset_len + single_dataset_len].T,
             v_dvl_full[:, i * single_dataset_len:i * single_dataset_len + single_dataset_len].T,
             euler_body_dvl_full[:, i * single_dataset_len:i * single_dataset_len + single_dataset_len].T])

    for i in range(index_of_split_series, num_of_simulated_datasets - 1):
        v_imu_dvl_valid_series_list.append(
            [v_imu_body_full[:, i * single_dataset_len:i * single_dataset_len + single_dataset_len].T,
             v_dvl_full[:, i * single_dataset_len:i * single_dataset_len + single_dataset_len].T,
             euler_body_dvl_full[:, i * single_dataset_len:i * single_dataset_len + single_dataset_len].T])

    for i in range(num_of_simulated_datasets - 1, num_of_simulated_datasets):
        v_imu_dvl_test_series_list.append(
            [v_imu_body_full[:, i * single_dataset_len:i * single_dataset_len + single_dataset_len].T,
             v_dvl_full[:, i * single_dataset_len:i * single_dataset_len + single_dataset_len].T,
             euler_body_dvl_full[:, i * single_dataset_len:i * single_dataset_len + single_dataset_len].T])

    v_imu_dvl_test_real_data_list.append([v_imu_body_real_data_full.T,v_dvl_real_data_full.T,euler_body_dvl_real_data_full.T])


################################## train model with simulated data ##################################################
    if(config['train_model']):
        for i in window_sizes:
            print('train with window of %d seconds', i)
            # Parameters
            window_size = i*simulation_freq

            train_loader = windowed_dataset(
                series_list=v_imu_dvl_train_series_list,
                window_size=window_size,
                batch_size=batch_size,
                shuffle=True
            )

            val_loader = windowed_dataset(
                series_list=v_imu_dvl_valid_series_list,
                window_size=window_size,
                batch_size=batch_size,
                shuffle=False
            )


            ## start of CNN ########################################################################
            # Set up the model, loss function, and optimizer
            # Create model with dropout
            model = SimplerIMUResNet(dropout_rate=0.3)

            # Loss function
            criterion = nn.MSELoss()

            # Optimizer with weight decay (L2 regularization)

            optimizer = optim.Adam(model.parameters(), lr=0.0000001, weight_decay=0.01)

#
            # Learning rate scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=4,
                verbose=True
            )
##
            # Train the model with the improved training function
            train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=50,
                device=device,
                scheduler=scheduler
                # l2_lambda=0.01
            )

            # Save the model and store its path
            model_path = save_model(model, i, trained_model_base_path)



    # Test model
    if(config['test_model']):
        for window_size in window_sizes:
            print(f'\nEvaluating model with window size {window_size}')
    
            # Construct the specific model path for this window size
            model_path = os.path.join(trained_model_base_path, f'imu_dvl_model_window_{window_size}.pth')
    
            # Load model
            model = SimplerIMUResNet(dropout_rate=0.3)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
    
    
            if(config['test_type'] == 'simulated_data'):
                # Create test loader for this window size
                test_loader = windowed_dataset(
                    series_list=v_imu_dvl_test_series_list,
                    window_size=window_size,
                    batch_size=batch_size,
                    shuffle=False
                )

            elif(config['test_type'] == 'real_data'):
                # Create test loader for this window size
                test_loader = windowed_dataset(
                    series_list=v_imu_dvl_test_real_data_list,
                    window_size=window_size,
                    batch_size=batch_size,
                    shuffle=False
                )



            #Loss function
            criterion = nn.MSELoss()

            # Evaluate model
            mean_rmse, test_loss, test_rmse_components, test_total_rmse = evaluate_model(
                model, test_loader, criterion, device
            )

            rmse_test_list.append(mean_rmse)
            rmse_roll_test_list.append(test_rmse_components[0])
            rmse_pitch_test_list.append(test_rmse_components[1])
            rmse_yaw_test_list.append(test_rmse_components[2])
            current_time = (window_size / real_dataset_len) * real_dataset_len
            current_time_test_list.append(current_time)

            print(f"Test Loss: {test_loss:.4f}")
            print(
                f"Test RMSE (Roll, Pitch, Yaw): {test_rmse_components[0]:.4f}, {test_rmse_components[1]:.4f}, {test_rmse_components[2]:.4f}")
            print(f"Test Total RMSE: {test_total_rmse:.4f}")



    
        #### Plot test results
        # Create a new figure with 3 subplots (one for each axis)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 15), sharex=True)
        # Set the color to purple for ax1, ax2, and ax3
        color = 'purple'
        # Plot rmse roll angle
        ax1.plot(current_time_test_list, rmse_roll_test_list, color=color)
        ax1.set_ylabel('Roll RMSE Test [deg]')
        ax1.legend()
        ax1.grid(True)
        # Plot rmse pitch angle
        ax2.plot(current_time_test_list, rmse_pitch_test_list, color=color)
        ax2.set_ylabel('Pitch RMSE Test [deg]')
        ax2.legend()
        ax2.grid(True)
        # Plot rmse yaw angle
        ax3.plot(current_time_test_list, rmse_yaw_test_list, color=color)
        ax3.set_ylabel('Yaw RMSE Test [deg]')
        ax3.legend()
        ax3.grid(True)
        # Plot rmse angle
        ax4.plot(current_time_test_list, rmse_test_list)
        ax4.set_ylabel('Total RMSE Test [deg]')
        ax4.set_xlabel('T [sec]')
        ax4.legend()
        ax4.grid(True)
        plt.tight_layout()
        plt.show(block = False)

    
    ### Base Line Model
    # Lists to store results
    num_samples_baseline_list = []
    rmse_svd_baseline_list = []
    rmse_all_test_iterations_svd_baseline_list = [[] for _ in range(num_of_check_baseline_iterations)]
    rmse_gd_baseline_list = []
    rmse_baseline_centered_list = []
    rmse_roll_baseline_list = []
    rmse_pitch_baseline_list = []
    rmse_yaw_baseline_list = []
    rmse_gd_roll_baseline_list = []
    rmse_gd_pitch_baseline_list = []
    rmse_gd_yaw_baseline_list = []
    squared_error_roll_baseline_list = []
    squared_error_pitch_baseline_list = []
    squared_error_yaw_baseline_list = []
    squared_error_svd_baseline_list = []
    squared_error_gd_baseline_list = []
    squared_centered_error_baseline_list = []
    current_time_baseline_list = []
    euler_angles_svd_degrees_list = []
    euler_angles_gd_degrees_list = []
    euler_angles_centered_degrees_list = []

    if (config['test_baseline_model']):

        if (config['test_type'] == "real_data"):


            for num_samples in tqdm(range(2, real_dataset_len, 20)):  # Start from 2, increment by 5

                current_time = (num_samples / real_dataset_len) * real_dataset_len #because it 1hz - one sample per second

                #prepare the data
                v_imu_sampled = v_imu_body_real_data_full[:, 0:num_samples]
                v_dvl_sampled = v_dvl_real_data_full[:, 0:num_samples]
                euler_body_dvl_gt = euler_body_dvl_real_data_full[:, 0]

                # Acceleration-based Method: Run Gradient Descent solution

                # Velocity-based Method: Run SVD solution
                euler_angles_svd_rads = run_svd_solution_for_wahba_problem(v_imu_sampled, v_dvl_sampled)
                euler_angles_svd_degrees = np.degrees(euler_angles_svd_rads)
                euler_angles_svd_degrees_list.append(euler_angles_svd_degrees)

                # # Acceleration-based Method: Run gradient descent solution
                # euler_angles_gd_degrees = run_acc_gradient_descent(v_imu_sampled, v_dvl_sampled, omega_skew_imu_sampled,
                #                                                    a_imu_sampled)
                # euler_angles_gd_degrees_list.append(euler_angles_gd_degrees)


                squared_error_svd_baseline = squared_angular_difference(np.array(euler_angles_svd_degrees),
                                                                        euler_body_dvl_gt)
                squared_error_svd_baseline_list.append(squared_error_svd_baseline)

                # squared_error_gd_baseline = squared_angular_difference(np.array(euler_angles_gd_degrees),
                #                                                        euler_body_dvl_gt)
                # squared_error_gd_baseline_list.append(squared_error_gd_baseline)


                squared_error_roll_baseline = squared_angle_difference(euler_angles_svd_degrees[0],
                                                                       euler_body_dvl_gt[0])
                squared_error_roll_baseline_list.append(
                    squared_angle_difference(euler_angles_svd_degrees[0], euler_body_dvl_gt[0]))

                squared_error_pitch_baseline = squared_angle_difference(euler_angles_svd_degrees[1],
                                                                        euler_body_dvl_gt[1])
                squared_error_pitch_baseline_list.append(
                    squared_angle_difference(euler_angles_svd_degrees[1], euler_body_dvl_gt[1]))

                squared_error_yaw_baseline = squared_angle_difference(euler_angles_svd_degrees[2], euler_body_dvl_gt[2])
                squared_error_yaw_baseline_list.append(
                    squared_angle_difference(euler_angles_svd_degrees[2], euler_body_dvl_gt[2]))

                # Calculate RMSE
                rmse_svd = calculate_rmse(squared_error_svd_baseline)
                rmse_svd_roll = calculate_rmse(squared_error_roll_baseline)
                rmse_svd_pitch = calculate_rmse(squared_error_pitch_baseline)
                rmse_svd_yaw = calculate_rmse(squared_error_yaw_baseline)

                #rmse_gd = calculate_rmse(squared_error_gd_baseline)

                # Store results
                num_samples_baseline_list.append(num_samples)
                current_time_baseline_list.append(current_time)
                rmse_svd_baseline_list.append(rmse_svd)
                #rmse_baseline_centered_list.append(rmse_centered)
                rmse_roll_baseline_list.append(rmse_svd_roll)
                rmse_pitch_baseline_list.append(rmse_svd_pitch)
                rmse_yaw_baseline_list.append(rmse_svd_yaw)

                # rmse_gd_baseline_list.append(rmse_gd)
                # rmse_gd_roll_baseline_list.append(rmse_svd_roll)
                # rmse_gd_pitch_baseline_list.append(rmse_svd_pitch)
                # rmse_gd_yaw_baseline_list.append(rmse_svd_yaw)


        elif (config['test_type'] == "simulated_data"):


            # Calculate number of sample points
            num_sample_points = (single_dataset_len - 10) // 10   # = 117 for single_dataset_len = 1170
            num_samples_range = range(10, single_dataset_len, 10)  # This will give us consistent lengths

            # Initialize arrays to store all RMSE values across runs
            all_rmse_baseline = np.zeros((len(num_samples_range), num_of_check_baseline_iterations))
            all_rmse_roll_baseline = np.zeros((len(num_samples_range), num_of_check_baseline_iterations))
            all_rmse_pitch_baseline = np.zeros((len(num_samples_range), num_of_check_baseline_iterations))
            all_rmse_yaw_baseline = np.zeros((len(num_samples_range), num_of_check_baseline_iterations))
            current_time_baseline = []

            print(f"num_of_samples:{num_of_simulated_datasets}")


            for check_iter in range(num_of_check_baseline_iterations):

                print(f"curr_check_idx: {num_of_simulated_datasets-num_of_check_baseline_iterations+check_iter}")

                # Lists to store results
                # num_samples_baseline_list = []
                # rmse_svd_baseline_list = []
                # rmse_gd_baseline_list = []
                # rmse_baseline_centered_list = []
                # rmse_roll_baseline_list = []
                # rmse_pitch_baseline_list = []
                # rmse_yaw_baseline_list = []
                # rmse_gd_roll_baseline_list = []
                # rmse_gd_pitch_baseline_list = []
                # rmse_gd_yaw_baseline_list = []
                # squared_error_roll_baseline_list = []
                # squared_error_pitch_baseline_list = []
                # squared_error_yaw_baseline_list = []
                # squared_error_svd_baseline_list = []
                # squared_error_gd_baseline_list = []
                # squared_centered_error_baseline_list = []
                # current_time_baseline_list = []
                # euler_angles_svd_degrees_list = []
                # euler_angles_gd_degrees_list = []
                # euler_angles_centered_degrees_list = []

                current_time_baseline_list.clear()

                for num_samples in tqdm(range(10, single_dataset_len, 20)):  # should Start from 2, increment by 5

                    current_time = (num_samples / single_dataset_len) * single_dataset_duration_sec

                    # Sample the data
                    start_idx = (num_of_simulated_datasets - num_of_check_baseline_iterations + check_iter) * single_dataset_len
                    v_imu_sampled = v_imu_body_full[:, start_idx:start_idx + num_samples]
                    v_dvl_sampled = v_dvl_full[:, start_idx:start_idx + num_samples]
                    a_imu_sampled = a_imu_body_full[:, start_idx:start_idx + num_samples]
                    omega_skew_imu_sampled = omega_skew_imu_body_full[:, :, start_idx:start_idx + num_samples]

                    euler_body_dvl_gt = single_quaternion_to_euler(euler_body_dvl_full[:, start_idx])


                    # Acceleration-based Method: Run Gradient Descent solution


                    # Velocity-based Method: Run SVD solution
                    euler_angles_svd_rads = run_svd_solution_for_wahba_problem(v_imu_sampled, v_dvl_sampled)
                    euler_angles_svd_degrees = np.degrees(euler_angles_svd_rads)
                    euler_angles_svd_degrees_list.append(euler_angles_svd_degrees)

                    # Acceleration-based Method: Run gradient descent solution
                    # euler_angles_gd_degrees = run_acc_gradient_descent(v_imu_sampled, v_dvl_sampled, omega_skew_imu_sampled, a_imu_sampled)
                    # euler_angles_gd_degrees_list.append(euler_angles_gd_degrees)

                    # The paper shows better results when removing the mean (VEL-SVD-RMV)
                    v_imu_centered = v_imu_sampled - np.mean(v_imu_sampled, axis=1, keepdims=True)
                    v_dvl_centered = v_dvl_sampled - np.mean(v_dvl_sampled, axis=1, keepdims=True)
                    euler_angles_centered_rads = run_svd_solution_for_wahba_problem(v_imu_centered, v_dvl_centered)
                    euler_angles_centered_degrees = np.degrees(euler_angles_centered_rads)
                    euler_angles_centered_degrees_list.append(euler_angles_centered_degrees)

                    squared_error_svd_baseline = squared_angular_difference(np.array(euler_angles_svd_degrees), euler_body_dvl_gt)
                    squared_error_svd_baseline_list.append(squared_error_svd_baseline)

                    # squared_error_gd_baseline = squared_angular_difference(np.array(euler_angles_gd_degrees),euler_body_dvl_gt)
                    # squared_error_gd_baseline_list.append(squared_error_gd_baseline)

                    squared_centered_error_baseline = squared_angular_difference(np.array(euler_angles_centered_degrees), euler_body_dvl_gt)
                    squared_centered_error_baseline_list.append(squared_angular_difference(np.array(euler_angles_centered_degrees), euler_body_dvl_gt))

                    squared_error_roll_baseline = squared_angle_difference(euler_angles_svd_degrees[0], euler_body_dvl_gt[0])
                    squared_error_roll_baseline_list.append(squared_angle_difference(euler_angles_svd_degrees[0], euler_body_dvl_gt[0]))

                    squared_error_pitch_baseline = squared_angle_difference(euler_angles_svd_degrees[1], euler_body_dvl_gt[1])
                    squared_error_pitch_baseline_list.append(squared_angle_difference(euler_angles_svd_degrees[1], euler_body_dvl_gt[1]))

                    squared_error_yaw_baseline = squared_angle_difference(euler_angles_svd_degrees[2], euler_body_dvl_gt[2])
                    squared_error_yaw_baseline_list.append(squared_angle_difference(euler_angles_svd_degrees[2], euler_body_dvl_gt[2]))


                    # Calculate RMSE
                    rmse_svd = calculate_rmse(squared_error_svd_baseline)
                    rmse_centered = calculate_rmse(squared_centered_error_baseline)
                    rmse_svd_roll = calculate_rmse(squared_error_roll_baseline)
                    rmse_svd_pitch = calculate_rmse(squared_error_pitch_baseline)
                    rmse_svd_yaw = calculate_rmse(squared_error_yaw_baseline)


                    # rmse_gd = calculate_rmse(squared_error_gd_baseline)


                    # Store results
                    num_samples_baseline_list.append(num_samples)
                    current_time_baseline_list.append(current_time)
                    rmse_all_test_iterations_svd_baseline_list[check_iter].append(rmse_svd)
                    rmse_baseline_centered_list.append(rmse_centered)
                    rmse_roll_baseline_list.append(rmse_svd_roll)
                    rmse_pitch_baseline_list.append(rmse_svd_pitch)
                    rmse_yaw_baseline_list.append(rmse_svd_yaw)

                    # # rmse_gd_baseline_list.append(rmse_gd)
                    # rmse_gd_roll_baseline_list.append(rmse_gd_roll)
                    # rmse_gd_pitch_baseline_list.append(rmse_gd_pitch)
                    # rmse_gd_yaw_baseline_list.append(rmse_gd_yaw)

                plt.figure(figsize=(12, 8))
                # Plot total RMSE for baseline and test
                plt.plot(current_time_baseline_list, rmse_all_test_iterations_svd_baseline_list[check_iter], label='Baseline', linewidth=2)

                # Add labels and title
                plt.xlabel('T [sec]', fontsize=16)
                plt.ylabel('Alignment RMSE [deg]', fontsize=16)

                # Add legend
                plt.legend(fontsize=12)

                # Adjust layout
                plt.tight_layout()

                # Show plot
                plt.show(block=False)



            # Create a list to store the means
            rmse_mean_svd_baseline_list = []

            # Calculate mean for each position across all iterations
            for i in range(len(rmse_all_test_iterations_svd_baseline_list[0])):  # Assumes all sublists have same length
                values_at_position = [iteration[i] for iteration in rmse_all_test_iterations_svd_baseline_list]
                mean_at_position = np.mean(values_at_position)
                rmse_mean_svd_baseline_list.append(mean_at_position)

            rmse_svd_baseline_list = rmse_all_test_iterations_svd_baseline_list[-1]



        ####RMSE

            ######### Complete results graph
            # Create a single figure for total RMSE plot
            plt.figure(figsize=(12, 8))

            # Set colors
            baseline_color = 'blue'
            test_color = 'red'

            # Plot total RMSE for baseline and test
            plt.plot(current_time_baseline_list, rmse_svd_baseline_list, color=baseline_color, linestyle='-',
                     label='Baseline',
                     linewidth=2)

            # Create combined tick marks for both small and large intervals
            small_intervals = np.array([25, 50, 100])
            large_intervals = np.arange(0, 201, 50)  # 0 to 200 in steps of 50
            all_ticks = np.unique(np.concatenate([small_intervals, large_intervals]))

            # Set x-axis ticks
            plt.xticks(all_ticks)

            # Add labels and title
            plt.xlabel('T [sec]', fontsize=20)
            plt.ylabel('Alignment RMSE [deg]', fontsize=20)

            # Add primary grid (solid lines for 50-second intervals)
            plt.grid(True, which='major', linestyle='-', alpha=0.5)

            # Add secondary grid (dotted lines for 5-second intervals)
            for x in small_intervals:
                plt.axvline(x=x, color='gray', linestyle=':', alpha=0.5)

            # Add legend
            plt.legend(fontsize=16)

            # Adjust layout
            plt.tight_layout()

            # Show plot
            plt.show(block=False)








        # # Create a new figure with 3 subplots (one for each axis)
        # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 15), sharex=True)
        #
        # # Set the color to purple for ax1, ax2, and ax3
        # color = 'purple'
        #
        # # Plot rmse roll angle
        # ax1.plot(current_time_baseline_list, rmse_roll_baseline_list, color=color)
        # ax1.set_ylabel('Roll Baseline RMSE [deg]')
        # ax1.legend()
        # ax1.grid(True)
        #
        # # Plot rmse pitch angle
        # ax2.plot(current_time_baseline_list, rmse_pitch_baseline_list, color=color)
        # ax2.set_ylabel('Pitch Baseline RMSE [deg]')
        # ax2.legend()
        # ax2.grid(True)
        #
        # # Plot rmse yaw angle
        # ax3.plot(current_time_baseline_list, rmse_yaw_baseline_list, color=color)
        # ax3.set_ylabel('Yaw Baseline RMSE [deg]')
        # ax3.legend()
        # ax3.grid(True)
        #
        # # Plot rmse angle
        # ax4.plot(current_time_baseline_list, rmse_svd_baseline_list, color='red', label='RMSE SVD Baseline')
        # #ax4.plot(current_time_baseline_list, rmse_gd_baseline_list, color='blue', label='RMSE GD Baseline')
        # ax4.set_ylabel('Total Baseline RMSE [deg]')
        # ax4.set_xlabel('T [sec]')
        # ax4.legend()
        # ax4.grid(True)
        # plt.tight_layout()
        # plt.show(block = False)
#
#
######### Complete results graph
        # Create a single figure for total RMSE plot
        plt.figure(figsize=(12, 8))

        # Set colors
        baseline_color = 'blue'
        test_color = 'red'

        # Plot total RMSE for baseline and test
        plt.plot(current_time_baseline_list, rmse_svd_baseline_list, color=baseline_color, linestyle='-', label='Baseline',
                 linewidth=2)
        plt.scatter(current_time_test_list, rmse_test_list, color=test_color, marker='o', label='AligNet', s=100)

        # Create combined tick marks for both small and large intervals
        small_intervals = np.array([25, 50, 100])
        large_intervals = np.arange(0, 201, 50)  # 0 to 200 in steps of 50
        all_ticks = np.unique(np.concatenate([small_intervals, large_intervals]))

        # Set x-axis ticks
        plt.xticks(all_ticks)

        # Add labels and title
        plt.xlabel('T [sec]', fontsize=20)
        plt.ylabel('Alignment RMSE [deg]', fontsize=20)


        # Add primary grid (solid lines for 50-second intervals)
        plt.grid(True, which='major', linestyle='-', alpha=0.5)

        # Add secondary grid (dotted lines for 5-second intervals)
        for x in small_intervals:
            plt.axvline(x=x, color='gray', linestyle=':', alpha=0.5)

        # Add legend
        plt.legend(fontsize=16)

        # Adjust layout
        plt.tight_layout()

        # Show plot
        plt.show(block=False)

    # if (config['check_data']):
    #     real_data_lla_rad = np.array(real_data_gt_pd.iloc[:, 1:4].T)
    #     # Convert LLA to ECEF coordinates
    #     real_ecef_data = navpy.lla2ecef(real_data_lla_rad[0], real_data_lla_rad[1], real_data_lla_rad[2],latlon_unit='rad',alt_unit='m')
    #
    #     # Extract x, y, z coordinates from the ECEF data
    #     x = real_ecef_data[:, 0]
    #     y = real_ecef_data[:, 1]
    #     z = real_ecef_data[:, 2]
    #
    #     # Create a 3D plot
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #
    #     # Plot the trajectory
    #     ax.plot(x, y, z, label='Trajectory')
    #
    #     # Set labels and title
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     ax.set_title('3D Position in ECEF')
    #
    #     # Add a legend
    #     ax.legend()
    #
    #     # Display the plot
    #     plt.show(block = False)
    #     plt.savefig('3D_position_ecef')







        # real_data_imu_pd
        # real_data_dvl_pd
        # real_data_gt_pd
        # real_data_pd = pd.read_csv(os.path.join(data_path, f'{real_data_file_name}'), header=None)
        # real_data_time = np.array(real_data_pd.iloc[:, 0].T)
        # v_imu_body_real_data_full = np.array(real_data_pd.iloc[:, 4:7].T)
        # v_dvl_real_data_full = np.array(real_data_pd.iloc[:, 13:16].T)
        # euler_body_dvl_real_data_full = np.array(real_data_pd.iloc[:, 16:19].T)
        # real_dataset_len = len(v_imu_body_real_data_full[1])




        # lla_rad = np.array([simulated_data_pd['latitude'], simulated_data_pd['longitude'], simulated_data_pd['altitude']])
        # lla_deg = np.array([np.rad2deg(simulated_data_pd['latitude']), np.rad2deg(simulated_data_pd['longitude']), simulated_data_pd['altitude']])
        #
        # # Create a 3D plot
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        #
        # # Plot the trajectory
        # ax.plot(lla_deg[0], lla_deg[1], lla_deg[2], label='Trajectory')
        #
        # # Set labels and title
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.set_title('3D Position in lla deg')
        #
        # # Add a legend
        # ax.legend()
        #
        # # Display the plot
        # plt.show(block = False)
        # plt.savefig('3D_position_lla_deg')





        # fig , axes = plt.subplots(ncols = 1 , nrows = 3 , figsize = (15,8))
        # fig.suptitle('LLA deg seperatly')
        # axes[0].plot(lla_deg[0], label = 'Latitude')
        # axes[0].grid()
        # axes[1].plot(lla_deg[1], label='longitude')
        # axes[1].grid()
        # axes[2].plot(lla_deg[2], label='altitude')
        # axes[2].grid()
        # plt.show(block = False)
        #
        #
        # # # Convert LLA to ECEF coordinates
        # ecef_data = navpy.lla2ecef(lla_deg[0], lla_deg[1], lla_deg[2],latlon_unit='rad',alt_unit='m')
        #


'''
        # Create a new figure with 3 subplots (one for each axis)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

        # Plot X-axis velocities
        ax1.plot(time, v_gt_body_full[0], label='GT')
        ax1.plot(time, v_imu_body_full[0], label='EST')
        ax1.set_ylabel('Vx [m/s]')
        ax1.legend()
        ax1.grid(True)

        # Plot Y-axis velocities
        ax2.plot(time, v_gt_body_full[1], label='GT')
        ax2.plot(time, v_imu_body_full[1], label='EST')
        ax2.set_ylabel('Vy [m/s]')
        ax2.legend()
        ax2.grid(True)

        # Plot Z-axis velocities
        ax3.plot(time, v_gt_body_full[2], label='GT')
        ax3.plot(time, v_imu_body_full[2], label='EST')
        ax3.set_xlabel('T [sec]')
        ax3.set_ylabel('Vz [m/s]')
        ax3.legend()
        ax3.grid(True)

        #plt.suptitle('GT velocity in body[m/s] and IMU velocity in body[m/s] vs Time[sec]')
        plt.tight_layout()
        plt.show(block = False)

'''

        #curr_v_gt_ned_x = v_gt_body_full[0,:].T

        # dt = np.diff(time).mean()# or whatever your time step is
        # x_n = cumulative_trapezoid(v_gt_body_full[0,:], dx=dt, initial=0)
        # y_n = cumulative_trapezoid(v_gt_body_full[1,:], dx=dt, initial=0)
        # z_n = cumulative_trapezoid(v_gt_body_full[2,:], dx=dt, initial=0)


        # x_n = pos_ned[0,:]
        # y_n = pos_ned[1,:]
        # z_n = pos_ned[2,:]



        # fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(x_n, y_n, z_n, 'b-', label='AUV trajectory')
        # ax.set_xlabel('X position (m)')
        # ax.set_ylabel('Y position (m)')
        # ax.set_zlabel('Z position (m)')
        # ax.set_title('AUV Trajectory NED in 3D')
        # plt.legend()
        # plt.grid(True)
        # plt.show(block=False)




        # Plot trajectory




        # # Create a new figure with 3 subplots (one for each axis)
        # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        #
        # # Plot X-axis velocities
        # ax1.plot(time, v_gt_body_full[0], label='V_gt_x')
        # ax1.plot(time, v_dvl_body_full[0], label='V_dvl_x')
        # ax1.set_ylabel('X Velocity [m/s]')
        # ax1.legend()
        # ax1.grid(True)
        #
        # # Plot Y-axis velocities
        # ax2.plot(time, v_gt_body_full[1], label='V_gt_y')
        # ax2.plot(time, v_dvl_body_full[1], label='V_dvl_y')
        # ax2.set_ylabel('Y Velocity [m/s]')
        # ax2.legend()
        # ax2.grid(True)
        #
        # # Plot Z-axis velocities
        # ax3.plot(time, v_gt_body_full[2], label='V_gt_z')
        # ax3.plot(time, v_dvl_body_full[2], label='V_dvl_z')
        # ax3.set_xlabel('Time (sec)')
        # ax3.set_ylabel('Z Velocity [m/s]')
        # ax3.legend()
        # ax3.grid(True)
        #
        # plt.suptitle('GT velocity in body [m/s] and est dvl velocity in body [m/s] vs Time [sec]')
        # plt.tight_layout()
        # plt.show(block = False)
        # plt.savefig('v_GT_and_v_dvl_body_vs_Time.png')



# ##################### Use real data from AUV SNAPIR experiments ########################################################
#     if config['auv_snapir_data']:
#         experiment_index = config['experiment_index']
#         auv_data_path = "C:\\Users\\damar\\Desktop\\Thesis\\AUV_Snapir\\auv_snapir_nadav_recordings\\DataAUV"
#         auv_data_dvl_path = "C:\\Users\\damar\\Desktop\\Thesis\\AUV_Snapir\\auv_snapir_nadav_recordings\\DataAUV\\DVL1.txt"
#         auv_data_imu_path = "C:\\Users\\damar\\Desktop\\Thesis\\AUV_Snapir\\auv_snapir_nadav_recordings\\DataAUV\\IMU1.txt"
#         auv_data_nav_path = "C:\\Users\\damar\\Desktop\\Thesis\\AUV_Snapir\\auv_snapir_nadav_recordings\\DataAUV\\NAV1.txt"
#
#         # auv_data_files = [f for f in os.listdir(auv_data_path)]
#
#         # Read data from the .txt file
#         dvl_data_file_name = f'DVL{experiment_index}.txt'
#         imu_data_file_name = f'IMU{experiment_index}.txt'
#         nav_data_file_name = f'Nav{experiment_index}.txt'
#
#         auv_data_dvl_pd = pd.read_csv(os.path.join(auv_data_path, dvl_data_file_name), sep='\t')
#         auv_data_imu_pd = pd.read_csv(os.path.join(auv_data_path, imu_data_file_name), sep='\t')
#         auv_data_nav_pd = pd.read_csv(os.path.join(auv_data_path, nav_data_file_name), sep='\t')
#
#
#         lla_data = np.array([auv_data_nav_pd['latitude'], auv_data_nav_pd['longitude'], auv_data_nav_pd['altitude']])
#         lla_data = lla_data.T
#
#         # # Convert LLA to ECEF coordinates
#         # ecef_data = navpy.lla2ecef(auv_data_nav_pd['latitude'], auv_data_nav_pd['longitude'], auv_data_nav_pd['altitude'])
#         #
#         # # Extract x, y, z coordinates from the ECEF data
#         # x = ecef_data[:, 0]
#         # y = ecef_data[:, 1]
#         # z = ecef_data[:, 2]
#
#         # Create a 3D plot
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#
#         # Plot the trajectory
#         #ax.plot(x, y, z, label='Trajectory')
#
#         # # Set labels and title
#         # ax.set_xlabel('X')
#         # ax.set_ylabel('Y')
#         # ax.set_zlabel('Z')
#         # ax.set_title('3D Trajectory')
#
#         # Add a legend
#         #ax.legend()
#
#         # Display the plot
#         # plt.show(block = False)
#
#         # Create a new DataFrame with modified 'time' column
#         auv_data_imu_modified_pd = auv_data_imu_pd.copy()
#         auv_data_imu_modified_pd['time'] = auv_data_imu_modified_pd['time'].astype(str).str[:-2]
#
#
#         # Create a new DataFrame with modified 'time' column
#         auv_data_nav_modified_pd = auv_data_nav_pd.copy()
#         auv_data_nav_modified_pd['time'] = auv_data_nav_modified_pd['time'].astype(str).str[:-5]
#
#
#
#         # Create a new DataFrame to store the matched IMU data, without last 2 digits of the 'time' column
#         matched_imu_data = pd.DataFrame(columns=auv_data_imu_pd.columns)
#
#         # Create a new DataFrame to store the matched NAV data, without last 2 digits of the 'time' column
#         matched_nav_data = pd.DataFrame(columns=auv_data_nav_pd.columns)
#
#         for time_dvl in auv_data_dvl_pd['time']:
#             time_dvl_str = str(time_dvl)[:-5]
#
#             # Find the matching row in auv_data_imu_pd based on the 'Time' column with the last two characters as "don't care"
#             matching_rows = auv_data_nav_modified_pd[auv_data_nav_modified_pd['time'].astype(str) == time_dvl_str]
#
#             if not matching_rows.empty:
#                 # Get the first matching row
#                 matching_row = matching_rows.iloc[0]
#
#                 # Concatenate the matching row with matched_imu_data
#                 matched_nav_data = pd.concat([matched_nav_data, matching_row.to_frame().T], ignore_index=True)
#             else:
#                 print(f"No matching row found for time: {time_dvl_str}")
#
#
#         v_dvl_sampled = np.array([auv_data_dvl_pd['speedX'], auv_data_dvl_pd['speedY'], auv_data_dvl_pd['speedZ']])
#
#         v_nav_sampled_enu = np.array([matched_nav_data['speedNorth'], matched_nav_data['speedEast'], matched_nav_data['speedUp']])
#
#         euler_angles_ref_body = np.array([matched_nav_data['roll'], matched_nav_data['pitch'], matched_nav_data['heading']])
#
#         R_enu_ned = np.array([[0, 1, 0],
#                             [1, 0, 0],
#                             [0, 0, -1]])
#
#         v_nav_sampled_ned = np.dot(R_enu_ned, v_nav_sampled_enu)
#
#
#
#         R_ned_body = []
#         v_nav_sampled_body_list = []
#
#         for i, time_value in enumerate(euler_angles_ref_body[0]):
#             R_ned_body.append(euler_angles_to_rotation_matrix(euler_angles_ref_body[0][i], euler_angles_ref_body[1][i], euler_angles_ref_body[2][i]))
#             # R_ned_body_array = np.array(R_ned_body)
#             # print('check')
#
#         R_ned_body_array = np.array(R_ned_body)
#
#
#         for i in range(0, len(R_ned_body_array)):
#             v_nav_sampled_body_list.append(np.dot(R_ned_body_array[i], v_nav_sampled_ned[:,i]))
#             # print('check')
#
#
#         v_nav_sampled_body = np.array(v_nav_sampled_body_list)
#
#         v_nav_sampled_body = v_nav_sampled_body.transpose()
#         v_nav_sampled_body = v_nav_sampled_body.astype(np.float64)
#
#         # Run SVD solution for the Wahba problem
#         euler_angles_rads = run_svd_solution_for_wahba_problem(v_nav_sampled_body, v_dvl_sampled)
#         euler_angles_svd_degrees = np.degrees(euler_angles_rads)
#
#         print("euler_angles_svd_degrees: ")
#         print(euler_angles_svd_degrees)




if __name__ == '__main__':
    # Default configuration
    default_config = {
        'roll_gt_deg': 45,
        'pitch_gt_deg': 10,
        'yaw_gt_deg': 120,
    }

    # User-defined configuration (can be read from a config file or command-line arguments)
    user_config = {
        'roll_gt_deg': -179.9,
        'pitch_gt_deg': 0.2,
        'yaw_gt_deg': -44.3,
        'data_path': "C:\\Users\\damar\\MATLAB\\Projects\\modeling-and-simulation-of-an-AUV-in-Simulink-master\\Work",
        'test_type': 'simulated_data', # Set to "real_data" or "simulated_data"
        'train_model': True,  # Set to False to use the saved trained model
        'test_model': True,
        'test_baseline_model': True,
        'check_data': False,
        'simulated_data_file_name': 'simulated_data_output.csv',
        'trained_model_path': "C:\\Users\\damar\\MATLAB\\Projects\\modeling-and-simulation-of-an-AUV-in-Simulink-master\\Work\\trained_model",
        'real_data_file_name': 'real_data_output.csv',
        'real_data_trajectory_index': 9,
        'real_imu_file_name': 'IMU_trajectory.csv',
        'real_dvl_file_name': 'DVL_trajectory.csv',
        'real_gt_file_name': 'GT_trajectory.csv'

    }

    # orzi_euler_config = {
    #     'roll_gt_deg': -179.9,
    #     'pitch_gt_deg': 0.2,
    #     'yaw_gt_deg': -44.3,
    # }

    # Merge default and user configurations
    config = {**default_config, **user_config}

    main(config)

    plt.show()
