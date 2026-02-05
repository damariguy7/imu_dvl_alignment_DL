import math
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch
import random
from torch import nn

from sklearn.model_selection import train_test_split

from calc_error_functions import calc_squared_err_angles, calc_err_angles
from calc_mean_rmse_svd_degrees_per_num_samples import calc_mean_rmse_svd_degrees_per_num_samples
from euler_angles_to_rotation_matrix import euler_angles_to_rotation_matrix
from load_baseline_results_numpy import load_baseline_results_numpy
from resnet1d import Resnet1chDnet
from convert_lat_lon_to_meters import convert_lat_lon_to_meters
from rotation_matrix_log_so3 import rotation_matrix_log_so3
from plots import plot_max_error_comparison, print_baseline_results_at_windows
from evaluate_model_aoe import evaluate_model_aoe, calc_aoe_single_sample
from models import IMUDVLCNN
from loss_functions import EulerAnglesLoss
from run_svd_solution_for_wahba_problem import run_svd_solution_for_wahba_problem
from save_model import save_model
from split_data_properly import split_data_properly

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)


def train_model(model, train_loader, val_loader, optimizer, num_epochs, device,
                scheduler=None):
    model.to(device)
    best_val_loss = float('inf')
    patience = 1
    patience_counter = 0
    best_model_state = None
    criterion = EulerAnglesLoss()  # Using the specialized loss for Euler angles

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Calculate loss considering periodic nature of angles
            loss = criterion(outputs, labels)
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

    return model, best_val_loss


def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_targets = []
    mse_angle_list = []
    angles_error_list = []

    test_max_err = 0;

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Move to CPU immediately and detach
            outputs_cpu = outputs.detach().cpu().numpy()
            targets_cpu = targets.detach().cpu().numpy()

            # Calculate per-batch metrics
            for ii in range(len(targets_cpu)):
                angles_error = calc_err_angles(targets_cpu[ii], outputs_cpu[ii])
                angles_error_list.append(angles_error)

                curr_test_max_err = np.max(np.abs(angles_error))
                if curr_test_max_err > test_max_err:
                    test_max_err = curr_test_max_err

                mse_angle = calc_squared_err_angles(targets_cpu[ii], outputs_cpu[ii])
                mse_angle_list.append(mse_angle)



            # Store on CPU
            # all_predictions.append(outputs_cpu)
            # all_targets.append(targets_cpu)

            # Clear GPU cache periodically
            if batch_idx % 10 == 0:  # Adjust frequency as needed
                torch.cuda.empty_cache()

    # Calculate metrics
    # all_predictions = np.concatenate(all_predictions)
    # all_targets = np.concatenate(all_targets)

    mean_angles_error = np.mean(angles_error_list, axis=0)
    rmse = np.sqrt(np.mean(mse_angle_list))
    print(f'rmse is {rmse}')

    return rmse, mean_angles_error, test_max_err


class IMUDVLWindowedDataset(Dataset):
    def __init__(self, series, window_size):
        self.imu_series = torch.FloatTensor(series[0])
        self.dvl_series = torch.FloatTensor(series[1])

        # Convert euler angles to degrees if they're in radians
        # Assuming the input is in degrees (based on your data),
        # but we'll normalize to the range [-180, 180]
        euler_angles = torch.FloatTensor(series[2])

        # Normalize angles to [-180, 180] range
        euler_angles = ((euler_angles + 180) % 360) - 180

        self.euler_body_dvl_series = euler_angles
        self.window_size = window_size

    def __len__(self):
        return len(self.imu_series) - self.window_size

    def __getitem__(self, idx):
        imu_window = self.imu_series[idx:idx + self.window_size]
        dvl_window = self.dvl_series[idx:idx + self.window_size]
        euler_body_dvl_window = self.euler_body_dvl_series[idx:idx + self.window_size]

        # Combine IMU and DVL data
        input_data = torch.cat((imu_window, dvl_window), dim=1)

        # Return features (IMU and DVL data for the window) and target (Euler angles in degrees)
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

# # Acceleration-based Method: Run gradient descent solution
# euler_angles_gd_degrees = run_acc_gradient_descent(v_imu_sampled, v_dvl_sampled, omega_skew_imu_sampled,
#                                                    a_imu_sampled)
# euler_angles_gd_degrees_list.append(euler_angles_gd_degrees)

# squared_error_gd_baseline = squared_angular_difference(np.array(euler_angles_gd_degrees),euler_body_dvl_gt)

# squared_error_gd_baseline_list.append(squared_error_gd_baseline)



def main(config):
    # Example usage

    # plot_trajectory_path(3)
    # Plot the 3D trajectories
    # straight_data, turn_data = plot_simulation_trajectories()

    # plot_individual_2d_trajectories(straight_data, turn_data)

    # Optionally plot 2D projections
    # plot_2d_trajectories(straight_data, turn_data)

    # Plot 2D trajectories separately (X-Y only, no depth)
    # plot_2d_trajectories_separate(straight_data, turn_data)

    # Plot 3D trajectories separately (side by side)
    # plot_3d_trajectory_separate(straight_data, turn_data)

    # plot_accelerometer_data()
    # plot_gyroscope_data()

    # Main variables to run the simulation
    roll_gt_deg = config['roll_gt_deg']
    pitch_gt_deg = config['pitch_gt_deg']
    yaw_gt_deg = config['yaw_gt_deg']

    sample_freq = None
    single_dataset_duration_sec = None # should be same parameter value like in matlab simulation
    single_dataset_len = None # should be same parameter value like in matlab simulation

    if(config['test_type'] == 'simulated_data'):
        single_dataset_len = config['simulated_dataset_len']
        # single_dataset_duration_sec = config['simulated_dataset_duration_sec']
        sample_freq = 5
    elif(config['test_type'] == 'convex_data'):
        single_dataset_len = config['convex_dataset_len']
        # single_dataset_duration_sec = config['simulated_dataset_duration_sec']
        sample_freq = 5
    elif((config['test_type'] == 'transformed_real_data') or (config['test_type'] == 'simulated_imu_from_real_gt_data')):
        single_dataset_len = config['real_dataset_len']
        # single_dataset_duration_sec = config['real_dataset_duration_sec']
        sample_freq = 1

    data_path = config['data_path']


    if(config['test_type'] == 'convex_data'):
        file_name = config['convex_data_file_name']
        data_pd = pd.read_csv(os.path.join(data_path, "convex_data_output", f'{file_name}'), header=None)

    elif(config['test_type'] == 'simulated_data'):
        file_name = config['simulated_data_file_name']
        data_pd = pd.read_csv(os.path.join(data_path, "simulated_data_output", f'{file_name}'), header=None)

        # use for very large datasets
        #ddf = dd.read_csv(os.path.join(data_path, f'{file_name}'), header=None)


    elif(config['test_type'] == 'transformed_real_data'):
        file_name = config['transformed_real_data_file_name']
        data_pd = pd.read_csv(os.path.join(data_path, "transformed_real_data_output", f'{file_name}'), header=None)
        # simulated_imu_from_real_gt_data_file_name = config['simulated_imu_from_real_gt_data_file_name']
        # simulated_imu_from_real_gt_data_pd = pd.read_csv(os.path.join(data_path, f'{simulated_imu_from_real_gt_data_file_name}'), header=None)
    elif (config['test_type'] == 'simulated_imu_from_real_gt_data'):
        file_name = config['simulated_imu_from_real_gt_data_file_name']
        data_pd = pd.read_csv(os.path.join(data_path, "simulated_imu_from_real_gt_data_output", f'{file_name}'), header=None)


    real_data_file_name = config['real_data_file_name']
    trained_model_base_path = config['trained_model_path']
    window_sizes_sec = config['window_sizes_sec']
    batch_size = config['batch_size']


    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert degrees to radians
    roll_gt_rads = np.radians(roll_gt_deg)
    pitch_gt_rads = np.radians(pitch_gt_deg)
    yaw_gt_rads = np.radians(yaw_gt_deg)

    # Convert Euler angles to rotation matrix
    rotation_matrix_ins_to_dvl = euler_angles_to_rotation_matrix(roll_gt_rads, pitch_gt_rads, yaw_gt_rads)


    ##prepare real data dataset - for sim2real model
    real_data_pd = pd.read_csv(os.path.join(data_path, f'{real_data_file_name}'), header=None)
    v_imu_body_real_data_full = np.array(real_data_pd.iloc[:, 1:4].T)
    v_dvl_real_data_full = np.array(real_data_pd.iloc[:, 4:7].T)
    euler_body_dvl_real_data_full = np.array(real_data_pd.iloc[:, 7:10].T)
    real_dataset_len = len(v_imu_body_real_data_full[1])

    current_time_test_list = []
    rmse_test_list = []
    aoe_test_list = []
    mean_error_angles_test_list = []
    v_imu_dvl_test_real_data_list = []

    # Calculate number of sequences
    num_of_simulated_datasets = len(data_pd) // single_dataset_len

    if not config['test_convex']:

        train_sequences, val_sequences, test_sequences, test_indices = split_data_properly(
            data_pd=data_pd,
            num_sequences=num_of_simulated_datasets,
            sequence_length=single_dataset_len,
            train_size=0.6,
            val_size=0.2
        )

        # Replace your existing lists with the new split sequences
        v_imu_dvl_train_series_list = train_sequences
        v_imu_dvl_valid_series_list = val_sequences
        v_imu_dvl_test_series_list = test_sequences

        v_imu_dvl_test_real_data_list.append(
            [v_imu_body_real_data_full.T, v_dvl_real_data_full.T, euler_body_dvl_real_data_full.T])

    # train model ##################################################################
    if config['train_model']:
        for i in window_sizes_sec:
            print(f'train with window of {i} seconds')
            # Parameters
            num_of_samples_window_size = i * sample_freq

            train_loader = windowed_dataset(
                series_list=v_imu_dvl_train_series_list,
                window_size=num_of_samples_window_size,
                batch_size=batch_size,
                shuffle=True
            )

            val_loader = windowed_dataset(
                series_list=v_imu_dvl_valid_series_list,
                window_size=num_of_samples_window_size,
                batch_size=batch_size,
                shuffle=True
            )

            model = Resnet1chDnet()
            # model = IMUDVLCNN()

            # # Loss function
            # criterion = EulerAnglesLoss()

            # Optimizer with weight decay (L2 regularization)

            optimizer = optim.Adam(model.parameters(), lr=0.0000001, weight_decay=0.01)

            #
            # Learning rate scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=0,
                verbose=True
            )

            # Train the model with the improved training function
            train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                num_epochs=25,
                device=device,
                scheduler=scheduler
                # l2_lambda=0.01
            )

            # Save the model and store its path
            model_path = save_model(model, config['test_type'], i, trained_model_base_path)

    # Test Model section #################
    if config['test_model']:
        for window_size in window_sizes_sec:
            print(f'\nEvaluating model with window size {window_size}')

            # Construct the specific model path for this window size
            test_type = config['test_type']
            #model_path = os.path.join(trained_model_base_path, f'imu_dvl_model_{test_type}_window_{window_size}.pth')
            #model_path = os.path.join(trained_model_base_path, f'imu_dvl_model_simulated_data_window_{window_size}.pth')
            # model_path = os.path.join(trained_model_base_path, f'imu_dvl_model_simulated_data_lawn_mower1_22_+2_ba_100_bg_0_01_window_{window_size}.pth')
            # model_path = os.path.join(trained_model_base_path, f'imu_dvl_model_simulated_data_long_turn_17_+0_3125_ba_100_bg_1_window_{window_size}.pth')
            # model_path = os.path.join(trained_model_base_path, f'imu_dvl_model_simulated_data_straight_line_17_+0_3125_ba_100_bg_1_window_{window_size}.pth')
            # model_path = os.path.join(trained_model_base_path, f'imu_dvl_model_transformed_real_data_alignet_window_{window_size}.pth')
            # model_path = os.path.join(trained_model_base_path, f'imu_dvl_model_transformed_real_data_alignet_traj7_17_+0_3125_window_{window_size}.pth')
            model_path = os.path.join(trained_model_base_path, f'imu_dvl_model_transformed_real_data_traj7_17_+0_3125_window_{window_size}.pth')
            # model_path = os.path.join(trained_model_base_path, f'imu_dvl_model_transformed_real_data_traj11_16_+0_3333_window_{window_size}.pth')
            # Load model
            model = Resnet1chDnet()
            # model = IMUDVLCNN()
            print(f"model.load_state_dict")
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.to(device)
            model.eval()

            print(f"test_loader")
            if (config['test_type'] == 'simulated_data') or (config['test_type'] == 'transformed_real_data') or (config['test_type'] == 'simulated_imu_from_real_gt_data'):
                # Create test loader for this window size
                test_loader = windowed_dataset(
                    series_list=v_imu_dvl_test_series_list,
                    window_size=window_size,
                    batch_size=batch_size,
                    shuffle=True
                )

            elif config['test_type'] == 'real_data':
                # Create test loader for this window size
                test_loader = windowed_dataset(
                    series_list=v_imu_dvl_test_real_data_list,
                    window_size=window_size,
                    batch_size=batch_size,
                    shuffle=True
                )

            # aoe_rads, aoe_degrees = evaluate_model_aoe(model, test_loader, device)
            # aoe_test_list.append(aoe_degrees)

            print(f"eval")
            # Evaluate model
            rmse, mean_angles_error, test_max_err = evaluate_model(
                model, test_loader, device
            )

            print(f"test_max_err: {test_max_err:.4f}")

            rmse_test_list.append(rmse)
            mean_error_angles_test_list.append(mean_angles_error)
            current_time = window_size
            current_time_test_list.append(current_time)
            print(f"Test RMSE: {rmse:.4f}")


    ### Test Baseline Model section ##########
    if config['test_baseline_model']:

        mean_rmse_svd_degrees_per_num_samples_list, mean_error_angles_svd_degrees_per_num_samples_list, svd_time_list, svd_max_err = calc_mean_rmse_svd_degrees_per_num_samples(v_imu_dvl_test_series_list, sample_freq, config)
        print(f"svd_max_err: {svd_max_err:.4f}")
        # mean_aoe_svd_degrees_per_num_samples_list, mean_angles_error_svd_degrees_per_num_samples_list, svd_time_list = calc_mean_aoe_svd_degrees_per_num_samples(v_imu_dvl_test_series_list, sample_freq, config)

        # # movmean_window_size = 15 # Best for transformed real traj 8
        # movmean_window_size = 1  #
        # mean_rmse_svd_degrees_per_num_samples_list_smoothed = movmean(mean_rmse_svd_degrees_per_num_samples_list,
        #                                                               movmean_window_size)
        #
        # mean_rmse_svd_degrees_per_num_samples_list = mean_rmse_svd_degrees_per_num_samples_list_smoothed

        # save_rmse_baseline_results_numpy(mean_rmse_svd_degrees_per_num_samples_list, svd_time_list,rmse_test_list, current_time_test_list, config)
        # save_aoe_baseline_results_numpy(mean_aoe_svd_degrees_per_num_samples_list, svd_time_list, aoe_test_list, current_time_test_list, config)

        # plot_results_graph_angle_me_net_and_angle_me_svd(svd_time_list, mean_error_angles_svd_degrees_per_num_samples_list, current_time_test_list, mean_error_angles_test_list)
        # plot_results_graph_rmse_net_and_rmse_svd(svd_time_list, mean_rmse_svd_degrees_per_num_samples_list, current_time_test_list, rmse_test_list)
        # plot_results_graph_aoe_net_and_aoe_svd(svd_time_list, mean_aoe_svd_degrees_per_num_samples_list, current_time_test_list, aoe_test_list)

        loaded_rmse_baseline_results_list, loaded_timeline_baseline_results_list = load_baseline_results_numpy(config)
        print("Baseline RMSE")
        # Create labels for the baselines (optional)
        baseline_labels = [
            f"SVD (baseline)", f"ResAlignNet - sim2real (ours)", f"ResAlignNet (ours)"
            # f"SVD (baseline) - Navigation Grade IMU", f"ResAlignNet (ours) - Navigation Grade IMU", f"SVD (baseline) - Tactical Grade IMU", f"ResAlignNet (ours) - Tactical Grade IMU"
        ]

        # # Plot all baselines on the same plot
        # plot_all_baseline_results(
        #     loaded_timeline_baseline_results_list,
        #     loaded_rmse_baseline_results_list,
        #     labels=baseline_labels,
        #     current_time_test_list=current_time_test_list if config['test_model'] else None,
        #     rmse_test_list=rmse_test_list if config['test_model'] else None
        # )

        # Print the results at specific window sizes
        print_baseline_results_at_windows(
            loaded_timeline_baseline_results_list,
            loaded_rmse_baseline_results_list,
            window_sizes_sec,
            labels=baseline_labels
        )

        plot_max_error_comparison()

    if config['test_convex']:

        num_of_simulated_datasets = len(data_pd) // single_dataset_len

        # Extract data arrays
        v_imu_body_full = np.array(data_pd.iloc[:, 1:4].T).T
        v_dvl_full = np.array(data_pd.iloc[:, 4:7].T).T
        euler_body_dvl_gt_full = np.array(data_pd.iloc[:, 7:10])
        # est_eul_ned_to_body_rad_full = np.array(data_pd.iloc[:, 10:13].T)
        # est_acc_eb_b_full = np.array(data_pd.iloc[:, 13:16].T)
        b_acc_full = np.array(data_pd.iloc[:, 10].T)
        b_gyro_full = np.array(data_pd.iloc[:, 11].T)

        # Create all sequences sequentially (no shuffling)
        all_sequences = []
        step_size = sample_freq * 1
        for idx in range(num_of_simulated_datasets):
            # print(f"num_of_simulated_datasets: {idx}/{num_of_simulated_datasets}")
            min_rse_value = 10000.0
            start_idx = idx * single_dataset_len
            end_idx = start_idx + single_dataset_len - single_dataset_len % 10
            b_acc_sampled = b_acc_full[start_idx]
            b_gyro_sampled = b_gyro_full[start_idx]

            for num_samples in range(start_idx + 60*step_size, end_idx, step_size):
                # print(f"num_samples: {num_samples}/{end_idx}")
                v_imu_sampled = v_imu_body_full[start_idx:num_samples, :]
                v_dvl_sampled = v_dvl_full[start_idx:num_samples, :]
                euler_body_dvl_gt = euler_body_dvl_gt_full[start_idx, :]

                euler_angles_svd_rads = run_svd_solution_for_wahba_problem(v_imu_sampled.T, v_dvl_sampled.T)
                euler_angles_svd_degrees = np.degrees(euler_angles_svd_rads)
                # print(f"euler_angles_svd_rads: {euler_angles_svd_degrees}")
                svd_squared_err_angles = calc_squared_err_angles(np.array(euler_angles_svd_degrees), euler_body_dvl_gt)
                # svd_sum_squared_err_angles = np.sum(svd_squared_err_angles)
                svd_root_squared_err_angles = np.sqrt(np.sum(svd_squared_err_angles))
                # print(f"svd_root_squared_err_angles: {svd_root_squared_err_angles}")
                if svd_root_squared_err_angles < min_rse_value:
                    min_rse_value = svd_root_squared_err_angles

            all_sequences.append([
                b_acc_sampled / 1000,  # Convert to mg
                b_gyro_sampled,
                min_rse_value
            ])

        # Extract data for plotting
        acc_bias_values = np.array([seq[0] for seq in all_sequences])
        gyro_bias_values = np.array([seq[1] for seq in all_sequences])
        svd_mse_values = np.array([seq[2] for seq in all_sequences])

        print(f"Number of data points: {len(all_sequences)}")
        print(f"Unique accelerometer bias values: {len(np.unique(acc_bias_values))}")
        print(f"Unique gyroscope bias values: {len(np.unique(gyro_bias_values))}")

        # Create 2D heatmap plot (as suggested by supervisor)
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)

        # Initialize variables for specific points
        specific_z_values = [0, 0]  # Default values

        # Check if we have enough points for interpolation
        if len(all_sequences) >= 4 and len(np.unique(acc_bias_values)) >= 2 and len(
                np.unique(gyro_bias_values)) >= 2:
            try:
                # Create a regular grid for interpolation
                acc_min, acc_max = acc_bias_values.min(), acc_bias_values.max()
                gyro_min, gyro_max = gyro_bias_values.min(), gyro_bias_values.max()

                # Create meshgrid for heatmap plotting
                acc_grid = np.linspace(acc_min, acc_max, 50)
                gyro_grid = np.linspace(gyro_min, gyro_max, 50)
                ACC_GRID, GYRO_GRID = np.meshgrid(acc_grid, gyro_grid)

                # Interpolate MSE values on the grid using scipy's griddata
                from scipy.interpolate import griddata

                # Stack the original points
                points = np.column_stack((acc_bias_values, gyro_bias_values))
                grid_points = np.column_stack((ACC_GRID.ravel(), GYRO_GRID.ravel()))

                # Try cubic interpolation first, fall back to linear if it fails
                try:
                    MSE_GRID = griddata(points, svd_mse_values, grid_points, method='cubic', fill_value=np.nan)
                except:
                    MSE_GRID = griddata(points, svd_mse_values, grid_points, method='linear', fill_value=np.nan)

                MSE_GRID = MSE_GRID.reshape(ACC_GRID.shape)

                # Create 2D heatmap with contourf
                heatmap = ax.contourf(ACC_GRID, GYRO_GRID, MSE_GRID, levels=20, cmap='viridis', alpha=0.9)

                # Add contour lines for better visualization
                contour_lines = ax.contour(ACC_GRID, GYRO_GRID, MSE_GRID, levels=10, colors='white', alpha=0.6,
                                           linewidths=0.8)
                ax.clabel(contour_lines, inline=True, fontsize=32, fmt='%.1f')

                # Add colorbar
                cbar = plt.colorbar(heatmap, ax=ax)
                cbar.set_label('Alignment RMSE [deg] - SVD (baseline)', fontsize=32)
                cbar.ax.tick_params(labelsize=30)

                plot_type = "2D Heatmap"

                # Interpolate Z values for the specific points we want to highlight
                specific_points = np.array([[1.0, 10.0], [0.1, 1.0]])  # [acc_bias, gyro_bias]

                try:
                    specific_z_values = griddata(points, svd_mse_values, specific_points, method='cubic')
                    # If cubic fails or returns NaN, try linear
                    if np.any(np.isnan(specific_z_values)):
                        specific_z_values = griddata(points, svd_mse_values, specific_points, method='linear')
                    # If still NaN, use nearest neighbor
                    if np.any(np.isnan(specific_z_values)):
                        specific_z_values = griddata(points, svd_mse_values, specific_points, method='nearest')
                except:
                    specific_z_values = griddata(points, svd_mse_values, specific_points, method='nearest')

            except Exception as e:
                print(f"Heatmap interpolation failed: {e}")
                print("Falling back to scatter plot...")
                # Fall back to scatter plot
                scatter = ax.scatter(acc_bias_values, gyro_bias_values, c=svd_mse_values,
                                     cmap='viridis', s=100, alpha=0.8)

                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Alignment MSE [deg] - SVD (baseline)', fontsize=32)
                cbar.ax.tick_params(labelsize=30)

                plot_type = "2D Scatter Plot (insufficient data for heatmap)"

                # For scatter plot, find nearest points or use average Z
                specific_z_values = [np.mean(svd_mse_values), np.mean(svd_mse_values)]

        else:
            print("Not enough data points for heatmap interpolation. Using scatter plot...")
            # Use scatter plot for insufficient data
            scatter = ax.scatter(acc_bias_values, gyro_bias_values, c=svd_mse_values,
                                 cmap='viridis', s=100, alpha=0.8)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Alignment MSE [deg] - SVD (baseline)', fontsize=32)
            cbar.ax.tick_params(labelsize=30)

            plot_type = "2D Scatter Plot (insufficient data for heatmap)"

            # For scatter plot, use average Z values for the specific points
            specific_z_values = [np.mean(svd_mse_values), np.mean(svd_mse_values)]

        # Add the two specific highlighted points
        # Point 1: acc_bias = 1 mg, gyro_bias = 10
        ax.scatter([1.0], [10.0], c='green', s=250, alpha=1.0, edgecolors='black', linewidth=3,
                   label='acc=1mg, gyro=10°/h', marker='o', zorder=5)

        # Point 2: acc_bias = 0.1 mg, gyro_bias = 1
        ax.scatter([0.1], [1.0], c='blue', s=250, alpha=1.0, edgecolors='black', linewidth=3,
                   label='acc=0.1mg, gyro=1°/h', marker='s', zorder=5)

        # Set labels with larger font sizes
        ax.set_xlabel('Accelerometer Bias [mg]', fontsize=34)
        ax.set_ylabel('Gyroscope Bias [deg/hour]', fontsize=34)

        # Set title
        # ax.set_title(f'Convex Optimization Results: {plot_type}', fontsize=28, pad=20)

        # Improve tick label sizes
        ax.tick_params(axis='x', labelsize=32)
        ax.tick_params(axis='y', labelsize=32)

        # Add grid for better readability
        ax.grid(True, alpha=0.3)

        # Add legend with all markers
        ax.legend(loc='lower right', fontsize=26)

        # Adjust layout
        plt.tight_layout()

        # Show plot
        plt.show()

        # Print some statistics including the specific points
        print(f"\nConvex Optimization Results Summary:")
        print(f"Number of sequences: {len(all_sequences)}")
        print(f"Accelerometer bias range: {min(acc_bias_values):.4f} to {max(acc_bias_values):.4f} mg")
        print(f"Gyroscope bias range: {min(gyro_bias_values):.4f} to {max(gyro_bias_values):.4f} deg/hour")
        print(f"SVD MSE range: {min(svd_mse_values):.4f} to {max(svd_mse_values):.4f} deg")
        print(f"Best (minimum) SVD MSE: {min(svd_mse_values):.4f} deg")
        print(f"\nHighlighted Points:")
        print(f"Green point (acc=1mg, gyro=10°/h): MSE = {specific_z_values[0]:.4f} deg")
        print(f"Blue point (acc=0.1mg, gyro=1°/h): MSE = {specific_z_values[1]:.4f} deg")

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
        # 'window_sizes_sec': [5, 25, 50, 75, 100, 125, 150],
        # 'window_sizes_sec': [75],
        # 'window_sizes_sec': [5],
        'window_sizes_sec': [5,25,50,75,100],
        # 'window_sizes_sec': [175],
        'batch_size': 32,
        # 'simulated_dataset_len': 1554, # - straight descent1 - important!! you have to update it, from the data output file, every time you change dataset
        # 'simulated_dataset_len': 1077, # - straight line - important!! you have to update it, from the data output file, every time you change dataset
        # 'simulated_dataset_len': 1075, # - straight line1 - important!! you have to update it, from the data output file, every time you change dataset
        # 'simulated_dataset_len': 5791, # - straight line1 - important!! you have to update it, from the data output file, every time you change dataset
        # 'simulated_dataset_len': 1503, # - straight line1 - important!! you have to update it, from the data output file, every time you change dataset
        'simulated_dataset_len': 1027,# - long turn - important!! you have to update it, from the data output file, every time you change dataset
        # 'simulated_dataset_len': 1585, # - straight_dive_n_turn - important!! you have to update it, from the data output file, every time you change dataset
        # 'simulated_dataset_len': 1509, # - straight_turn_left_180 - important!! you have to update it, from the data output file, every time you change dataset
        # 'simulated_dataset_len': 3234, # - lawn_mower_1 - important!! you have to update it, from the data output file, every time you change dataset - should be length of single trajectory
        # 'simulated_dataset_len': 2907, # - lawn_mower_4 - important!! you have to update it, from the data output file, every time you change dataset - should be length of single trajectory
        # 'simulated_dataset_len': 3703, # - lawn_mower_5 - important!! you have to update it, from the data output file, every time you change dataset - should be length of single trajectory
        # 'simulated_dataset_len': 1638, # - squared - important!! you have to update it, from the data output file, every time you change dataset - should be length of single trajectory
        # 'real_dataset_len': 400, ## important!! you have to update it, from the data output file, every time you change dataset - should be length of single trajectory
        'real_dataset_len': 200, # - traj7, traj11 - important!! you have to update it, from the data output file, every time you change dataset - should be length of single trajectory
        # 'real_dataset_len': 185, # - traj10_a - important!! you have to update it, from the data output file, every time you change dataset - should be length of single trajectory
        # 'real_dataset_len': 249, # - traj10_b - important!! you have to update it, from the data output file, every time you change dataset - should be length of single trajectory
        #'real_dataset_len': 199, # - traj10_c - important!! you have to update it, from the data output file, every time you change dataset - should be length of single trajectory
        'convex_dataset_len': 2054, # - turn pattern simulated for convex -  important!! you have to update it, from the data output file, and every time you change dataset
        'data_path': "C:\\Users\\damar\\MATLAB\\Projects\\modeling-and-simulation-of-an-AUV-in-Simulink-master\\Work",
        'test_type': 'transformed_real_data',  # Set to "convex_data" or "simulated_data" or "transformed_real_data" or "simulated_imu_from_real_gt_data" or "real_data"
        'model': 'ResAligNet', # "AligNet", "ResAligNet", "MLP", "BeamsNet"
        'train_model': False,
        'test_model': True,
        'test_baseline_model': True,
        'test_convex': False,
        'trained_model_path': "C:\\Users\\damar\\MATLAB\\Projects\\modeling-and-simulation-of-an-AUV-in-Simulink-master\\Work\\trained_model",
        #'simulated_data_file_name': 'simulated_data_output.csv',
        'simulated_data_file_name': 'simulated_data_output_long_turn_17_+0_3125_ba_real_bg_10.csv',
        # 'simulated_data_file_name': 'simulated_data_output_long_turn_17_+2_8125_ba_real_bg_10.csv',
        # imu_dvl_model_simulated_data_straight_line_17_ + 2_8125_ba_real_bg_10_window_75
        # 'simulated_data_file_name': 'simulated_data_output.csv',
        'convex_data_file_name': 'convex_data_output.csv',
        'real_data_file_name': 'real_data_output.csv',
        # 'transformed_real_data_file_name': 'transformed_real_data_output.csv',
        'transformed_real_data_file_name': 'transformed_real_data_output_traj7_17_+0_3125.csv',
        # 'transformed_real_data_file_name': 'transformed_real_data_output_traj11_16_+0_3333.csv',
        # 'transformed_real_data_file_name': 'transformed_real_data_output_traj11_26_+1.csv',
        'simulated_imu_from_real_gt_data_file_name': 'simulated_imu_from_real_gt_data_output.csv',
        # 'saved_rmse_results_file_name': 'simulated_data_long_turn_17_+0_3125_ba_real_bg_10',
        'saved_rmse_results_file_name': 'sim2real_transformed_real_data_traj7_17_+0_3125',
        # 'saved_rmse_results_file_name': 'sim2real_transformed_real_data_output_traj11_16_+0_3333',
        # 'saved_rmse_results_file_name': 'transformed_real_data_output_traj11_16_+0_3333',
        # 'saved_rmse_results_file_name': 'transformed_real_data_traj7_11_+2_8125',
        # 'loaded_rmse_results_file_names_list': ['baseline_transformed_real_data_traj12_22_+0_227','sim2real_transformed_real_data_traj12_22_+0_227', 'transformed_real_data_traj12_22_+0_227'],
        'loaded_rmse_results_file_names_list': ['baseline_transformed_real_data_traj7_17_+0_3125', 'sim2real_transformed_real_data_traj7_17_+0_3125', 'transformed_real_data_traj7_17_+0_3125'],
        # 'loaded_rmse_results_file_names_list': ['baseline_transformed_real_data_traj11_16_+0_3333', 'sim2real_transformed_real_data_traj11_16_+0_3333', 'transformed_real_data_traj11_16_+0_3333'],
        # 'loaded_rmse_results_file_names_list': ['baseline_simulated_data_long_turn_17_+0_3125_ba_100_bg_1','simulated_data_long_turn_17_+0_3125_ba_100_bg_1', 'baseline_simulated_data_long_turn_17_+0_3125_ba_real_bg_10','simulated_data_long_turn_17_+0_3125_ba_real_bg_10'],
        # 'loaded_rmse_results_file_names_list': ['baseline_simulated_data_long_turn_17_+0_3125_ba_100_bg_1','simulated_data_long_turn_17_+0_3125_ba_100_bg_1', 'baseline_simulated_data_long_turn_17_+0_3125_ba_real_bg_10','simulated_data_long_turn_17_+0_3125_ba_real_bg_10'],
        'eval_metric': 'AOE',  ##'RMSE', 'AOE'
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


