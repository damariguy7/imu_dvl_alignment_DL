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
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
from sklearn.model_selection import train_test_split
from math import cos, radians
from mpl_toolkits.mplot3d import Axes3D
import dask.dataframe as dd


# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


__all__ = [
    "ResNet",
    "resnet18_1d",
    "resnet34_1d",
    "resnet50_1d",
    "resnet101_1d",
    "resnet152_1d",
    "resnext50_32x4d_1d",
    "resnext101_32x8d_1d",
    "wide_resnet50_2_1d",
    "wide_resnet101_2_1d",
]


# model_urls = {
#     "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
#     "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
#     "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
#     "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
#     "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
#     "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
#     "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
#     "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
#     "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
# }


def convert_lat_lon_to_meters(data):
    """
    Convert latitude/longitude from radians to meters (local ENU coordinates)
    Assumes data columns are: [time, lat_rad, lon_rad, alt_m]
    """
    # Extract lat, lon, alt
    lat_rad = data.iloc[:, 1].values
    lon_rad = data.iloc[:, 2].values
    alt_m = data.iloc[:, 3].values

    # Use first point as reference (origin)
    lat0 = lat_rad[0]
    lon0 = lon_rad[0]

    # Earth radius in meters
    R = 6378137.0  # WGS84 equatorial radius

    # Convert to local ENU coordinates (East, North, Up)
    # East (X) - longitude difference
    east = (lon_rad - lon0) * R * np.cos(lat0)

    # North (Y) - latitude difference
    north = (lat_rad - lat0) * R

    # Up (Z) - altitude (already in meters, but negative for depth)
    up = -alt_m  # Negative because we want depth below surface

    return east, north, up


def plot_3d_trajectory_separate(straight_data, turn_data):
    """
    Plot each trajectory in 3D separately - with proper coordinate conversion
    """
    if straight_data is None or turn_data is None:
        print("Error: No data available for plotting")
        return

    # Convert lat/lon coordinates to meters
    straight_x, straight_y, straight_z = convert_lat_lon_to_meters(straight_data)
    turn_x, turn_y, turn_z = convert_lat_lon_to_meters(turn_data)

    # Create two separate 3D plots
    fig = plt.figure(figsize=(16, 6))

    # Plot 1: Straight line trajectory
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(straight_x, straight_y, straight_z, 'b-', linewidth=2, label='Straight Line')
    ax1.set_xlabel('East [m]')
    ax1.set_ylabel('North [m]')
    ax1.set_zlabel('Depth [m]')
    ax1.set_title('Trajectory #1 - Straight Line (3D)')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Turn trajectory
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(turn_x, turn_y, turn_z, 'r--', linewidth=2, label='Right Turn')
    ax2.set_xlabel('East [m]')
    ax2.set_ylabel('North [m]')
    ax2.set_zlabel('Depth [m]')
    ax2.set_title('Trajectory #2 - Right Turn (3D)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_individual_3d_trajectories(straight_data, turn_data):
    """
    Plot each trajectory in completely separate 3D figures - with coordinate conversion
    """
    if straight_data is None or turn_data is None:
        print("Error: No data available for plotting")
        return

    # Convert lat/lon coordinates to meters
    straight_x, straight_y, straight_z = convert_lat_lon_to_meters(straight_data)
    turn_x, turn_y, turn_z = convert_lat_lon_to_meters(turn_data)

    # Plot straight line trajectory
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot(straight_x, straight_y, straight_z, 'b-', linewidth=3, label='Straight Line')
    ax1.set_xlabel('East [m]')
    ax1.set_ylabel('North [m]')
    ax1.set_zlabel('Depth [m]')
    ax1.set_title('Trajectory #1 - Straight Line (3D)')
    ax1.legend()
    ax1.grid(True)
    plt.show()

    # Plot turn trajectory
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot(turn_x, turn_y, turn_z, 'r-', linewidth=3, label='Right Turn')
    ax2.set_xlabel('East [m]')
    ax2.set_ylabel('North [m]')
    ax2.set_zlabel('Depth [m]')
    ax2.set_title('Trajectory #2 - Right Turn (3D)')
    ax2.legend()
    ax2.grid(True)
    plt.show()


def plot_2d_trajectories_separate(straight_data, turn_data):
    """
    Plot each trajectory in 2D separately (X-Y only, no depth) - with coordinate conversion
    """
    if straight_data is None or turn_data is None:
        print("Error: No data available for plotting")
        return

    # Convert lat/lon coordinates to meters
    straight_x, straight_y, straight_z = convert_lat_lon_to_meters(straight_data)
    turn_x, turn_y, turn_z = convert_lat_lon_to_meters(turn_data)

    # Create two separate 2D plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Straight line trajectory
    ax1.plot(straight_x, straight_y, 'b-', linewidth=2)
    ax1.set_xlabel('East [m]')
    ax1.set_ylabel('North [m]')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')

    # Plot 2: Turn trajectory
    ax2.plot(turn_x, turn_y, 'r-', linewidth=2)
    ax2.set_xlabel('East [m]')
    ax2.set_ylabel('North [m]')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()


def plot_individual_2d_trajectories(straight_data, turn_data):
    """
    Plot each trajectory in completely separate 2D figures (X-Y only) - with coordinate conversion
    """
    if straight_data is None or turn_data is None:
        print("Error: No data available for plotting")
        return

    # Convert lat/lon coordinates to meters
    straight_x, straight_y, straight_z = convert_lat_lon_to_meters(straight_data)
    turn_x, turn_y, turn_z = convert_lat_lon_to_meters(turn_data)

    # Plot straight line trajectory
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.plot(straight_x, straight_y, 'b-', linewidth=3)
    ax1.set_xlabel('East [m]')
    ax1.set_ylabel('North [m]')
    ax1.set_title('Trajectory #1 - Straight Line (2D)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    plt.show()

    # Plot turn trajectory
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.plot(turn_x, turn_y, 'r-', linewidth=3)
    ax2.set_xlabel('East [m]')
    ax2.set_ylabel('North [m]')
    ax2.set_title('Trajectory #2 - Right Turn (2D)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    plt.show()

def plot_2d_trajectories(straight_data, turn_data):
    """
    Optional: Create 2D plots showing X-Y, X-Z, and Y-Z projections
    """
    if straight_data is None or turn_data is None:
        return

    # Extract position data
    straight_x = straight_data.iloc[:, 1].values
    straight_y = straight_data.iloc[:, 2].values
    straight_z = straight_data.iloc[:, 3].values

    turn_x = turn_data.iloc[:, 1].values
    turn_y = turn_data.iloc[:, 2].values
    turn_z = turn_data.iloc[:, 3].values

    # Create 2D projection plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # X-Y projection (top view)
    axes[0, 0].plot(straight_x, straight_y, 'b-', linewidth=2, label='Straight line')
    axes[0, 0].plot(turn_x, turn_y, 'r--', linewidth=2, label='Long turn')
    axes[0, 0].set_xlabel('East [m]')
    axes[0, 0].set_ylabel('North [m]')
    axes[0, 0].set_title('Top View (X-Y)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].axis('equal')

    # X-Z projection (side view)
    axes[0, 1].plot(straight_x, straight_z, 'b-', linewidth=2, label='Straight line')
    axes[0, 1].plot(turn_x, turn_z, 'r--', linewidth=2, label='Long turn')
    axes[0, 1].set_xlabel('East [m]')
    axes[0, 1].set_ylabel('Depth [m]')
    axes[0, 1].set_title('Side View (X-Z)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Y-Z projection (front view)
    axes[1, 0].plot(straight_y, straight_z, 'b-', linewidth=2, label='Straight line')
    axes[1, 0].plot(turn_y, turn_z, 'r--', linewidth=2, label='Long turn')
    axes[1, 0].set_xlabel('North [m]')
    axes[1, 0].set_ylabel('Depth [m]')
    axes[1, 0].set_title('Front View (Y-Z)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Time vs Z (depth profile)
    time_straight = straight_data.iloc[:, 0].values
    time_turn = turn_data.iloc[:, 0].values

    axes[1, 1].plot(time_straight, straight_z, 'b-', linewidth=2, label='Straight line')
    axes[1, 1].plot(time_turn, turn_z, 'r--', linewidth=2, label='Long turn')
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Depth [m]')
    axes[1, 1].set_title('Depth Profile vs Time')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

#
# def plot_accelerometer_data():
#     """
#     Plot accelerometer X, Y, Z values over time from IMU trajectory file.
#
#     Args:
#         data_path (str): Path to the data directory
#         file_name (str): Name of the IMU trajectory CSV file
#     """
#     # Construct full file path
#     data_path = "C:\\Users\\damar\\MATLAB\\Projects\\modeling-and-simulation-of-an-AUV-in-Simulink-master\\Work"
#     file = os.path.join(data_path, 'IMU_trajectory7.csv')
#     file_path = os.path.join(data_path, file)
#
#     try:
#         # Read the CSV file with header row
#         df = pd.read_csv(file_path, header=0)
#
#         # Extract data columns
#         time = df.iloc[:, 0].values  # First column: time [sec]
#         acc_x = df.iloc[:, 1].values * 100  # Second column: ACC X [m/s²]
#         acc_y = df.iloc[:, 2].values * 100  # Third column: ACC Y [m/s²]
#         acc_z = df.iloc[:, 3].values * 100 # Fourth column: ACC Z [m/s²]
#
#         # Create the plot
#         fig, ax = plt.subplots(figsize=(12, 8))
#
#         # Plot each accelerometer axis
#         ax.plot(time, acc_x, 'r-', linewidth=1.5, label='ACC X')
#         ax.plot(time, acc_y, 'g-', linewidth=1.5, label='ACC Y')
#         ax.plot(time, acc_z, 'b-', linewidth=1.5, label='ACC Z')
#
#         # Set labels and title
#         ax.set_xlabel('Time [sec]', fontsize=35)
#         ax.set_ylabel('Specific Force [m/s²]', fontsize=35)
#
#         # Add grid and legend
#         ax.grid(True, alpha=0.3)
#         ax.legend(fontsize=35)
#
#         # Improve tick label size
#         ax.tick_params(axis='both', which='major', labelsize=30)
#
#         # Adjust layout
#         plt.tight_layout()
#
#         # Show the plot
#         plt.show()
#
#         return fig, ax
#
#     except FileNotFoundError:
#         print(f"Error: File {file_path} not found.")
#         return None, None
#     except Exception as e:
#         print(f"Error reading file {file_path}: {str(e)}")
#         return None, None
#
#
# def plot_gyroscope_data():
#
#     # Construct full file path
#     data_path = "C:\\Users\\damar\\MATLAB\\Projects\\modeling-and-simulation-of-an-AUV-in-Simulink-master\\Work"
#     file = os.path.join(data_path, 'IMU_trajectory111.csv')
#     file_path = os.path.join(data_path, file)
#
#     try:
#         # Read the CSV file with header row
#         df = pd.read_csv(file_path, header=0)
#
#         # Extract data columns
#         time = df.iloc[:, 0].values  # First column: time [sec]
#         gyro_x = df.iloc[:, 4].values * 100  # Second column: ACC X [m/s²]
#         gyro_y = df.iloc[:, 5].values * 100  # Third column: ACC Y [m/s²]
#         gyro_z = df.iloc[:, 6].values * 100 # Fourth column: ACC Z [m/s²]
#
#         # Create the plot
#         fig, ax = plt.subplots(figsize=(12, 8))
#
#         # Plot each accelerometer axis
#         ax.plot(time, gyro_x, 'r-', linewidth=1.5, label='GYRO X')
#         ax.plot(time, gyro_y, 'g-', linewidth=1.5, label='GYRO Y')
#         ax.plot(time, gyro_z, 'b-', linewidth=1.5, label='GYRO Z')
#
#         # Set labels and title
#         ax.set_xlabel('Time [sec]', fontsize=35)
#         ax.set_ylabel('Angular Rate [rad/sec]', fontsize=35)
#
#         # Add grid and legend - moved to lower right corner
#         ax.grid(True, alpha=0.3)
#         ax.legend(fontsize=20, loc='upper right')
#
#         # Improve tick label size
#         ax.tick_params(axis='both', which='major', labelsize=30)
#
#         # Adjust layout
#         plt.tight_layout()
#
#         # Show the plot
#         plt.show()
#
#         return fig, ax
#
#     except FileNotFoundError:
#         print(f"Error: File {file_path} not found.")
#         return None, None
#     except Exception as e:
#         print(f"Error reading file {file_path}: {str(e)}")
#         return None, None


# def plot_gyroscope_data():
#     """
#     Plot gyroscope X, Y, Z values over time from IMU trajectory file.
#
#     Reads columns 5, 6, 7 for XYZ gyro values in rad/sec from the same IMU file.
#     """
#     # Construct full file path
#     data_path = "C:\\Users\\damar\\MATLAB\\Projects\\modeling-and-simulation-of-an-AUV-in-Simulink-master\\Work"
#     file = os.path.join(data_path, 'IMU_trajectory11.csv')
#     file_path = os.path.join(data_path, file)
#
#     try:
#         # Read the CSV file with header row
#         df = pd.read_csv(file_path, header=0)
#
#         # Extract data columns
#         time = df.iloc[:, 0].values  # First column: time [sec]
#         gyro_x = df.iloc[:, 4].values * 100  # Fifth column: GYRO X [rad/sec]
#         gyro_y = df.iloc[:, 5].values * 100  # Sixth column: GYRO Y [rad/sec]
#         gyro_z = df.iloc[:, 6].values * 100  # Seventh column: GYRO Z [rad/sec]
#
#         # Create the plot
#         fig, ax = plt.subplots(figsize=(12, 8))
#
#         # Plot each gyroscope axis
#         ax.plot(time, gyro_x, 'r-', linewidth=1.5, label='GYRO X')
#         ax.plot(time, gyro_y, 'g-', linewidth=1.5, label='GYRO Y')
#         ax.plot(time, gyro_z, 'b-', linewidth=1.5, label='GYRO Z')
#
#         # Set labels and title
#         ax.set_xlabel('Time [sec]', fontsize=35)
#         ax.set_ylabel('Angular Rate [rad/sec]', fontsize=35)
#
#
#         # Improve Y-axis precision
#         # Get the data range to set appropriate tick spacing
#         all_gyro_data = np.concatenate([gyro_x, gyro_y, gyro_z])
#         data_min = np.min(all_gyro_data)
#         data_max = np.max(all_gyro_data)
#         data_range = data_max - data_min
#
#         # Set more precise Y-axis ticks based on data range
#         # Gyroscope data is typically in smaller ranges than accelerometer
#         if data_range < 0.1:
#             # Very small range - use 0.01 spacing
#             tick_spacing = 0.01
#         elif data_range < 0.5:
#             # Small range - use 0.05 spacing
#             tick_spacing = 0.05
#         elif data_range < 1.0:
#             # Small-medium range - use 0.1 spacing
#             tick_spacing = 0.1
#         elif data_range < 5.0:
#             # Medium range - use 0.5 spacing
#             tick_spacing = 0.5
#         else:
#             # Large range - use 1.0 spacing
#             tick_spacing = 1.0
#
#         # Create custom Y-axis ticks
#         y_min = np.floor(data_min / tick_spacing) * tick_spacing
#         y_max = np.ceil(data_max / tick_spacing) * tick_spacing
#         y_ticks = np.arange(y_min, y_max + tick_spacing, tick_spacing)
#
#         ax.set_yticks(y_ticks)
#
#         # Format Y-axis labels with appropriate decimal precision
#         if tick_spacing <= 0.01:
#             ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
#         elif tick_spacing <= 0.1:
#             ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
#         elif tick_spacing < 1:
#             ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
#         else:
#             ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
#
#         # Add minor ticks for even finer granularity
#         ax.yaxis.set_minor_locator(ticker.MultipleLocator(tick_spacing / 2))
#
#         # Add grid and legend
#         ax.grid(True, alpha=0.3)
#         ax.grid(True, which='minor', alpha=0.1)  # Add minor grid
#         ax.legend(fontsize=35)
#
#         # Improve tick label size
#         ax.tick_params(axis='both', which='major', labelsize=30)
#
#         # Adjust layout
#         plt.tight_layout()
#
#         # Show the plot
#         plt.show()
#
#         return fig, ax
#
#     except FileNotFoundError:
#         print(f"Error: File {file_path} not found.")
#         return None, None
#     except Exception as e:
#         print(f"Error reading file {file_path}: {str(e)}")
#         return None, None

def plot_simulation_trajectories():
    """
    Plot two simulation trajectories: straight line and turn

    """

    # Data path and file names
    data_path = "C:\\Users\\damar\\MATLAB\\Projects\\modeling-and-simulation-of-an-AUV-in-Simulink-master\\Work"
    straight_line_file = os.path.join(data_path, 'GT_trajectory11.csv')
    long_turn_file = os.path.join(data_path, 'GT_trajectory7.csv')
    # straight_line_file = os.path.join(data_path, 'simulation_straight_line_time_and_llh_position_gt_traj.csv')
    # long_turn_file = os.path.join(data_path, 'simulation_long_turn_time_and_llh_position_gt_traj.csv')
    convert_to_meters = True

    try:
        # Read the CSV files
        straight_data = pd.read_csv(straight_line_file, header=0)
        turn_data = pd.read_csv(long_turn_file, header=0)

        print("Straight line data columns:", straight_data.columns.tolist())
        print("Turn data columns:", turn_data.columns.tolist())
        print("Straight line data shape:", straight_data.shape)
        print("Turn data shape:", turn_data.shape)

        # Handle coordinate conversion based on flag
        if convert_to_meters:
            print("Converting lat/lon coordinates to meters...")
            # Convert lat/lon coordinates to meters for both trajectories
            straight_x, straight_y, straight_z = convert_lat_lon_to_meters(straight_data)
            turn_x, turn_y, turn_z = convert_lat_lon_to_meters(turn_data)
        else:
            print("Using data as-is (assuming already in meters)...")
            # Assume data is already in meters - extract X, Y, Z columns
            # You may need to adjust column names based on your CSV structure
            straight_x = straight_data.iloc[:, 1].values  # Assuming X is 2nd column
            straight_y = straight_data.iloc[:, 2].values  # Assuming Y is 3rd column
            straight_z = straight_data.iloc[:, 3].values  # Assuming Z is 4th column

            turn_x = turn_data.iloc[:, 1].values  # Assuming X is 2nd column
            turn_y = turn_data.iloc[:, 2].values  # Assuming Y is 3rd column
            turn_z = turn_data.iloc[:, 3].values  # Assuming Z is 4th column

        # Create 2D plot (X-Y projection - top view)
        fig, ax = plt.subplots(figsize=(12, 9))

        # # Plot trajectories with different styles
        ax.plot(straight_x, straight_y,
                color='blue', linewidth=2, linestyle='-', label='Trajectory #1')

        ax.plot(turn_x, turn_y, color='red', linewidth=2, linestyle='--')
        ax.plot(turn_x, turn_y, color='red', linewidth=2, linestyle='--', label='Trajectory #2')

        # Set labels
        ax.set_xlabel('East[m]', fontsize=30)
        ax.set_ylabel('North[m]', fontsize=30)

        # Make axis tick labels bigger
        ax.tick_params(axis='both', which='major', labelsize=40)

        # # Add legend with position based on conversion flag
        if convert_to_meters:
            ax.legend(fontsize=24, loc='upper right')
        else:
            ax.legend(fontsize=24, loc='lower right')

        # Add grid
        ax.grid(True, alpha=0.3)

        # Set equal aspect ratio for accurate representation
        ax.set_aspect('equal', adjustable='box')

        # Print trajectory statistics in meters
        print("\n=== Trajectory Statistics (in meters) ===")
        print(f"Straight line trajectory:")
        print(f"  Points: {len(straight_x)}")
        print(f"  East range: {np.min(straight_x):.2f} to {np.max(straight_x):.2f} m")
        print(f"  North range: {np.min(straight_y):.2f} to {np.max(straight_y):.2f} m")
        print(f"  Depth range: {np.min(straight_z):.2f} to {np.max(straight_z):.2f} m")

        print(f"\nLong turn trajectory:")
        print(f"  Points: {len(turn_x)}")
        print(f"  East range: {np.min(turn_x):.2f} to {np.max(turn_x):.2f} m")
        print(f"  North range: {np.min(turn_y):.2f} to {np.max(turn_y):.2f} m")
        print(f"  Depth range: {np.min(turn_z):.2f} to {np.max(turn_z):.2f} m")

        plt.tight_layout()
        plt.show()

        return straight_data, turn_data

    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        print("Please make sure the following files are in the directory:")
        print(f"- {straight_line_file}")
        print(f"- {long_turn_file}")
        return None, None
    except Exception as e:
        print(f"Error reading or plotting data: {e}")
        return None, None
# def convert_lat_lon_to_meters(data):
#     """
#     Convert latitude/longitude from radians to meters (local ENU coordinates)
#     Assumes data columns are: [time, lat_rad, lon_rad, alt_m]
#     """
#     # Extract lat, lon, alt
#     lat_rad = data.iloc[:, 1].values
#     lon_rad = data.iloc[:, 2].values
#     alt_m = data.iloc[:, 3].values
#
#     # Use first point as reference (origin)
#     lat0 = lat_rad[0]
#     lon0 = lon_rad[0]
#
#     # Earth radius in meters
#     R = 6378137.0  # WGS84 equatorial radius
#
#     # Convert to local ENU coordinates (East, North, Up)
#     # East (X) - longitude difference
#     east = (lon_rad - lon0) * R * np.cos(lat0)
#
#     # North (Y) - latitude difference
#     north = (lat_rad - lat0) * R
#
#     # Up (Z) - altitude (already in meters, but negative for depth)
#     up = -alt_m  # Negative because we want depth below surface
#
#     return east, north, up


def plot_3d_trajectory_separate(straight_data, turn_data):
    """
    Plot each trajectory in 3D separately - with proper coordinate conversion
    """
    if straight_data is None or turn_data is None:
        print("Error: No data available for plotting")
        return

    # Convert lat/lon coordinates to meters
    straight_x, straight_y, straight_z = convert_lat_lon_to_meters(straight_data)
    turn_x, turn_y, turn_z = convert_lat_lon_to_meters(turn_data)

    # Create two separate 3D plots
    fig = plt.figure(figsize=(16, 6))

    # Plot 1: Straight line trajectory
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(straight_x, straight_y, straight_z, 'b-', linewidth=2, label='Straight Line')
    ax1.set_xlabel('East [m]')
    ax1.set_ylabel('North [m]')
    ax1.set_zlabel('Depth [m]')
    ax1.set_title('Trajectory #1 - Straight Line (3D)')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Turn trajectory
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(turn_x, turn_y, turn_z, 'r--', linewidth=2, label='Right Turn')
    ax2.set_xlabel('East [m]')
    ax2.set_ylabel('North [m]')
    ax2.set_zlabel('Depth [m]')
    ax2.set_title('Trajectory #2 - Right Turn (3D)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_individual_3d_trajectories(straight_data, turn_data):
    """
    Plot each trajectory in completely separate 3D figures - with coordinate conversion
    """
    if straight_data is None or turn_data is None:
        print("Error: No data available for plotting")
        return

    # Convert lat/lon coordinates to meters
    straight_x, straight_y, straight_z = convert_lat_lon_to_meters(straight_data)
    turn_x, turn_y, turn_z = convert_lat_lon_to_meters(turn_data)

    # Plot straight line trajectory
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot(straight_x, straight_y, straight_z, 'b-', linewidth=3, label='Straight Line')
    ax1.set_xlabel('East [m]')
    ax1.set_ylabel('North [m]')
    ax1.set_zlabel('Depth [m]')
    ax1.set_title('Trajectory #1 - Straight Line (3D)')
    ax1.legend()
    ax1.grid(True)
    plt.show()

    # Plot turn trajectory
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot(turn_x, turn_y, turn_z, 'r-', linewidth=3, label='Right Turn')
    ax2.set_xlabel('East [m]')
    ax2.set_ylabel('North [m]')
    ax2.set_zlabel('Depth [m]')
    ax2.set_title('Trajectory #2 - Right Turn (3D)')
    ax2.legend()
    ax2.grid(True)
    plt.show()


def plot_2d_trajectories_separate(straight_data, turn_data):
    """
    Plot each trajectory in 2D separately (X-Y only, no depth) - with coordinate conversion
    """
    if straight_data is None or turn_data is None:
        print("Error: No data available for plotting")
        return

    # Convert lat/lon coordinates to meters
    straight_x, straight_y, straight_z = convert_lat_lon_to_meters(straight_data)
    turn_x, turn_y, turn_z = convert_lat_lon_to_meters(turn_data)

    # Create two separate 2D plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Straight line trajectory
    ax1.plot(straight_x, straight_y, 'b-', linewidth=2, label='Straight Line')
    ax1.set_xlabel('East [m]')
    ax1.set_ylabel('North [m]')
    ax1.set_title('Trajectory #1 - Straight Line (2D)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')

    # Plot 2: Turn trajectory
    ax2.plot(turn_x, turn_y, 'r-', linewidth=2, label='Right Turn')
    ax2.set_xlabel('East [m]')
    ax2.set_ylabel('North [m]')
    ax2.set_title('Trajectory #2 - Right Turn (2D)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()


def plot_individual_2d_trajectories(straight_data, turn_data):
    """
    Plot each trajectory in completely separate 2D figures (X-Y only) - with coordinate conversion
    """
    if straight_data is None or turn_data is None:
        print("Error: No data available for plotting")
        return

    # Convert lat/lon coordinates to meters
    straight_x, straight_y, straight_z = convert_lat_lon_to_meters(straight_data)
    turn_x, turn_y, turn_z = convert_lat_lon_to_meters(turn_data)

    # Plot straight line trajectory
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.plot(straight_x, straight_y, 'b-', linewidth=3, label='Straight Line')
    ax1.set_xlabel('East [m]')
    ax1.set_ylabel('North [m]')
    ax1.set_title('Trajectory #1 - Straight Line (2D)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    plt.show()

    # Plot turn trajectory
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.plot(turn_x, turn_y, 'r-', linewidth=3, label='Right Turn')
    ax2.set_xlabel('East [m]')
    ax2.set_ylabel('North [m]')
    ax2.set_title('Trajectory #2 - Right Turn (2D)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    plt.show()



def convert_lat_lon_to_meters(data):
    """
    Convert latitude/longitude from radians to meters (local ENU coordinates)
    Assumes data columns are: [time, lat_rad, lon_rad, alt_m]
    """
    # Extract lat, lon, alt
    lat_rad = data.iloc[:, 1].values
    lon_rad = data.iloc[:, 2].values
    alt_m = data.iloc[:, 3].values

    # Use first point as reference (origin)
    lat0 = lat_rad[0]
    lon0 = lon_rad[0]

    # Earth radius in meters
    R = 6378137.0  # WGS84 equatorial radius

    # Convert to local ENU coordinates (East, North, Up)
    # East (X) - longitude difference
    east = (lon_rad - lon0) * R * np.cos(lat0)

    # North (Y) - latitude difference
    north = (lat_rad - lat0) * R

    # Up (Z) - altitude (already in meters, but negative for depth)
    up = -alt_m  # Negative because we want depth below surface

    return east, north, up


def plot_3d_trajectory_separate(straight_data, turn_data):
    """
    Plot each trajectory in 3D separately - with proper coordinate conversion
    """
    if straight_data is None or turn_data is None:
        print("Error: No data available for plotting")
        return

    # Convert lat/lon coordinates to meters
    straight_x, straight_y, straight_z = convert_lat_lon_to_meters(straight_data)
    turn_x, turn_y, turn_z = convert_lat_lon_to_meters(turn_data)

    # Create two separate 3D plots
    fig = plt.figure(figsize=(16, 6))

    # Plot 1: Straight line trajectory
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(straight_x, straight_y, straight_z, 'b-', linewidth=2, label='Straight Line')
    ax1.set_xlabel('East [m]')
    ax1.set_ylabel('North [m]')
    ax1.set_zlabel('Depth [m]')
    ax1.set_title('Trajectory #1 - Straight Line (3D)')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Turn trajectory
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(turn_x, turn_y, turn_z, 'r--', linewidth=2, label='Right Turn')
    ax2.set_xlabel('East [m]')
    ax2.set_ylabel('North [m]')
    ax2.set_zlabel('Depth [m]')
    ax2.set_title('Trajectory #2 - Right Turn (3D)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_individual_3d_trajectories(straight_data, turn_data):
    """
    Plot each trajectory in completely separate 3D figures - with coordinate conversion
    """
    if straight_data is None or turn_data is None:
        print("Error: No data available for plotting")
        return

    # Convert lat/lon coordinates to meters
    straight_x, straight_y, straight_z = convert_lat_lon_to_meters(straight_data)
    turn_x, turn_y, turn_z = convert_lat_lon_to_meters(turn_data)

    # Plot straight line trajectory
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot(straight_x, straight_y, straight_z, 'b-', linewidth=3, label='Straight Line')
    ax1.set_xlabel('East [m]')
    ax1.set_ylabel('North [m]')
    ax1.set_zlabel('Depth [m]')
    ax1.set_title('Trajectory #1 - Straight Line (3D)')
    ax1.legend()
    ax1.grid(True)
    plt.show()

    # Plot turn trajectory
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot(turn_x, turn_y, turn_z, 'r-', linewidth=3, label='Right Turn')
    ax2.set_xlabel('East [m]')
    ax2.set_ylabel('North [m]')
    ax2.set_zlabel('Depth [m]')
    ax2.set_title('Trajectory #2 - Right Turn (3D)')
    ax2.legend()
    ax2.grid(True)
    plt.show()


def plot_2d_trajectories_separate(straight_data, turn_data):
    """
    Plot each trajectory in 2D separately (X-Y only, no depth) - with coordinate conversion
    """
    if straight_data is None or turn_data is None:
        print("Error: No data available for plotting")
        return

    # Convert lat/lon coordinates to meters
    straight_x, straight_y, straight_z = convert_lat_lon_to_meters(straight_data)
    turn_x, turn_y, turn_z = convert_lat_lon_to_meters(turn_data)

    # Create two separate 2D plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Straight line trajectory
    ax1.plot(straight_x, straight_y, 'b-', linewidth=2, label='Straight Line')
    ax1.set_xlabel('East [m]')
    ax1.set_ylabel('North [m]')
    ax1.set_title('Trajectory #1 - Straight Line (2D)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')

    # Plot 2: Turn trajectory
    ax2.plot(turn_x, turn_y, 'r-', linewidth=2, label='Right Turn')
    ax2.set_xlabel('East [m]')
    ax2.set_ylabel('North [m]')
    ax2.set_title('Trajectory #2 - Right Turn (2D)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()


def plot_individual_2d_trajectories(straight_data, turn_data):
    """
    Plot each trajectory in completely separate 2D figures (X-Y only) - with coordinate conversion
    """
    if straight_data is None or turn_data is None:
        print("Error: No data available for plotting")
        return

    # Convert lat/lon coordinates to meters
    straight_x, straight_y, straight_z = convert_lat_lon_to_meters(straight_data)
    turn_x, turn_y, turn_z = convert_lat_lon_to_meters(turn_data)

    # Plot straight line trajectory
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.plot(straight_x, straight_y, 'b-', linewidth=3, label='Straight Line')
    ax1.set_xlabel('East [m]')
    ax1.set_ylabel('North [m]')
    ax1.set_title('Trajectory #1 - Straight Line (2D)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    plt.show()

    # Plot turn trajectory
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.plot(turn_x, turn_y, 'r-', linewidth=3, label='Right Turn')
    ax2.set_xlabel('East [m]')
    ax2.set_ylabel('North [m]')
    ax2.set_title('Trajectory #2 - Right Turn (2D)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    plt.show()



def plot_2d_trajectories(straight_data, turn_data):
    """
    Optional: Create 2D plots showing X-Y, X-Z, and Y-Z projections
    """
    if straight_data is None or turn_data is None:
        return

    # Extract position data
    straight_x = straight_data.iloc[:, 1].values
    straight_y = straight_data.iloc[:, 2].values
    straight_z = straight_data.iloc[:, 3].values

    turn_x = turn_data.iloc[:, 1].values
    turn_y = turn_data.iloc[:, 2].values
    turn_z = turn_data.iloc[:, 3].values

    # Create 2D projection plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # X-Y projection (top view)
    axes[0, 0].plot(straight_x, straight_y, 'b-', linewidth=2, label='Straight line')
    axes[0, 0].plot(turn_x, turn_y, 'r--', linewidth=2, label='Long turn')
    axes[0, 0].set_xlabel('East [m]')
    axes[0, 0].set_ylabel('North [m]')
    axes[0, 0].set_title('Top View (X-Y)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].axis('equal')

    # X-Z projection (side view)
    axes[0, 1].plot(straight_x, straight_z, 'b-', linewidth=2, label='Straight line')
    axes[0, 1].plot(turn_x, turn_z, 'r--', linewidth=2, label='Long turn')
    axes[0, 1].set_xlabel('East [m]')
    axes[0, 1].set_ylabel('Depth [m]')
    axes[0, 1].set_title('Side View (X-Z)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Y-Z projection (front view)
    axes[1, 0].plot(straight_y, straight_z, 'b-', linewidth=2, label='Straight line')
    axes[1, 0].plot(turn_y, turn_z, 'r--', linewidth=2, label='Long turn')
    axes[1, 0].set_xlabel('North [m]')
    axes[1, 0].set_ylabel('Depth [m]')
    axes[1, 0].set_title('Front View (Y-Z)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Time vs Z (depth profile)
    time_straight = straight_data.iloc[:, 0].values
    time_turn = turn_data.iloc[:, 0].values

    axes[1, 1].plot(time_straight, straight_z, 'b-', linewidth=2, label='Straight line')
    axes[1, 1].plot(time_turn, turn_z, 'r--', linewidth=2, label='Long turn')
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Depth [m]')
    axes[1, 1].set_title('Depth Profile vs Time')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

def plot_trajectory_path(index,
                         data_path="C:\\Users\\damar\\MATLAB\\Projects\\modeling-and-simulation-of-an-AUV-in-Simulink-master\\Work"):
    """
    Plot trajectory path from GT_trajectory file in meters.

    Args:
        index (int): Index number for the trajectory file
        data_path (str): Path to the data directory
    """

    # Construct filename
    filename = f'GT_trajectory{index}.csv'
    filepath = os.path.join(data_path, filename)

    try:
        # Read the CSV file with header row
        df = pd.read_csv(filepath, header=0)

        # Extract data (skip header row by using header=0)
        time = df.iloc[:, 0].values  # Time [sec]
        longitude_rad = df.iloc[:, 1].values  # Longitude [rad]
        latitude_rad = df.iloc[:, 2].values  # Latitude [rad]
        altitude = df.iloc[:, 3].values  # Altitude [m]

        # Convert lat/lon from radians to degrees for reference point calculation
        latitude_deg = np.degrees(latitude_rad)
        longitude_deg = np.degrees(longitude_rad)

        # Use first point as reference (origin)
        lat_ref = latitude_rad[0]
        lon_ref = longitude_rad[0]

        # Earth radius in meters
        R = 6378137.0  # WGS84 equatorial radius

        # Convert lat/lon to local coordinates in meters
        # Using simple equirectangular projection (good for small areas)
        x_meters = (longitude_rad - lon_ref) * R * cos(lat_ref)
        y_meters = (latitude_rad - lat_ref) * R

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 2D trajectory plot (X-Y)
        ax1.plot(x_meters, y_meters, 'b-', linewidth=2, label='Trajectory')
        ax1.plot(x_meters[0], y_meters[0], 'go', markersize=10, label='Start')
        ax1.plot(x_meters[-1], y_meters[-1], 'ro', markersize=10, label='End')
        ax1.set_xlabel('East (m)')
        ax1.set_ylabel('North (m)')
        ax1.set_title(f'Trajectory {index} - Horizontal Path')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.axis('equal')

        # 2. Altitude vs time
        ax2.plot(time, altitude, 'r-', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Altitude (m)')
        ax2.set_title(f'Trajectory {index} - Altitude Profile')
        ax2.grid(True, alpha=0.3)

        # 3. 3D trajectory plot
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        ax3.plot(x_meters, y_meters, altitude, 'b-', linewidth=2)
        ax3.scatter(x_meters[0], y_meters[0], altitude[0], color='green', s=100, label='Start')
        ax3.scatter(x_meters[-1], y_meters[-1], altitude[-1], color='red', s=100, label='End')
        ax3.set_xlabel('East (m)')
        ax3.set_ylabel('North (m)')
        ax3.set_zlabel('Altitude (m)')
        ax3.set_title(f'Trajectory {index} - 3D Path')
        ax3.legend()

        # 4. Distance traveled vs time
        # Calculate cumulative distance
        distances = np.sqrt(np.diff(x_meters) ** 2 + np.diff(y_meters) ** 2 + np.diff(altitude) ** 2)
        cumulative_distance = np.concatenate(([0], np.cumsum(distances)))

        ax4.plot(time, cumulative_distance, 'g-', linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Distance Traveled (m)')
        ax4.set_title(f'Trajectory {index} - Cumulative Distance')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle(f'GT Trajectory {index} Analysis', fontsize=16, y=1.02)

        # Print trajectory statistics
        total_distance = cumulative_distance[-1]
        duration = time[-1] - time[0]
        max_altitude = np.max(altitude)
        min_altitude = np.min(altitude)

        print(f"\nTrajectory {index} Statistics:")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Total distance: {total_distance:.2f} meters")
        print(f"Average speed: {total_distance / duration:.2f} m/s")
        print(f"Altitude range: {min_altitude:.2f} to {max_altitude:.2f} meters")
        print(f"Number of data points: {len(time)}")

        plt.show()

        return x_meters, y_meters, altitude, time

    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None, None, None, None
    except Exception as e:
        print(f"Error reading file {filepath}: {str(e)}")
        return None, None, None, None


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv1d:
    """3x3 convolution with padding"""
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(6, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def resnet18_1d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34_1d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50_1d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101_1d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnet152_1d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)


def resnext50_32x4d_1d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet("resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnext101_32x8d_1d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet("resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def wide_resnet50_2_1d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def wide_resnet101_2_1d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)






class Resnet1chDnet(nn.Module):
    def __init__(self, in_channels=6, output_features=3):
        self.in_channels = in_channels
        super(Resnet1chDnet, self).__init__()

        self.model = resnet18_1d()

        # "ResNet",
        # "resnet18_1d",
        # "resnet34_1d",
        # "resnet50_1d",
        # "resnet101_1d",
        # "resnet152_1d",
        # "resnext50_32x4d_1d",
        # "resnext101_32x8d_1d",
        # "wide_resnet50_2_1d",
        # "wide_resnet101_2_1d",

        # Changed Conv2d to Conv1d since we're working with 1D data
        self.model.conv1 = nn.Conv1d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Update the final fully connected layer to match the expected dimensions
        num_features = self.model.fc.in_features  # Get the number of input features
        self.model.fc = nn.Linear(num_features, output_features)

    def forward(self, x):
        # Permute the dimensions from [batch_size, sequence_length, channels] to [batch_size, channels, sequence_length]
        x = x.permute(0, 2, 1)
        return self.model(x)


#
# # Define the CNN model
# class IMUDVLCNN(nn.Module):
#     def __init__(self, dropout_rate=0.2):  # Reduced dropout rate
#         super(IMUDVLCNN, self).__init__()
#         # Increase network capacity slightly
#         self.bn1 = nn.BatchNorm1d(128)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.bn3 = nn.BatchNorm1d(512)
#
#         self.conv1 = nn.Conv1d(6, 128, kernel_size=5, padding=1)
#         self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=1)
#         self.conv3 = nn.Conv1d(256, 512, kernel_size=5, padding=1)
#
#         self.pool = nn.AdaptiveAvgPool1d(1)
#         self.dropout = nn.Dropout(dropout_rate)
#
#         self.fc1 = nn.Linear(512, 1024)
#         self.fc2 = nn.Linear(1024, 3)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = x.permute(0, 2, 1)
#
#         # Residual connection for first block
#         identity = self.conv1(x)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = x + identity  # Residual connection
#         x = self.dropout(x)
#
#         # Second block
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#
#         # Third block
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#
#         return x
#
#
# class ResBlock1D(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3):
#         super(ResBlock1D, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
#         self.bn1 = nn.BatchNorm1d(out_channels)
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
#         self.bn2 = nn.BatchNorm1d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#
#         # Projection shortcut if dimensions change
#         self.shortcut = nn.Sequential()
#         if in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv1d(in_channels, out_channels, 1),
#                 nn.BatchNorm1d(out_channels)
#             )
#
#     def forward(self, x):
#         identity = self.shortcut(x)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#
# class SimplerIMUResNet(nn.Module):
#     def __init__(self, dropout_rate=0.2):
#         super(SimplerIMUResNet, self).__init__()
#
#         # Initial convolution
#         self.conv1 = nn.Conv1d(6, 64, kernel_size=7, padding=3)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
#
#         # ResNet blocks with increasing channels
#         self.res1_1 = ResBlock1D(64, 128)
#         # self.res1_2 = ResBlock1D(64, 64)
#
#         self.res2_1 = ResBlock1D(128, 256)
#         # self.res2_2 = ResBlock1D(128, 128)
#
#         self.res3_1 = ResBlock1D(256, 512)
#         # self.res3_2 = ResBlock1D(256, 256)
#
#         self.res4_1 = ResBlock1D(512, 1024)
#         # self.res4_2 = ResBlock1D(512, 512)
#
#         # Global pooling and final layers
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.fc = nn.Linear(1024, 3)
#
#     def forward(self, x):
#         # Input shape: (batch, time, features)
#         x = x.permute(0, 2, 1)  # to (batch, features, time)
#
#         # Initial convolution block
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         # ResNet blocks
#         # Stage 1
#         x = self.res1_1(x)
#         # x = self.res1_2(x)
#
#         # Stage 2
#         x = self.res2_1(x)
#         # x = self.res2_2(x)
#
#         # Stage 3
#         x = self.res3_1(x)
#         # x = self.res3_2(x)
#
#         # Stage 4
#         x = self.res4_1(x)
#         # x = self.res4_2(x)
#
#         # Global pooling and prediction
#         x = self.avgpool(x)
#         x = torch.flatten(x, start_dim=1)
#         # x = self.dropout(x)
#         x = self.fc(x)
#
#         return x


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


class EulerAnglesLoss(nn.Module):
    def __init__(self):
        """
        Custom loss function for Euler angles that handles periodicity and works with angles in degrees
        """
        super(EulerAnglesLoss, self).__init__()

    def normalize_angle(self, angle):
        """
        Normalize angle to [-180, 180] range
        Args:
            angle (torch.Tensor): Input angle in degrees
        Returns:
            torch.Tensor: Normalized angle in [-180, 180] range
        """
        return ((angle + 180) % 360) - 180

    def forward(self, pred, target):
        """
        Calculate the loss between predicted and target Euler angles
        Args:
            pred (torch.Tensor): Predicted Euler angles [batch_size, 3] in degrees
            target (torch.Tensor): Target Euler angles [batch_size, 3] in degrees
        Returns:
            torch.Tensor: Scalar loss value
        """
        # Normalize both predicted and target angles to [-180, 180] range
        pred_normalized = torch.stack([self.normalize_angle(a) for a in pred.t()]).t()
        target_normalized = torch.stack([self.normalize_angle(a) for a in target.t()]).t()

        # Calculate the angular difference considering periodicity
        diff = pred_normalized - target_normalized
        diff_normalized = torch.stack([self.normalize_angle(d) for d in diff.t()]).t()

        # Calculate MSE loss on the normalized differences
        loss = torch.mean(diff_normalized ** 2)

        return loss




class QuaternionNormLoss(nn.Module):
    def __init__(self, norm_weight=0.1):
        super(QuaternionNormLoss, self).__init__()
        self.norm_weight = norm_weight

    def forward(self, pred, target):
        # Orientation loss
        inner_product = torch.sum(pred * target, dim=-1)
        orientation_loss = 1 - torch.abs(inner_product)

        # Unit norm constraint
        norm_loss = torch.abs(torch.sum(pred * pred, dim=-1) - 1.0)

        return torch.mean(orientation_loss + self.norm_weight * norm_loss)


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

def save_baseline_results_numpy(mean_rmse_svd_degrees_per_num_samples_list, svd_time_list,rmse_test_list, current_time_test_list, config):
    """
    Save results as NumPy files.

    Args:
        mean_rmse_svd_degrees_per_num_samples_list: List of RMSE values
        svd_time_list: List of time values
        config: Includes directory to save the results
    """
    data_path = config['data_path']
    saved_rmse_results_file_name = config['saved_rmse_results_file_name']
    # rmse_saved_rmse_results_file_name = f"rmse_{saved_rmse_results_file_name}.npy"



    np.save(os.path.join(data_path, 'rmse_results_dir', f"rmse_baseline_{saved_rmse_results_file_name}.npy"), np.array(mean_rmse_svd_degrees_per_num_samples_list))
    np.save(os.path.join(data_path, 'rmse_results_dir', f"timeline_baseline_{saved_rmse_results_file_name}.npy"), np.array(svd_time_list))
    np.save(os.path.join(data_path, 'rmse_results_dir', f"rmse_{saved_rmse_results_file_name}.npy"), np.array(rmse_test_list))
    np.save(os.path.join(data_path, 'rmse_results_dir', f"timeline_{saved_rmse_results_file_name}.npy"), np.array(current_time_test_list))




    print(f"Results saved to {saved_rmse_results_file_name}")


def load_baseline_results_numpy(config):
    """
    Load results from NumPy files.

    Args:
        config: Includes directory containing the results

    Returns:
        Loaded arrays
    """
    data_path = config['data_path']
    loaded_file_names_list = config['loaded_rmse_results_file_names_list']
    loaded_rmse_results_file_names_list = []
    loaded_timeline_results_file_names_list = []


    for file_name in loaded_file_names_list:
        loaded_rmse_results_file_names_list.append(np.load(os.path.join(data_path,'rmse_results_dir',f"rmse_{file_name}.npy")).tolist())
        loaded_timeline_results_file_names_list.append(np.load(os.path.join(data_path,'rmse_results_dir',f"timeline_{file_name}.npy")).tolist())


    # mean_rmse_svd = np.load(f"rmse_results_dir/mean_rmse_svd.npy").tolist()
    # svd_time = np.load(f"rmse_results_dir/svd_time.npy").tolist()

    return loaded_rmse_results_file_names_list, loaded_timeline_results_file_names_list


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

def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_targets = []
    mse_angle_list = []
    angles_error_list = []

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

    return rmse, mean_angles_error

# def evaluate_model(model, test_loader, device):
#
#     model.eval()
#     total_loss = 0.0
#     all_predictions = []
#     all_targets = []
#     mse_angle_list = []
#
#     # Lists to store per-dataset metrics
#     dataset_rmse_lists = []
#     current_dataset_predictions = []
#     current_dataset_targets = []
#     samples_per_dataset = len(next(iter(test_loader))[0])  # Get batch size
#     batch_count = 0
#
#     with torch.no_grad():
#         for inputs, targets in test_loader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#
#             # Store predictions and targets
#             current_dataset_predictions.extend(outputs.cpu().numpy())
#             current_dataset_targets.extend(targets.cpu().numpy())
#             batch_count += 1
#
#             # Calculate per-batch metrics
#             for ii in range(len(targets)):
#                 mse_angle = calc_squared_err_angles(targets[ii].cpu().numpy(), outputs[ii].cpu().numpy())
#
#                 mse_angle_list.append(mse_angle)
#
#                 #         rmse_angles_svd = np.sqrt(squared_error_svd_baseline)
#                 #
#                 #         rmse_svd_degrees_per_num_samples_list.append(rmse_angles_svd)
#                 #
#                 #         rmse_svd_across_all_samples = np.sqrt(np.mean(rmse_svd_degrees_per_num_samples_list))
#
#
#
#
#             # # Check if we've completed a dataset
#             # if batch_count * samples_per_dataset >= len(test_loader.dataset):
#             #     # Calculate dataset-level metrics
#             #     dataset_predictions = np.array(current_dataset_predictions)
#             #     dataset_targets = np.array(current_dataset_targets)
#             #
#             #     # Calculate RMSE components for this dataset
#             #     rmse_components = np.sqrt(np.mean((dataset_predictions - dataset_targets) ** 2, axis=0))
#             #     dataset_rmse_lists.append(rmse_components)
#             #
#             #     # Reset for next dataset
#             #     current_dataset_predictions = []
#             #     current_dataset_targets = []
#             #     batch_count = 0
#
#             # # Calculate loss
#             # criterion = nn.CosineSimilarity()
#             # loss = torch.mean(torch.abs(criterion(targets, outputs)))
#             # loss = 1 - loss
#             # total_loss += loss.item()
#
#             all_predictions.append(outputs.cpu().numpy())
#             all_targets.append(targets.cpu().numpy())
#
#     # Calculate overall metrics
#     # avg_loss = total_loss / len(test_loader)
#     all_predictions = np.concatenate(all_predictions)
#     all_targets = np.concatenate(all_targets)
#
#     # Calculate overall RMSE components
#     rmse_components = np.sqrt(np.mean((all_predictions - all_targets) ** 2, axis=0))
#
#     # Calculate total RMSE
#     total_rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
#
#     # Calculate mean RMSE across all samples
#     rmse = np.sqrt(np.mean(mse_angle_list))
#
#     print(f'rmse is {rmse}')
#
#     # # Calculate mean RMSE components across datasets
#     # if dataset_rmse_lists:
#     #     mean_dataset_rmse = np.mean(dataset_rmse_lists, axis=0)
#     # else:
#     #     mean_dataset_rmse = rmse_components
#
#     return rmse


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


def save_model(model, test_type, window_size, base_path="models"):
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
    filepath = os.path.join(base_path, f'imu_dvl_model_{test_type}_window_{window_size}.pth')

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
        result[:, :, i] = vector_to_skew(vectors[:, i])

    return result


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
    roll = np.arctan2(r23, r33)

    # Computing pitch (around Y axis)
    pitch = np.arcsin(-r13)

    # Computing yaw (around Z axis)
    yaw = np.arctan2(r12, r11)

    return np.array([roll, pitch, yaw])


# # Acceleration-based Method: Run gradient descent solution
# euler_angles_gd_degrees = run_acc_gradient_descent(v_imu_sampled, v_dvl_sampled, omega_skew_imu_sampled,
#                                                    a_imu_sampled)
# euler_angles_gd_degrees_list.append(euler_angles_gd_degrees)

# squared_error_gd_baseline = squared_angular_difference(np.array(euler_angles_gd_degrees),euler_body_dvl_gt)

# squared_error_gd_baseline_list.append(squared_error_gd_baseline)


def calc_mean_rmse_svd_degrees_per_num_samples(v_imu_dvl_test_series_list, sample_freq, config):

    dataset_len = len(v_imu_dvl_test_series_list[0][0])
    window_sizes_sec = config['window_sizes_sec']

    # fix graph parameters for transformed_real_data
    # start_num_of_samples = sample_freq*10
    # end_num_of_samples = sample_freq*200
    # #end_num_of_samples = dataset_len
    # num_of_samples_slot = sample_freq*5


    start_num_of_samples = sample_freq*1
    # end_num_of_samples = sample_freq*160
    # end_num_of_samples = 200
    end_num_of_samples = dataset_len - dataset_len % 10
    num_of_samples_slot = sample_freq*1
    rmse_svd_all_tests_by_num_of_samples_dict = {}
    angles_err_svd_all_tests_by_num_of_samples_dict = {}
    # euler_rads_svd_all_tests_by_num_of_samples_dict = {}
    rmse_acc_gd_all_tests_by_num_of_samples_dict = {}

    for test_idx, test_sequence in enumerate(v_imu_dvl_test_series_list):
        # print(f'Calc baseline for test sequence {test_idx} ')
        v_imu_seq = test_sequence[0]
        v_dvl_seq = test_sequence[1]
        eul_body_dvl_gt_seq = test_sequence[2]
        omega_ned_to_body_rad = test_sequence[3]
        est_acc_eb_b = test_sequence[4]

        for num_samples in range(start_num_of_samples, dataset_len, num_of_samples_slot):
            v_imu_sampled = v_imu_seq[0:num_samples, :]
            v_dvl_sampled = v_dvl_seq[0:num_samples, :]
            euler_body_dvl_gt = eul_body_dvl_gt_seq[0, :]
            #omega_ned_to_body_rad_sampled = omega_ned_to_body_rad[0:num_samples, :]
            #omega_skew_ned_to_body_rad_sampled = skew_symetric(omega_ned_to_body_rad_sampled.T)
            #est_acc_eb_b_sampled = est_acc_eb_b[0:num_samples, :]

            euler_angles_svd_rads = run_svd_solution_for_wahba_problem(v_imu_sampled.T, v_dvl_sampled.T)
            euler_angles_svd_degrees = np.degrees(euler_angles_svd_rads)
            svd_squared_err_angles = calc_squared_err_angles(np.array(euler_angles_svd_degrees), euler_body_dvl_gt)
            svd_err_angles = calc_err_angles(np.array(euler_angles_svd_degrees), euler_body_dvl_gt)


            # euler_angles_acc_gd_rads = run_acc_gradient_descent(v_imu_sampled.T, v_dvl_sampled.T, omega_skew_ned_to_body_rad_sampled, est_acc_eb_b_sampled.T)
            # euler_angles_acc_gd_degrees = np.degrees(euler_angles_acc_gd_rads)
            # acc_gd_squared_err_angles = calc_squared_err_angles(np.array(euler_angles_acc_gd_degrees), euler_body_dvl_gt)

            # Store in dictionary
            if num_samples not in rmse_svd_all_tests_by_num_of_samples_dict:
                rmse_svd_all_tests_by_num_of_samples_dict[num_samples] = []
                angles_err_svd_all_tests_by_num_of_samples_dict[num_samples] = []
                # euler_rads_svd_all_tests_by_num_of_samples_dict[num_samples] = []

            # # Store in dictionary
            # if num_samples not in rmse_acc_gd_all_tests_by_num_of_samples_dict:
            #     rmse_acc_gd_all_tests_by_num_of_samples_dict[num_samples] = []



            # rmse_svd_all_tests_by_num_of_samples_dict[num_samples].append(svd_squared_err_angles)
            rmse_svd_all_tests_by_num_of_samples_dict[num_samples].append(np.sum(svd_squared_err_angles) / 3)
            angles_err_svd_all_tests_by_num_of_samples_dict[num_samples].append(svd_err_angles)
            # euler_rads_svd_all_tests_by_num_of_samples_dict[num_samples].append(euler_angles_svd_rads)
            # rmse_acc_gd_all_tests_by_num_of_samples_dict[num_samples].append(acc_gd_squared_err_angles)

    mean_rmse_svd_degrees_per_num_samples_list = []
    mean_angles_error_svd_degrees_per_num_samples_list = []
    # mean_rmse_acc_gd_degrees_per_num_samples_list = []

    for num_samples in range(start_num_of_samples, end_num_of_samples, num_of_samples_slot):
        mean_angles_error_svd_per_num_samples = np.mean(angles_err_svd_all_tests_by_num_of_samples_dict[num_samples], axis=0)
        mean_angles_error_svd_degrees_per_num_samples_list.append(mean_angles_error_svd_per_num_samples)

        mean_squared_error_angles_svd_per_num_samples = np.mean(rmse_svd_all_tests_by_num_of_samples_dict[num_samples])
        mean_rmse_svd_degrees_per_num_samples_list.append(np.sqrt(mean_squared_error_angles_svd_per_num_samples))

        # mean_euler_angles_acc_gd_per_num_samples = np.mean(rmse_acc_gd_all_tests_by_num_of_samples_dict[num_samples])
        # mean_rmse_acc_gd_degrees_per_num_samples_list.append(np.sqrt(mean_euler_angles_acc_gd_per_num_samples))

    #svd_time_list = list(range(0, dataset_len-start_num_of_samples))
    svd_time_list = list(range(start_num_of_samples//sample_freq, (end_num_of_samples)//sample_freq, num_of_samples_slot//sample_freq))

    for window in window_sizes_sec:
        print(f'mean rmse svd window {window} is:{mean_rmse_svd_degrees_per_num_samples_list[window // (num_of_samples_slot//sample_freq)]}')
        print(f'mean angles error svd window {window} is:{mean_angles_error_svd_degrees_per_num_samples_list[window // (num_of_samples_slot//sample_freq)]}')

    return mean_rmse_svd_degrees_per_num_samples_list, mean_angles_error_svd_degrees_per_num_samples_list, svd_time_list


def plot_results_graph_rmse_net_and_rmse_svd(svd_time_list, mean_rmse_svd_degrees_per_num_samples_list,
                                             current_time_test_list, rmse_test_list):
    # Import required modules at the top
    from matplotlib.patches import Rectangle

    # Create a single figure for total RMSE plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Increase font size globally
    plt.rcParams.update({'font.size': 14})

    # Set colors
    baseline_color = 'blue'
    test_color = 'red'

    # Plot total RMSE for baseline and test
    ax.plot(svd_time_list, mean_rmse_svd_degrees_per_num_samples_list, color=baseline_color, linestyle='-',
            label='Baseline', linewidth=2)
    ax.scatter(current_time_test_list, rmse_test_list, color=test_color, marker='o', label='AlignNet', s=100)

    # Create tick marks that align with both datasets
    # First, ensure we have the key points for small intervals
    small_intervals = np.array([5, 25, 50, 75])

    # Then create regularly spaced large intervals
    max_time = max(max(svd_time_list), max(current_time_test_list))
    large_intervals = np.arange(0, max_time, 50)  # 0 to max_time in steps of 50

    # Combine and ensure all ticks are unique and sorted
    all_ticks = np.unique(np.concatenate([small_intervals, large_intervals]))

    # Set x-axis ticks with larger font
    ax.set_xticks(all_ticks)
    ax.tick_params(axis='x', labelsize=22)

    # Increase y-axis tick font size
    ax.tick_params(axis='y', labelsize=22)

    # Add labels and title with larger font
    ax.set_xlabel('Time [sec]', fontsize=30)
    ax.set_ylabel('Alignment RMSE [deg]', fontsize=30)

    # Add primary grid - make sure grid aligns with actual ticks
    ax.grid(True, which='major', linestyle='-', alpha=0.5)

    # Add special grid lines for important intervals
    for x in small_intervals:
        ax.axvline(x=x, color='gray', linestyle=':', alpha=0.5)

    # Add legend with larger font
    ax.legend(fontsize=24, loc='upper right')

    # # Create inset zoom plot focusing on the scatter points (AlignNet values)
    # if current_time_test_list and rmse_test_list:
    #     # Calculate zoom region based on scatter points
    #     x_min, x_max = min(current_time_test_list), max(current_time_test_list)
    #     y_min, y_max = min(rmse_test_list), max(rmse_test_list)
    #
    #     # Add some padding to the zoom region (increase padding for better visibility)
    #     x_padding = max(5, (x_max - x_min) * 0.2)  # At least 5 units padding
    #     y_padding = max(0.5, (y_max - y_min) * 0.3)  # At least 0.5 units padding
    #
    #     zoom_x_min = max(0, x_min - x_padding)
    #     zoom_x_max = x_max + x_padding
    #     zoom_y_min = max(0, y_min - y_padding)
    #     zoom_y_max = y_max + y_padding
    #
    #     # Create inset axes - using a simpler approach
    #     # Position: [left, bottom, width, height] in axes coordinates
    #     axins = fig.add_axes([0.7, 0.30, 0.30, 0.25])  # Upper left corner at 70% height
    #
    #     # Plot the same data in the inset, but focus on the zoom region
    #     # Filter baseline data within zoom region
    #     zoom_svd_times = []
    #     zoom_svd_rmse = []
    #     for i, t in enumerate(svd_time_list):
    #         if zoom_x_min <= t <= zoom_x_max:
    #             zoom_svd_times.append(t)
    #             zoom_svd_rmse.append(mean_rmse_svd_degrees_per_num_samples_list[i])
    #
    #     if zoom_svd_times:
    #         axins.plot(zoom_svd_times, zoom_svd_rmse, color=baseline_color, linestyle='-',
    #                    linewidth=2, label='Baseline')
    #
    #     # Plot scatter points in inset
    #     axins.scatter(current_time_test_list, rmse_test_list, color=test_color, marker='o',
    #                   s=120, label='AlignNet', zorder=5)
    #
    #     # Set the zoom limits
    #     axins.set_xlim(zoom_x_min, zoom_x_max)
    #     axins.set_ylim(zoom_y_min, zoom_y_max)
    #
    #     # Add grid to inset
    #     axins.grid(True, alpha=0.3, linewidth=0.5)
    #
    #     # Customize inset appearance
    #     axins.tick_params(labelsize=12)
    #     axins.set_xlabel('Time [sec]', fontsize=12)
    #     axins.set_ylabel('RMSE [deg]', fontsize=12)
    #
    #     # Add a title to the inset
    #     axins.set_title('Zoom: AlignNet Region', fontsize=12, pad=10)
    #
    #     # Add border around inset
    #     for spine in axins.spines.values():
    #         spine.set_edgecolor('black')
    #         spine.set_linewidth(1.5)

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show(block=False)


def plot_results_graph_angle_me_net_and_angle_me_svd(svd_time_list,
                                                     mean_mean_error_angles_svd_degrees_per_num_samples_list,
                                                     current_time_test_list, mean_error_angles_test_list):
    # Create a single figure for individual angle errors
    fig, ax = plt.subplots(figsize=(12, 8))

    # Increase font size globally
    plt.rcParams.update({'font.size': 14})

    # Convert lists to numpy arrays for easier indexing
    svd_errors = np.array(mean_mean_error_angles_svd_degrees_per_num_samples_list)
    test_errors = np.array(mean_error_angles_test_list)

    # Define colors and labels for each angle
    angle_names = ['Roll', 'Pitch', 'Yaw']
    colors = ['red', 'green', 'blue']

    # Plot each angle error for SVD baseline
    for i in range(3):
        ax.plot(svd_time_list, svd_errors[:, i], color=colors[i], linestyle='-',
                label=f'SVD {angle_names[i]}', linewidth=2, alpha=0.7)

    # Plot each angle error for AlignNet
    for i in range(3):
        ax.scatter(current_time_test_list, test_errors[:, i], color=colors[i],
                   marker='o', label=f'AlignNet {angle_names[i]}', s=100,
                   edgecolors='black', linewidth=1)

    # Create tick marks that align with both datasets
    small_intervals = np.array([5, 25, 50, 75])
    max_time = max(max(svd_time_list), max(current_time_test_list))
    large_intervals = np.arange(0, max_time, 50)
    all_ticks = np.unique(np.concatenate([small_intervals, large_intervals]))

    # Remove 0 from the ticks while keeping it for grid lines
    display_ticks = all_ticks[all_ticks != 0]

    # Set x-axis ticks with larger font (excluding 0)
    ax.set_xticks(display_ticks)
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)

    # Add labels and title with larger font
    ax.set_xlabel('Time [sec]', fontsize=30)
    ax.set_ylabel('Mean Alignment Error [deg]', fontsize=30)

    # Add primary grid - this will still include the 0 line for the grid
    ax.grid(True, which='major', linestyle='-', alpha=0.5)

    # Add special grid lines for important intervals
    for x in small_intervals:
        ax.axvline(x=x, color='gray', linestyle=':', alpha=0.5)

    # Add legend with larger font - you might want to adjust location due to more entries
    ax.legend(fontsize=20, loc='upper right', ncol=2)

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show(block=False)

# load plots for real data
def plot_all_baseline_results(timeline_lists, rmse_lists, labels=None, current_time_test_list=None,
                              rmse_test_list=None):
    """
    Enhanced version with inset zoom capability.
    Plot multiple baseline results on the same plot with the same style as plot_results_graph_rmse_net_and_rmse_svd.

    Args:
        timeline_lists: List of lists containing timeline data for each baseline
        rmse_lists: List of lists containing RMSE data for each baseline
        labels: Optional list of labels for each baseline
        current_time_test_list: Optional list of window sizes for test data
        rmse_test_list: Optional list of RMSE values for test data
    """
    # Import required modules
    from matplotlib.patches import Rectangle

    # Define window sizes to print values for
    window_sizes_sec = [5, 25, 50, 75, 100]

    # Print RMSE values at specific window sizes for all datasets
    print("\n" + "=" * 60)
    print("RMSE VALUES AT SPECIFIC WINDOW SIZES")
    print("=" * 60)
    for i, (timeline, rmse, label) in enumerate(
            zip(timeline_lists, rmse_lists, labels or [f"Dataset {i + 1}" for i in range(len(timeline_lists))])):
        print(f"\n{label}:")
        for window_size in window_sizes_sec:
            # Find the closest timeline point to the window size
            closest_idx = None
            min_diff = float('inf')
            for j, time_point in enumerate(timeline):
                diff = abs(time_point - window_size)
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = j

            if closest_idx is not None:
                actual_time = timeline[closest_idx]
                rmse_value = rmse[closest_idx]
                print(f"  Window {window_size}s: RMSE = {rmse_value:.4f} (at time {actual_time}s)")
            else:
                print(f"  Window {window_size}s: No data point found")
    print("=" * 60 + "\n")

    # Create a single figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Increase font size globally
    plt.rcParams.update({'font.size': 14})

    # Colors for different baselines
    colors = ['blue', 'green', 'red']

    # Define the plotting order for 3 datasets: first, second, third (indices 0, 1, 2)
    plot_order = list(range(min(3, len(timeline_lists))))  # Handle up to 3 datasets

    # Plot each baseline - first as continuous line, others as scatter points
    for plot_idx in plot_order:
        if plot_idx < len(timeline_lists):
            timeline = timeline_lists[plot_idx]
            rmse = rmse_lists[plot_idx]
            color = colors[plot_idx % len(colors)]
            label = labels[plot_idx] if labels is not None and plot_idx < len(labels) else f'Baseline {plot_idx + 1}'

            if plot_idx == 0:  # First (index 0) as continuous line
                ax.plot(timeline, rmse, color=color, linestyle='-', label=label, linewidth=2)
            else:  # Others (indices 1, 2) as scatter points
                ax.scatter(timeline, rmse, color=color, marker='o', label=label, s=250)  # Reduced from 60 to 30

    # Plot test data if provided (scatter points like AlignNet in original function)
    if current_time_test_list is not None and rmse_test_list is not None:
        ax.scatter(current_time_test_list, rmse_test_list, color='red', marker='o', label='AlignNet',
                   s=250)  # Reduced from 60 to 30

    # Create tick marks for x-axis - same logic as original function
    max_time = max(max(t) for t in timeline_lists)
    if current_time_test_list:
        max_time = max(max_time, max(current_time_test_list))

    # Small intervals for important points
    small_intervals = np.array([5, 25, 50, 75])
    # Large intervals for regular spacing - exclude 0
    large_intervals = np.arange(50, max_time + 50, 50)  # Start from 50 instead of 0

    # Combine and ensure all ticks are unique and sorted
    all_ticks = np.unique(np.concatenate([small_intervals, large_intervals]))
    all_ticks = all_ticks[all_ticks <= max_time]

    # Set x-axis ticks with larger font
    ax.set_xticks(all_ticks)
    ax.tick_params(axis='x', labelsize=40)

    # Increase y-axis tick font size
    ax.tick_params(axis='y', labelsize=40)

    # Add labels and title with larger font
    ax.set_xlabel('Time [sec]', fontsize=40)
    ax.set_ylabel('Alignment RMSE [deg]', fontsize=40)

    # Add primary grid - make sure grid aligns with actual ticks
    ax.grid(True, which='major', linestyle='-', alpha=0.5)

    # Add special grid lines for important intervals
    for x in small_intervals:
        ax.axvline(x=x, color='gray', linestyle=':', alpha=0.5)

    # Create inset zoom plot - focus on second and third baselines at specific times
    zoom_times = [5, 25, 50, 75, 100]
    zoom_indices = [1, 2]  # Second and third baselines (0-indexed)

    if len(timeline_lists) >= 3:  # Ensure we have at least 3 baselines
        print(f"Creating zoom window for baselines {zoom_indices} at times {zoom_times}...")

        # Collect zoom data points
        all_zoom_x = []
        all_zoom_y = []

        for idx in zoom_indices:
            if idx < len(timeline_lists):
                timeline = timeline_lists[idx]
                rmse = rmse_lists[idx]

                # Find data points closest to zoom times
                for target_time in zoom_times:
                    closest_idx = None
                    min_diff = float('inf')
                    for j, time_point in enumerate(timeline):
                        diff = abs(time_point - target_time)
                        if diff < min_diff:
                            min_diff = diff
                            closest_idx = j

                    if closest_idx is not None:
                        all_zoom_x.append(timeline[closest_idx])
                        all_zoom_y.append(rmse[closest_idx])

        if all_zoom_x and all_zoom_y:
            # Calculate zoom region
            x_min, x_max = min(all_zoom_x), max(all_zoom_x)
            y_min, y_max = min(all_zoom_y), max(all_zoom_y)

            print(f"Zoom region - X: [{x_min}, {x_max}], Y: [{y_min}, {y_max}]")

            # Add padding to the zoom region
            x_padding = max(5, (x_max - x_min) * 0.2)
            y_padding = max(0.5, (y_max - y_min) * 0.3)

            zoom_x_min = max(0, x_min - x_padding)
            zoom_x_max = x_max + x_padding
            zoom_y_min = max(0, y_min - y_padding)
            zoom_y_max = y_max + y_padding

            # Create inset axes - positioned higher and moved further right
            axins = fig.add_axes([0.55, 0.65, 0.30, 0.25])  # Moved from x=0.35 to x=0.45 and y=0.55 to y=0.65

            # Plot the specified baselines in the inset within zoom region
            for i in zoom_indices:
                if i < len(timeline_lists):
                    timeline = timeline_lists[i]
                    rmse = rmse_lists[i]
                    color = colors[i % len(colors)]

                    # Filter data within zoom region
                    zoom_timeline = []
                    zoom_rmse = []
                    for j, t in enumerate(timeline):
                        if zoom_x_min <= t <= zoom_x_max:
                            zoom_timeline.append(t)
                            zoom_rmse.append(rmse[j])

                    if zoom_timeline:
                        # All zoom datasets as scatter points since they're indices 1 and 2
                        axins.scatter(zoom_timeline, zoom_rmse, color=color, marker='o', s=250)  # Reduced from 80 to 40

            # Also plot the first dataset (index 0) if it has data points within the zoom region
            if len(timeline_lists) >= 1:
                timeline = timeline_lists[0]
                rmse = rmse_lists[0]
                color = colors[0 % len(colors)]

                # Filter data within zoom region for the first dataset
                zoom_timeline_first = []
                zoom_rmse_first = []
                for j, t in enumerate(timeline):
                    if zoom_x_min <= t <= zoom_x_max:
                        zoom_timeline_first.append(t)
                        zoom_rmse_first.append(rmse[j])

                if zoom_timeline_first:
                    # Plot first dataset as continuous line (same style as main plot)
                    axins.plot(zoom_timeline_first, zoom_rmse_first, color=color, linestyle='-', linewidth=2)

            # Set the zoom limits
            axins.set_xlim(zoom_x_min, zoom_x_max)
            axins.set_ylim(zoom_y_min, zoom_y_max)

            # Set custom x-axis ticks to include 25, 50, 75, 100
            zoom_x_ticks = [25, 50, 75, 100]
            # Filter to only include ticks within the zoom region
            zoom_x_ticks = [tick for tick in zoom_x_ticks if zoom_x_min <= tick <= zoom_x_max]

            # Add any existing data points that might be important
            existing_x_values = list(set(all_zoom_x))
            for x_val in existing_x_values:
                if zoom_x_min <= x_val <= zoom_x_max and x_val not in zoom_x_ticks:
                    # Only add if it's reasonably spaced from existing ticks
                    min_distance = min([abs(x_val - tick) for tick in zoom_x_ticks] + [float('inf')])
                    if min_distance > 5:  # At least 5 units apart
                        zoom_x_ticks.append(x_val)

            zoom_x_ticks = sorted(zoom_x_ticks)
            axins.set_xticks(zoom_x_ticks)

            # Add grid to inset - this will now align with the custom x-axis ticks
            axins.grid(True, alpha=0.3, linewidth=0.5)

            # Add vertical grid lines specifically for the key values
            key_grid_values = [25, 50, 75, 100]
            for grid_val in key_grid_values:
                if zoom_x_min <= grid_val <= zoom_x_max:
                    axins.axvline(x=grid_val, color='gray', linestyle=':', alpha=0.7, linewidth=1)

            # Customize inset appearance with more precise y-axis
            axins.tick_params(labelsize=35)

            # Format y-axis to show more decimal precision
            import matplotlib.ticker as ticker
            axins.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

            # Increase number of y-axis ticks for better precision
            axins.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune=None))

            # Add border around inset
            for spine in axins.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.5)

    # Add legend with smaller font - positioned lower to avoid overlap with graph
    legend = ax.legend(fontsize=30, loc='lower right')
    # Manually adjust legend position to be lower
    legend.set_bbox_to_anchor((1.0, 0.0))

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show(block=False)



# load plots for simulated data
#
# def plot_all_baseline_results(timeline_lists, rmse_lists, labels=None, current_time_test_list=None,
#                               rmse_test_list=None):
#     """
#     Enhanced version with inset zoom capability.
#     Plot multiple baseline results on the same plot with the same style as plot_results_graph_rmse_net_and_rmse_svd.
#
#     Args:
#         timeline_lists: List of lists containing timeline data for each baseline
#         rmse_lists: List of lists containing RMSE data for each baseline
#         labels: Optional list of labels for each baseline
#         current_time_test_list: Optional list of window sizes for test data
#         rmse_test_list: Optional list of RMSE values for test data
#     """
#     # Import required modules
#     from matplotlib.patches import Rectangle
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import matplotlib.ticker as ticker
#
#     # Define window sizes to print values for
#     window_sizes_sec = [5, 25, 50, 75, 100]
#
#     # Print RMSE values at specific window sizes for all datasets
#     print("\n" + "=" * 60)
#     print("RMSE VALUES AT SPECIFIC WINDOW SIZES")
#     print("=" * 60)
#     for i, (timeline, rmse, label) in enumerate(
#             zip(timeline_lists, rmse_lists, labels or [f"Dataset {i + 1}" for i in range(len(timeline_lists))])):
#         print(f"\n{label}:")
#         for window_size in window_sizes_sec:
#             # Find the closest timeline point to the window size
#             closest_idx = None
#             min_diff = float('inf')
#             for j, time_point in enumerate(timeline):
#                 diff = abs(time_point - window_size)
#                 if diff < min_diff:
#                     min_diff = diff
#                     closest_idx = j
#
#             if closest_idx is not None:
#                 actual_time = timeline[closest_idx]
#                 rmse_value = rmse[closest_idx]
#                 print(f"  Window {window_size}s: RMSE = {rmse_value:.4f} (at time {actual_time}s)")
#             else:
#                 print(f"  Window {window_size}s: No data point found")
#     print("=" * 60 + "\n")
#
#     # Create a single figure
#     fig, ax = plt.subplots(figsize=(12, 8))
#
#     # Increase font size globally
#     plt.rcParams.update({'font.size': 14})
#
#     # Colors for different baselines
#     colors = ['blue', 'blue', 'green', 'green']
#
#     # Define the plotting order: first, third, second, fourth (indices 0, 2, 1, 3)
#     plot_order = [0, 2, 1, 3]
#
#     # Plot each baseline in the specified order - first and third as continuous lines, others as scatter points
#     for plot_idx in plot_order:
#         if plot_idx < len(timeline_lists):
#             timeline = timeline_lists[plot_idx]
#             rmse = rmse_lists[plot_idx]
#             color = colors[plot_idx % len(colors)]
#             label = labels[plot_idx] if labels is not None and plot_idx < len(labels) else f'Baseline {plot_idx + 1}'
#
#             if plot_idx == 0 or plot_idx == 2:  # First (index 0) and third (index 2) as continuous lines
#                 ax.plot(timeline, rmse, color=color, linestyle='-', label=label, linewidth=2)
#             else:  # Others as scatter points
#                 ax.scatter(timeline, rmse, color=color, marker='o', label=label, s=250)  # Reduced from 60 to 30
#
#     # Plot test data if provided (scatter points like AlignNet in original function)
#     if current_time_test_list is not None and rmse_test_list is not None:
#         ax.scatter(current_time_test_list, rmse_test_list, color='red', marker='o', label='AlignNet',
#                    s=250)  # Reduced from 60 to 30
#
#     # Create tick marks for x-axis - same logic as original function
#     max_time = max(max(t) for t in timeline_lists)
#     if current_time_test_list:
#         max_time = max(max_time, max(current_time_test_list))
#
#     # Small intervals for important points
#     small_intervals = np.array([5, 25, 50, 75])
#     # Large intervals for regular spacing - exclude 0
#     large_intervals = np.arange(50, max_time + 50, 50)  # Start from 50 instead of 0
#
#     # Combine and ensure all ticks are unique and sorted
#     all_ticks = np.unique(np.concatenate([small_intervals, large_intervals]))
#     all_ticks = all_ticks[all_ticks <= max_time]
#
#     # Set x-axis ticks with larger font
#     ax.set_xticks(all_ticks)
#     ax.tick_params(axis='x', labelsize=40)
#
#     # Increase y-axis tick font size
#     ax.tick_params(axis='y', labelsize=40)
#
#     # Add labels and title with larger font
#     ax.set_xlabel('Time [sec]', fontsize=40)
#     ax.set_ylabel('Alignment RMSE [deg]', fontsize=40)
#
#     # Add primary grid - make sure grid aligns with actual ticks
#     ax.grid(True, which='major', linestyle='-', alpha=0.5)
#
#     # Add special grid lines for important intervals
#     for x in small_intervals:
#         ax.axvline(x=x, color='gray', linestyle=':', alpha=0.5)
#
#     # Create inset zoom plot - focus on second and fourth baselines at specific times
#     zoom_times = [5, 25, 50, 75, 100]
#     zoom_focus_indices = [1, 3]  # Second and fourth baselines (0-indexed) - these determine zoom region
#     zoom_additional_indices = [0, 2]  # First and third baselines (0-indexed) - these are shown if they appear in zoom
#
#     if len(timeline_lists) >= 4:  # Ensure we have at least 4 baselines
#         print(
#             f"Creating zoom window focused on baselines {zoom_focus_indices} with additional baselines {zoom_additional_indices} if present...")
#
#         # Collect zoom data points ONLY from focus indices (second and fourth) to determine zoom region
#         all_zoom_x = []
#         all_zoom_y = []
#
#         for idx in zoom_focus_indices:
#             if idx < len(timeline_lists):
#                 timeline = timeline_lists[idx]
#                 rmse = rmse_lists[idx]
#
#                 # Find data points closest to zoom times
#                 for target_time in zoom_times:
#                     closest_idx = None
#                     min_diff = float('inf')
#                     for j, time_point in enumerate(timeline):
#                         diff = abs(time_point - target_time)
#                         if diff < min_diff:
#                             min_diff = diff
#                             closest_idx = j
#
#                     if closest_idx is not None:
#                         all_zoom_x.append(timeline[closest_idx])
#                         all_zoom_y.append(rmse[closest_idx])
#
#         if all_zoom_x and all_zoom_y:
#             # Calculate zoom region based ONLY on focus datasets (second and fourth)
#             x_min, x_max = min(all_zoom_x), max(all_zoom_x)
#             y_min, y_max = min(all_zoom_y), max(all_zoom_y)
#
#             print(f"Zoom region based on focus datasets - X: [{x_min}, {x_max}], Y: [{y_min}, {y_max}]")
#
#             # Add padding to the zoom region
#             x_padding = max(5, (x_max - x_min) * 0.2)
#             y_padding = max(0.5, (y_max - y_min) * 0.3)
#
#             zoom_x_min = max(0, x_min - x_padding)
#             zoom_x_max = x_max + x_padding
#
#             # Set Y-axis to start from 0 and include more lower values than higher values
#             zoom_y_min = 0  # Always start from 0
#             # Add modest padding above the maximum values
#             zoom_y_max = y_max + y_padding
#
#             # Create inset axes - keep original position and scale
#             axins = fig.add_axes([0.3, 0.65, 0.30, 0.25])  # Original position and scale
#
#             # First plot the focus baselines (second and fourth) in the inset
#             for i in zoom_focus_indices:
#                 if i < len(timeline_lists):
#                     timeline = timeline_lists[i]
#                     rmse = rmse_lists[i]
#                     color = colors[i % len(colors)]
#
#                     # Filter data within zoom region
#                     zoom_timeline = []
#                     zoom_rmse = []
#                     for j, t in enumerate(timeline):
#                         if zoom_x_min <= t <= zoom_x_max:
#                             zoom_timeline.append(t)
#                             zoom_rmse.append(rmse[j])
#
#                     if zoom_timeline:
#                         # Plot as scatter points (since indices 1 and 3 are scatter in main plot)
#                         axins.scatter(zoom_timeline, zoom_rmse, color=color, marker='o', s=250)
#
#             # Then check if additional baselines (first and third) have data in the zoom region and plot them
#             for i in zoom_additional_indices:
#                 if i < len(timeline_lists):
#                     timeline = timeline_lists[i]
#                     rmse = rmse_lists[i]
#                     color = colors[i % len(colors)]
#
#                     # Filter data within zoom region
#                     zoom_timeline = []
#                     zoom_rmse = []
#                     for j, t in enumerate(timeline):
#                         if zoom_x_min <= t <= zoom_x_max:
#                             zoom_timeline.append(t)
#                             zoom_rmse.append(rmse[j])
#
#                     if zoom_timeline:
#                         # Plot as continuous lines (since indices 0 and 2 are lines in main plot)
#                         axins.plot(zoom_timeline, zoom_rmse, color=color, linestyle='-', linewidth=2)
#                         print(f"Added baseline {i} to zoom window (found {len(zoom_timeline)} data points)")
#
#             # Set the zoom limits
#             axins.set_xlim(zoom_x_min, zoom_x_max)
#             axins.set_ylim(zoom_y_min, zoom_y_max)
#
#             # Set custom X-axis ticks for zoom window
#             custom_x_ticks = [5, 25, 50, 75, 100]
#             # Filter ticks that are within the zoom region
#             visible_x_ticks = [tick for tick in custom_x_ticks if zoom_x_min <= tick <= zoom_x_max]
#             axins.set_xticks(visible_x_ticks)
#
#             # Add grid to inset
#             axins.grid(True, alpha=0.3, linewidth=0.5)
#
#             # Add vertical grid lines specifically for the key values
#             key_grid_values = [25, 50, 75, 100]
#             for grid_val in key_grid_values:
#                 if zoom_x_min <= grid_val <= zoom_x_max:
#                     axins.axvline(x=grid_val, color='gray', linestyle=':', alpha=0.7, linewidth=1)
#
#             # Customize inset appearance with more precise y-axis
#             axins.tick_params(labelsize=35)
#
#             # Format y-axis to show more decimal precision
#             axins.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
#
#             # Increase number of y-axis ticks for better precision
#             axins.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune=None))
#
#             # Add border around inset
#             for spine in axins.spines.values():
#                 spine.set_edgecolor('black')
#                 spine.set_linewidth(1.5)
#
#     # Add legend with smaller font - positioned lower to avoid overlap with graph
#     legend = ax.legend(fontsize=21, loc='lower right')
#     # Manually adjust legend position to be lower
#     legend.set_bbox_to_anchor((1.0, 0.1))  # Move legend lower from (0.99, 0.15)
#
#     # Adjust layout
#     plt.tight_layout()
#
#     # Show plot
#     plt.show(block=False)

def split_data_properly(data_pd, num_sequences, sequence_length, train_size=0.6, val_size=0.2):
    # Create sequence indices
    sequence_indices = np.arange(num_sequences)

    # First split to separate test set
    train_val_indices, test_indices = train_test_split(
        sequence_indices,
        test_size=0.2,
        shuffle=True,
        random_state=42
    )

    # Then split remaining data into train and validation
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_size / (train_size + val_size),
        shuffle=True,
        random_state=42
    )

    # Extract data arrays
    v_imu_body_full = np.array(data_pd.iloc[:, 1:4].T)
    v_dvl_full = np.array(data_pd.iloc[:, 4:7].T)
    euler_body_dvl_full = np.array(data_pd.iloc[:, 7:10].T)
    est_eul_ned_to_body_rad_full = np.array(data_pd.iloc[:, 10:13].T)
    est_acc_eb_b_full = np.array(data_pd.iloc[:, 13:16].T)

    # Create sequence lists
    train_sequences = []
    val_sequences = []
    test_sequences = []

    # Fill training sequences
    for idx in train_indices:
        start_idx = idx * sequence_length
        end_idx = start_idx + sequence_length
        train_sequences.append([
            v_imu_body_full[:, start_idx:end_idx].T,
            v_dvl_full[:, start_idx:end_idx].T,
            euler_body_dvl_full[:, start_idx:end_idx].T,
            est_eul_ned_to_body_rad_full[:, start_idx:end_idx].T,
            est_acc_eb_b_full[:, start_idx:end_idx].T
        ])

    # Fill validation sequences
    for idx in val_indices:
        start_idx = idx * sequence_length
        end_idx = start_idx + sequence_length
        val_sequences.append([
            v_imu_body_full[:, start_idx:end_idx].T,
            v_dvl_full[:, start_idx:end_idx].T,
            euler_body_dvl_full[:, start_idx:end_idx].T,
            est_eul_ned_to_body_rad_full[:, start_idx:end_idx].T,
            est_acc_eb_b_full[:, start_idx:end_idx].T
        ])

    # Fill test sequences
    for idx in test_indices:
        start_idx = idx * sequence_length
        end_idx = start_idx + sequence_length
        test_sequences.append([
            v_imu_body_full[:, start_idx:end_idx].T,
            v_dvl_full[:, start_idx:end_idx].T,
            euler_body_dvl_full[:, start_idx:end_idx].T,
            est_eul_ned_to_body_rad_full[:, start_idx:end_idx].T,
            est_acc_eb_b_full[:, start_idx:end_idx].T
        ])

    return train_sequences, val_sequences, test_sequences, test_indices

def main(config):
    # Example usage

    # plot_trajectory_path(3)
    # Plot the 3D trajectories
    straight_data, turn_data = plot_simulation_trajectories()

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
    roll_gt_rads, pitch_gt_rads, yaw_gt_rads = convert_deg_to_rads(roll_gt_deg, pitch_gt_deg, yaw_gt_deg)

    # Convert Euler angles to rotation matrix
    rotation_matrix_ins_to_dvl = euler_angles_to_rotation_matrix(roll_gt_rads, pitch_gt_rads, yaw_gt_rads)

    # Read data from the .csv file


    ##prepare real data dataset - for sim2real model
    real_data_pd = pd.read_csv(os.path.join(data_path, f'{real_data_file_name}'), header=None)
    v_imu_body_real_data_full = np.array(real_data_pd.iloc[:, 1:4].T)
    v_dvl_real_data_full = np.array(real_data_pd.iloc[:, 4:7].T)
    euler_body_dvl_real_data_full = np.array(real_data_pd.iloc[:, 7:10].T)
    real_dataset_len = len(v_imu_body_real_data_full[1])

    current_time_test_list = []
    rmse_test_list = []
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
            model_path = os.path.join(trained_model_base_path, f'imu_dvl_model_simulated_data_long_turn_17_+0_3125_ba_100_bg_1_window_{window_size}.pth')
            # model_path = os.path.join(trained_model_base_path, f'imu_dvl_model_transformed_real_data_traj7_17_+0_3125_window_{window_size}.pth')
            # model_path = os.path.join(trained_model_base_path, f'imu_dvl_model_transformed_real_data_traj11_16_+0_3333_window_{window_size}.pth')
            # Load model
            # model = SimplerIMUResNet(dropout_rate=0.3)
            model = Resnet1chDnet()
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

            print(f"eval")
            # Evaluate model
            rmse, mean_angles_error = evaluate_model(
                model, test_loader, device
            )

            rmse_test_list.append(rmse)
            mean_error_angles_test_list.append(mean_angles_error)
            current_time = window_size
            current_time_test_list.append(current_time)
            print(f"Test RMSE: {rmse:.4f}")


    # Test Baseline Model section #################
    if config['test_baseline_model']:

        mean_rmse_svd_degrees_per_num_samples_list, mean_error_angles_svd_degrees_per_num_samples_list, svd_time_list = calc_mean_rmse_svd_degrees_per_num_samples(v_imu_dvl_test_series_list, sample_freq, config)

        # # movmean_window_size = 15 # Best for transformed real traj 8
        # movmean_window_size = 1  #
        # mean_rmse_svd_degrees_per_num_samples_list_smoothed = movmean(mean_rmse_svd_degrees_per_num_samples_list,
        #                                                               movmean_window_size)
        #
        # mean_rmse_svd_degrees_per_num_samples_list = mean_rmse_svd_degrees_per_num_samples_list_smoothed

        # save_baseline_results_numpy(mean_rmse_svd_degrees_per_num_samples_list, svd_time_list,rmse_test_list, current_time_test_list, config)

        # plot_results_graph_angle_me_net_and_angle_me_svd(svd_time_list, mean_error_angles_svd_degrees_per_num_samples_list, current_time_test_list, mean_error_angles_test_list)
        plot_results_graph_rmse_net_and_rmse_svd(svd_time_list, mean_rmse_svd_degrees_per_num_samples_list, current_time_test_list, rmse_test_list)

        # loaded_rmse_baseline_results_list, loaded_timeline_baseline_results_list = load_baseline_results_numpy(config)
        # print("Baseline RMSE")
        # # Create labels for the baselines (optional)
        # baseline_labels = [
        #     f"SVD (baseline)", f"ResAlignNet - sim2real (ours)", f"ResAlignNet (ours)"
        #     # f"SVD (baseline) - Navigation Grade IMU", f"ResAlignNet (ours) - Navigation Grade IMU", f"SVD (baseline) - Tactical Grade IMU", f"ResAlignNet (ours) - Tactical Grade IMU"
        # ]
        #
        # # Plot all baselines on the same plot
        # plot_all_baseline_results(
        #     loaded_timeline_baseline_results_list,
        #     loaded_rmse_baseline_results_list,
        #     labels=baseline_labels,
        #     current_time_test_list=current_time_test_list if config['test_model'] else None,
        #     rmse_test_list=rmse_test_list if config['test_model'] else None
        # )

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
        'window_sizes_sec': [5],
        # 'window_sizes_sec': [5,25,50,75,100],
        # 'window_sizes_sec': [175],
        'batch_size': 32,
        # 'simulated_dataset_len': 1554, # - straight descent1 - important!! you have to update it, from the data output file, every time you change dataset
        # 'simulated_dataset_len': 1077, # - straight line - important!! you have to update it, from the data output file, every time you change dataset
        # 'simulated_dataset_len': 1075, # - straight line1 - important!! you have to update it, from the data output file, every time you change dataset
        # 'simulated_dataset_len': 5791, # - straight line1 - important!! you have to update it, from the data output file, every time you change dataset
        # 'simulated_dataset_len': 1503, # - straight line1 - important!! you have to update it, from the data output file, every time you change dataset
        'simulated_dataset_len': 1027, # - long turn - important!! you have to update it, from the data output file, every time you change dataset
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
        'test_type': 'convex_data',  # Set to "convex_data" or "simulated_data" or "transformed_real_data" or "simulated_imu_from_real_gt_data" or "real_data"
        'train_model': False,
        'test_model': False,
        'test_baseline_model': False,
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
        # 'transformed_real_data_file_name': 'transformed_real_data_output_traj7_17_+0_3125.csv',
        # 'transformed_real_data_file_name': 'transformed_real_data_output_traj11_16_+0_3333.csv',
        'transformed_real_data_file_name': 'transformed_real_data_output_traj11_16_+0_3333_roll_4.csv',
        # 'transformed_real_data_file_name': 'transformed_real_data_output_traj11_26_+1.csv',
        'simulated_imu_from_real_gt_data_file_name': 'simulated_imu_from_real_gt_data_output.csv',
        # 'saved_rmse_results_file_name': 'simulated_data_long_turn_17_+0_3125_ba_real_bg_10',
        # 'saved_rmse_results_file_name': 'sim2real_transformed_real_data_traj7_17_+0_3125',
        # 'saved_rmse_results_file_name': 'transformed_real_data_traj7_17_+2_8125',
        'saved_rmse_results_file_name': 'sim2real_transformed_real_data_traj7_11_+2_8125',
        # 'loaded_rmse_results_file_names_list': ['baseline_transformed_real_data_traj12_22_+0_227','sim2real_transformed_real_data_traj12_22_+0_227', 'transformed_real_data_traj12_22_+0_227'],
        # 'loaded_rmse_results_file_names_list': ['baseline_transformed_real_data_traj7_17_+0_3125', 'sim2real_transformed_real_data_traj7_17_+0_3125', 'transformed_real_data_traj7_17_+0_3125'],
        'loaded_rmse_results_file_names_list': ['baseline_transformed_real_data_traj11_16_+0_3333', 'sim2real_transformed_real_data_traj11_16_+0_3333', 'transformed_real_data_traj11_16_+0_3333'],
        # 'loaded_rmse_results_file_names_list': ['baseline_simulated_data_long_turn_17_+0_3125_ba_100_bg_1','simulated_data_long_turn_17_+0_3125_ba_100_bg_1', 'baseline_simulated_data_long_turn_17_+0_3125_ba_real_bg_10','simulated_data_long_turn_17_+0_3125_ba_real_bg_10'],
        # 'loaded_rmse_results_file_names_list': ['baseline_simulated_data_long_turn_17_+0_3125_ba_100_bg_1','simulated_data_long_turn_17_+0_3125_ba_100_bg_1', 'baseline_simulated_data_long_turn_17_+0_3125_ba_real_bg_10','simulated_data_long_turn_17_+0_3125_ba_real_bg_10'],
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


