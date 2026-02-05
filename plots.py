import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from convert_lat_lon_to_meters import convert_lat_lon_to_meters

def plot_max_error_comparison():
    """
    Plot side-by-side bar chart comparing maximum errors across
    baseline, real, and sim2real methods for two trajectories.
    """
    # Data
    trajectories = ['Trajectory #1', 'Trajectory #2']

    # Maximum errors for each method
    baseline_errors = [179.8625, 179.8716]
    real_errors = [3.6604, 3.4161]
    sim2real_errors = [43.5888, 69.6055]

    # Set up the bar positions
    x = np.arange(len(trajectories))  # Label locations
    width = 0.25  # Width of bars

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create bars with updated colors and labels
    bars1 = ax.bar(x - width, baseline_errors, width, label='SVD (baseline)',
                   color='blue', alpha=0.8)
    bars2 = ax.bar(x, real_errors, width, label='ResAlignNet (ours)',
                   color='red', alpha=0.8)
    bars3 = ax.bar(x + width, sim2real_errors, width, label='ResAlignNet - Sim2Real (ours)',
                   color='green', alpha=0.8)

    # Add value labels on top of bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.2f}°',
                    ha='center', va='bottom', fontsize=20, fontweight='bold')

    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)

    # Customize the plot
    ax.set_xlabel('Trajectory', fontsize=35)
    ax.set_ylabel('Maximum Alignment Error [deg]', fontsize=35)
    # ax.set_title('Maximum Alignment Error Comparison', fontsize=32, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(trajectories, fontsize=30)
    ax.tick_params(axis='y', labelsize=30)

    # Move legend to the right side
    ax.legend(fontsize=20, loc='upper right')

    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()


def plot_max_error_comparison_log_scale():
    """
    Alternative version with logarithmic scale for better visualization
    of the large difference between baseline and other methods.
    """
    # Data
    trajectories = ['Trajectory #1', 'Trajectory #2']

    # Maximum errors for each method
    baseline_errors = [179.8625, 179.8716]
    real_errors = [3.6604, 3.4161]
    sim2real_errors = [43.5888, 69.6055]

    # Set up the bar positions
    x = np.arange(len(trajectories))  # Label locations
    width = 0.25  # Width of bars

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create bars with updated colors and labels
    bars1 = ax.bar(x - width, baseline_errors, width, label='SVD (baseline)',
                   color='blue', alpha=0.8)
    bars2 = ax.bar(x, real_errors, width, label='ResAlignNet (ours)',
                   color='green', alpha=0.8)
    bars3 = ax.bar(x + width, sim2real_errors, width, label='ResAlignNet - Sim2Real (ours)',
                   color='red', alpha=0.8)

    # Add value labels on top of bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.1f}°',
                    ha='center', va='bottom', fontsize=20, fontweight='bold')

    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)

    # Use logarithmic scale for y-axis
    ax.set_yscale('log')

    # Customize the plot
    ax.set_xlabel('Trajectory', fontsize=30)
    ax.set_ylabel('Maximum Alignment Error [deg] (log scale)', fontsize=30)
    ax.set_title('Maximum Alignment Error Comparison (Log Scale)', fontsize=32, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(trajectories, fontsize=28)
    ax.tick_params(axis='y', labelsize=26)

    # Move legend to the right side
    ax.legend(fontsize=24, loc='upper right')

    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y', which='both')
    ax.set_axisbelow(True)

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()



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
            x_meters = (longitude_rad - lon_ref) * R * np.cos(lat_ref)
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
        x_meters = (longitude_rad - lon_ref) * R * np.cos(lat_ref)
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


def print_baseline_results_at_windows(timeline_lists, rmse_lists, window_sizes_sec, labels=None):
    """
    Print RMSE values at specific window sizes for multiple baseline results.

    Args:
        timeline_lists: List of lists containing timeline data for each baseline
        rmse_lists: List of lists containing RMSE data for each baseline
        window_sizes_sec: List of window sizes (in seconds) to print values for
        labels: Optional list of labels for each baseline
    """

    eval_metric = config['eval_metric']

    if eval_metric == 'RMSE':
        eval_name = 'RMSE'
    elif eval_metric == 'AOE':
        eval_name = 'AOE'

    # Use default labels if none provided
    if labels is None:
        labels = [f"Dataset {i + 1}" for i in range(len(timeline_lists))]

    # Print header
    print("\n" + "=" * 60)
    print(f"{eval_name} VALUES AT SPECIFIC WINDOW SIZES")
    print("=" * 60)

    # Iterate through each dataset
    for i, (timeline, rmse, label) in enumerate(zip(timeline_lists, rmse_lists, labels)):
        print(f"\n{label}:")

        # Print RMSE at each window size
        for window_size in window_sizes_sec:
            # Find the closest timeline point to the window size
            closest_idx = None
            min_diff = float('inf')
            for j, time_point in enumerate(timeline):
                diff = abs(time_point - window_size)
                if diff < min_diff:
                    min_diff = diff
                    closest_idx = j

            # Print the result
            if closest_idx is not None:
                actual_time = timeline[closest_idx]
                rmse_value = rmse[closest_idx]
                print(f"  Window {window_size}s: {eval_name} = {rmse_value:.4f} (at time {actual_time}s)")
            else:
                print(f"  Window {window_size}s: No data point found")

    print("=" * 60 + "\n")


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


def plot_results_graph_aoe_net_and_aoe_svd(svd_time_list, mean_aoe_svd_degrees_per_num_samples_list,
                                           current_time_test_list, aoe_test_list):
    """
    Plot AOE (Angle of Error) results comparing baseline SVD method and AlignNet.

    This function is similar to plot_results_graph_rmse_net_and_rmse_svd but plots
    AOE results instead of RMSE results.

    Parameters:
        svd_time_list: Time values for SVD baseline (list)
        mean_aoe_svd_degrees_per_num_samples_list: Mean AOE values for SVD baseline in degrees (list)
        current_time_test_list: Time values for neural network test points (list)
        aoe_test_list: AOE values for neural network in degrees (list)
    """

    # Create a single figure for total AOE plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Increase font size globally
    plt.rcParams.update({'font.size': 14})

    # Set colors
    baseline_color = 'blue'
    test_color = 'red'

    # Plot total AOE for baseline and test
    ax.plot(svd_time_list, mean_aoe_svd_degrees_per_num_samples_list, color=baseline_color, linestyle='-',
            label='Baseline', linewidth=2)
    ax.scatter(current_time_test_list, aoe_test_list, color=test_color, marker='o', label='AlignNet', s=100)

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
    ax.set_ylabel('Alignment AOE [deg]', fontsize=30)

    # Add primary grid - make sure grid aligns with actual ticks
    ax.grid(True, which='major', linestyle='-', alpha=0.5)

    # Add special grid lines for important intervals
    for x in small_intervals:
        ax.axvline(x=x, color='gray', linestyle=':', alpha=0.5)

    # Add legend with larger font
    ax.legend(fontsize=24, loc='upper right')

    # Optional: Create inset zoom plot focusing on the scatter points (AlignNet values)
    # Uncomment the code below if you want to add a zoom inset
    """
    if current_time_test_list and aoe_test_list:
        # Calculate zoom region based on scatter points
        x_min, x_max = min(current_time_test_list), max(current_time_test_list)
        y_min, y_max = min(aoe_test_list), max(aoe_test_list)

        # Add some padding to the zoom region (increase padding for better visibility)
        x_padding = max(5, (x_max - x_min) * 0.2)  # At least 5 units padding
        y_padding = max(0.5, (y_max - y_min) * 0.3)  # At least 0.5 units padding

        zoom_x_min = max(0, x_min - x_padding)
        zoom_x_max = x_max + x_padding
        zoom_y_min = max(0, y_min - y_padding)
        zoom_y_max = y_max + y_padding

        # Create inset axes - using a simpler approach
        # Position: [left, bottom, width, height] in axes coordinates
        axins = fig.add_axes([0.7, 0.30, 0.30, 0.25])  # Upper left corner at 70% height

        # Plot the same data in the inset, but focus on the zoom region
        # Filter baseline data within zoom region
        zoom_svd_times = []
        zoom_svd_aoe = []
        for i, t in enumerate(svd_time_list):
            if zoom_x_min <= t <= zoom_x_max:
                zoom_svd_times.append(t)
                zoom_svd_aoe.append(mean_aoe_svd_degrees_per_num_samples_list[i])

        if zoom_svd_times:
            axins.plot(zoom_svd_times, zoom_svd_aoe, color=baseline_color, linestyle='-',
                       linewidth=2, label='Baseline')

        # Plot scatter points in inset
        axins.scatter(current_time_test_list, aoe_test_list, color=test_color, marker='o',
                      s=120, label='AlignNet', zorder=5)

        # Set the zoom limits
        axins.set_xlim(zoom_x_min, zoom_x_max)
        axins.set_ylim(zoom_y_min, zoom_y_max)

        # Add grid to inset
        axins.grid(True, alpha=0.3, linewidth=0.5)

        # Customize inset appearance
        axins.tick_params(labelsize=12)
        axins.set_xlabel('Time [sec]', fontsize=12)
        axins.set_ylabel('AOE [deg]', fontsize=12)

        # Draw a box around the zoom region on the main plot
        rect = Rectangle((zoom_x_min, zoom_y_min), 
                         zoom_x_max - zoom_x_min, 
                         zoom_y_max - zoom_y_min,
                         fill=False, edgecolor='black', linestyle='--', linewidth=1.5)
        ax.add_patch(rect)

        # Connect the inset to the zoom region with lines
        from matplotlib.patches import ConnectionPatch
        # You can add connection lines here if desired
    """

    plt.tight_layout()
    plt.show()


# Example usage (add this to your main code where you currently call plot_results_graph_rmse_net_and_rmse_svd):
"""
if config['test_baseline_model']:
    # Calculate AOE for SVD baseline
    mean_aoe_svd_degrees_per_num_samples_list, mean_angles_error_svd_degrees_per_num_samples_list, svd_time_list = calc_mean_aoe_svd_degrees_per_num_samples(v_imu_dvl_test_series_list, sample_freq, config)

    # Plot AOE results
    plot_results_graph_aoe_net_and_aoe_svd(svd_time_list, mean_aoe_svd_degrees_per_num_samples_list, current_time_test_list, aoe_test_list)
"""

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