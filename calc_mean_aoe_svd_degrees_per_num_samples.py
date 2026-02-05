import numpy as np

from calc_error_functions import calc_err_angles
from evaluate_model_aoe import calc_aoe_single_sample
from run_svd_solution_for_wahba_problem import run_svd_solution_for_wahba_problem


def calc_mean_aoe_svd_degrees_per_num_samples(v_imu_dvl_test_series_list, sample_freq, config):
    """
    Calculate mean AOE for SVD baseline across different window sizes.

    This is the AOE version of calc_mean_rmse_svd_degrees_per_num_samples.
    Instead of computing RMSE on Euler angle differences, it computes AOE
    using the geodesic distance on SO(3).

    Parameters:
        v_imu_dvl_test_series_list: List of test sequences
        sample_freq: Sampling frequency
        config: Configuration dictionary

    Returns:
        mean_aoe_svd_degrees_per_num_samples_list: Mean AOE values in degrees
        mean_angles_error_svd_degrees_per_num_samples_list: Mean Euler angle errors
        svd_time_list: Time values for plotting
    """

    dataset_len = len(v_imu_dvl_test_series_list[0][0])
    window_sizes_sec = config['window_sizes_sec']

    start_num_of_samples = sample_freq * 1
    end_num_of_samples = dataset_len - dataset_len % 10
    num_of_samples_slot = sample_freq * 1

    # Dictionary to store AOE values (squared errors in radians^2)
    aoe_squared_svd_all_tests_by_num_of_samples_dict = {}

    # Optional: Keep Euler angle errors for comparison/debugging
    angles_err_svd_all_tests_by_num_of_samples_dict = {}

    # Process each test sequence
    for test_idx, test_sequence in enumerate(v_imu_dvl_test_series_list):
        v_imu_seq = test_sequence[0]
        v_dvl_seq = test_sequence[1]
        eul_body_dvl_gt_seq = test_sequence[2]
        omega_ned_to_body_rad = test_sequence[3]
        est_acc_eb_b = test_sequence[4]

        # Iterate over different window sizes (number of samples)
        for num_samples in range(start_num_of_samples, dataset_len, num_of_samples_slot):
            v_imu_sampled = v_imu_seq[0:num_samples, :]
            v_dvl_sampled = v_dvl_seq[0:num_samples, :]
            euler_body_dvl_gt = eul_body_dvl_gt_seq[0, :]

            # Run SVD solution (same as before)
            euler_angles_svd_rads = run_svd_solution_for_wahba_problem(v_imu_sampled.T, v_dvl_sampled.T)
            euler_angles_svd_degrees = np.degrees(euler_angles_svd_rads)

            # ===== NEW: Calculate AOE instead of RMSE =====
            aoe_squared_error = calc_aoe_single_sample(euler_body_dvl_gt, euler_angles_svd_degrees)

            # Optional: Keep Euler angle errors for debugging
            svd_err_angles = calc_err_angles(np.array(euler_angles_svd_degrees), euler_body_dvl_gt)

            # Store in dictionary
            if num_samples not in aoe_squared_svd_all_tests_by_num_of_samples_dict:
                aoe_squared_svd_all_tests_by_num_of_samples_dict[num_samples] = []
                angles_err_svd_all_tests_by_num_of_samples_dict[num_samples] = []

            # Store squared error (in radians^2)
            aoe_squared_svd_all_tests_by_num_of_samples_dict[num_samples].append(aoe_squared_error)
            angles_err_svd_all_tests_by_num_of_samples_dict[num_samples].append(svd_err_angles)

    # Calculate mean AOE and mean angle errors
    mean_aoe_svd_degrees_per_num_samples_list = []
    mean_angles_error_svd_degrees_per_num_samples_list = []

    for num_samples in range(start_num_of_samples, end_num_of_samples, num_of_samples_slot):
        # Mean Euler angle errors (for debugging/comparison)
        mean_angles_error_svd_per_num_samples = np.mean(
            angles_err_svd_all_tests_by_num_of_samples_dict[num_samples], axis=0
        )
        mean_angles_error_svd_degrees_per_num_samples_list.append(mean_angles_error_svd_per_num_samples)

        # Mean AOE: sqrt(mean(squared_errors))
        mean_squared_error_aoe_svd_per_num_samples = np.mean(
            aoe_squared_svd_all_tests_by_num_of_samples_dict[num_samples]
        )
        # Convert from radians to degrees
        aoe_rad = np.sqrt(mean_squared_error_aoe_svd_per_num_samples)
        aoe_deg = np.degrees(aoe_rad)
        mean_aoe_svd_degrees_per_num_samples_list.append(aoe_deg)

    # Time list for plotting
    svd_time_list = list(range(
        start_num_of_samples // sample_freq,
        end_num_of_samples // sample_freq,
        num_of_samples_slot // sample_freq
    ))

    # Print results for specified window sizes
    for window in window_sizes_sec:
        idx = window // (num_of_samples_slot // sample_freq)
        print(f'Mean AOE SVD window {window} is: {mean_aoe_svd_degrees_per_num_samples_list[idx]:.4f}°')
        print(f'Mean angles error SVD window {window} is: {mean_angles_error_svd_degrees_per_num_samples_list[idx]}')

    return mean_aoe_svd_degrees_per_num_samples_list, mean_angles_error_svd_degrees_per_num_samples_list, svd_time_list

