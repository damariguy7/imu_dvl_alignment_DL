import numpy as np

from calc_error_functions import calc_squared_err_angles, calc_err_angles
from run_svd_solution_for_wahba_problem import run_svd_solution_for_wahba_problem


def calc_mean_rmse_svd_degrees_per_num_samples(v_imu_dvl_test_series_list, sample_freq, config):

    dataset_len = len(v_imu_dvl_test_series_list[0][0])
    window_sizes_sec = config['window_sizes_sec']

    max_err = 0

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
            curr_max_err = np.max(np.abs(euler_angles_svd_degrees))
            if curr_max_err > max_err:
                max_err = curr_max_err
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

    return mean_rmse_svd_degrees_per_num_samples_list, mean_angles_error_svd_degrees_per_num_samples_list, svd_time_list, max_err
