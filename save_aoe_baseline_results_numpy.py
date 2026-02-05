import os

import numpy as np


def save_aoe_baseline_results_numpy(mean_aoe_svd_degrees_per_num_samples_list, svd_time_list,aoe_test_list, current_time_test_list, config):
    """
    Save results as NumPy files.

    Args:
        mean_rmse_svd_degrees_per_num_samples_list: List of RMSE values
        svd_time_list: List of time values
        config: Includes directory to save the results
    """
    data_path = config['data_path']
    saved_aoe_results_file_name = config['saved_rmse_results_file_name']  ##should be rmse here and not aoe
    # rmse_saved_rmse_results_file_name = f"rmse_{saved_rmse_results_file_name}.npy"



    np.save(os.path.join(data_path, 'aoe_results_dir', f"aoe_baseline_{saved_aoe_results_file_name}.npy"), np.array(mean_aoe_svd_degrees_per_num_samples_list))
    np.save(os.path.join(data_path, 'aoe_results_dir', f"timeline_baseline_{saved_aoe_results_file_name}.npy"), np.array(svd_time_list))
    np.save(os.path.join(data_path, 'aoe_results_dir', f"aoe_{saved_aoe_results_file_name}.npy"), np.array(aoe_test_list))
    np.save(os.path.join(data_path, 'aoe_results_dir', f"timeline_{saved_aoe_results_file_name}.npy"), np.array(current_time_test_list))

    print(f"Results saved to {saved_aoe_results_file_name}")