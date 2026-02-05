import os

import numpy as np


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
    eval_metric = config['eval_metric']
    loaded_rmse_results_file_names_list = []
    loaded_timeline_results_file_names_list = []

    if eval_metric == 'RMSE':
        eval_name = 'rmse'
    elif eval_metric == 'AOE':
        eval_name = 'aoe'

    for file_name in loaded_file_names_list:
        loaded_rmse_results_file_names_list.append(np.load(os.path.join(data_path,f'{eval_name}_results_dir',f"{eval_name}_{file_name}.npy")).tolist())
        loaded_timeline_results_file_names_list.append(np.load(os.path.join(data_path,f'{eval_name}_results_dir',f"timeline_{file_name}.npy")).tolist())

    # mean_rmse_svd = np.load(f"rmse_results_dir/mean_rmse_svd.npy").tolist()
    # svd_time = np.load(f"rmse_results_dir/svd_time.npy").tolist()

    return loaded_rmse_results_file_names_list, loaded_timeline_results_file_names_list