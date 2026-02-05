import numpy as np
from sklearn.model_selection import train_test_split


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