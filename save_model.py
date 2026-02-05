import os


def save_model(model, test_type, window_size, base_path="models"):
    """
    Save the trained model to a file with window size in the filename.

    Args:
        model (torch.nn.Module): The trained model to save
        window_size (int): The window size used for training
        base_path (str): Base directory to save models
    """
    # # Create models directory if it doesn't exist
    # os.makedirs(base_path, exist_ok=True)

    # Create filename with window size
    filepath = os.path.join(base_path, f'imu_dvl_model_{test_type}_window_{window_size}.pth')

    # Save the model
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")
    return filepath