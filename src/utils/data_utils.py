import pickle
import os
import yaml


def save_pickle_data(data, save_path):
    """
    Saves data to disk in pickle format.

    :param data: Dictionary containing data.
    :param save_path: Path to save the pickle file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Data saved to {save_path}")


def load_pickle_data(file_path):
    """
    Loads data from a pickle file.

    :param file_path: Path to the pickle file.
    :return: Dictionary containing data.
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    print(f"Data loaded from {file_path}")
    return data


def read_calibration_from_yaml(file_path):
    """
    Reads calibration data from a YAML file.
    :param file_path: Path to the YAML file.
    :return: Tuple of rotation (dict) and translation (dict).
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    rotation = data['camera_to_lidar']['rotation']
    translation = data['camera_to_lidar']['translation']
    return rotation, translation
