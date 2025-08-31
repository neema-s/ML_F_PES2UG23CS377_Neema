import numpy as np

def get_entropy_of_dataset(data: np.ndarray) -> float:
    target_col = data[:, -1]
    unique_vals, counts = np.unique(target_col, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy)


def get_avg_info_of_attribute(data: np.ndarray, attribute: int) -> float:
    n = data.shape[0]
    attr_values = data[:, attribute]
    unique_vals = np.unique(attr_values)
    avg_info = 0.0
    for v in unique_vals:
        subset = data[attr_values == v]
        weight = subset.shape[0] / n
        subset_entropy = get_entropy_of_dataset(subset)
        avg_info += weight * subset_entropy
    return float(avg_info)


def get_information_gain(data: np.ndarray, attribute: int) -> float:
    dataset_entropy = get_entropy_of_dataset(data)
    avg_info = get_avg_info_of_attribute(data, attribute)
    info_gain = dataset_entropy - avg_info
    return round(float(info_gain), 4)


def get_selected_attribute(data: np.ndarray) -> tuple:
    num_attributes = data.shape[1] - 1
    information_gains = {}
    for attr in range(num_attributes):
        information_gains[attr] = get_information_gain(data, attr)
    selected_attribute = max(information_gains, key=information_gains.get)
    return information_gains, selected_attribute
