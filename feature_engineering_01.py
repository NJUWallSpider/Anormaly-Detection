# Feature Engineering
import pandas as pd
import numpy as np
import scipy
from typing import Tuple, List
# from dtaidistance import dtw_barycenter
# from dtaidistance import dtw
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

def load_wafer_data(filepath: str) -> List[np.ndarray]:
    """
    Load and normalize wafer production data from an Excel file.
    
    Args:
        filepath: Path to data.xlsx file containing 52 sheets
    
    Returns:
        multi_profile_data: List of normalized wafer profile arrays.
    """
    # Read Excel file
    print(f"Loading Excel file: {filepath}")
    excel_file = pd.ExcelFile(filepath)
    
    # Initialize list to store profile data
    multi_profile_data = []
    
    # Process each sheet
    for sheet_name in excel_file.sheet_names:
        # Read sheet data
        sheet_data = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        # Convert sheet data to numpy array
        # Assuming all columns except any potential label column are sensor data
        sensor_data = sheet_data.values
        
        # Add to profile data list
        multi_profile_data.append(sensor_data)
    
    print(f"Loaded {len(multi_profile_data)} wafer profiles")
    
    # Print shape information for verification
    for i, profile in enumerate(multi_profile_data):
        print(f"Profile {i+1} shape: {profile.shape}")
    
    return multi_profile_data

def preprocess_data(data):
    """
    Preprocess multi-profile data addressing redundant sensors and normalization
    """
    print("Starting data preprocessing...")
    
    # Handle different sample lengths by padding or truncating
    max_length = max([sample.shape[0] for sample in data])
    min_length = min([sample.shape[0] for sample in data])
    print(f"Sample length range: {min_length} - {max_length}")
    
    # Standardize to common length (use minimum to avoid extrapolation)
    standardized_data = []
    for i, sample in enumerate(data):
        standardized_sample = sample[:min_length, :]
        standardized_data.append(standardized_sample)
    
    # Convert to 3D array: (samples, time_points, sensors)
    data_array = np.array(standardized_data)
    print(f"Data shape after standardization: {data_array.shape}")
    
    return data_array, min_length

def normalize_profiles_robustly(multi_profile_data: List[np.ndarray]) -> List[np.ndarray]:
    """
    Standardize multi-profile data using RobustScaler.

    Each feature (sensor) is scaled independently based on its statistics
    (median and IQR) across all time points and all samples. This handles 
    variable-length samples correctly and is robust to outliers.

    Args:
        multi_profile_data: A list of wafer profile arrays, where each
                            array has a shape of (time_steps, num_sensors).
                            time_steps can be variable.

    Returns:
        A new list of scaled wafer profile arrays with the same structure.
    """
    if not multi_profile_data:
        return []

    print("\nApplying RobustScaler to all sensor data...")
    
    # 1. Concatenate all samples vertically into one large 2D array.
    # This works because the number of sensors (columns) is consistent.
    # The shape will be (total_time_steps_across_all_samples, num_sensors).
    all_data_concatenated = np.vstack(multi_profile_data)
    
    # 2. Initialize and fit the RobustScaler.
    # It will compute median and IQR for each sensor column independently.
    scaler = RobustScaler()
    scaler.fit(all_data_concatenated)
    print("RobustScaler fitted on the complete dataset.")

    # 3. Transform each original sample individually using the fitted scaler.
    scaled_multi_profile_data = []
    for i, sample in enumerate(multi_profile_data):
        scaled_sample = scaler.transform(sample)
        scaled_multi_profile_data.append(scaled_sample)
        if i < 3: # Print info for the first few samples to verify
             print(f"Sample {i+1} transformed. Original shape: {sample.shape}, Scaled shape: {scaled_sample.shape}")

    return scaled_multi_profile_data

def feature_extraction(data_array, n_sensors_to_select=38):
    """
    Select top sensors based on variance, then filter them based on correlation.
    This method first selects the most informative sensors and then removes
    redundant ones to create a robust feature set.
    
    Args:
        data_array (np.ndarray): The preprocessed data array.
        n_sensors_to_select (int): The number of most informative sensors to select.
        
    Returns:
        np.ndarray: A 3D array containing the final selected sensor data.
    """
    print(f"\nStarting feature selection...")
    print(f"Step 1: Select top {n_sensors_to_select} sensors based on variance.")
    n_samples, n_timepoints, n_sensors = data_array.shape

    # 1. Select the most informative sensors (e.g., based on overall variance)
    sensor_variances = np.var(data_array.reshape(-1, n_sensors), axis=0)
    top_sensor_indices = np.argsort(sensor_variances)[-n_sensors_to_select:]
    print(f"  - Initial selection (by variance): {sorted(top_sensor_indices.tolist())}")

    # 2. Extract the data for these sensors
    selected_data = data_array[:, :, top_sensor_indices]

    # 3. Filter selected sensors based on correlation
    print("Step 2: Filter sensors based on high correlation (threshold > 0.95).")
    n_selected_sensors = selected_data.shape[2]

    # Reshape for correlation calculation: (total_time_points, n_selected_sensors)
    reshaped_for_corr = selected_data.reshape(-1, n_selected_sensors)

    # Calculate correlation matrix using Spearman's rank correlation
    corr_result = scipy.stats.spearmanr(reshaped_for_corr)
    corr_matrix = corr_result.correlation

    # Identify highly correlated features to remove
    # We iterate through the upper triangle of the correlation matrix
    indices_to_remove = set()
    for i in range(n_selected_sensors):
        for j in range(i + 1, n_selected_sensors):
            if abs(corr_matrix[i, j]) > 0.95:
                # If a pair is highly correlated, we mark the second one (j) for removal.
                # This is a deterministic approach, which is better for reproducibility
                # than randomly choosing one.
                indices_to_remove.add(j)

    if indices_to_remove:
        print(f"  - Found {len(indices_to_remove)} redundant sensors to remove.")
        indices_to_keep_mask = [i for i in range(n_selected_sensors) if i not in indices_to_remove]
        final_data = selected_data[:, :, indices_to_keep_mask]
        final_sensor_indices = top_sensor_indices[indices_to_keep_mask]
    else:
        print("  - No highly correlated sensors found. Keeping all selected sensors.")
        final_data = selected_data
        final_sensor_indices = top_sensor_indices

    print(f"  - Final number of sensors after filtering: {final_data.shape[2]}")
    print(f"  - Final selected sensor indices: {sorted(final_sensor_indices.tolist())}")

    return final_data, final_sensor_indices

def _align_series_with_path(series: np.ndarray, template: np.ndarray, path: list) -> np.ndarray:
    """
    Helper function to align a single multi-dimensional series to a template using a DTW path.
    For each point in the template, it averages the corresponding points from the original series.
    """
    target_length, num_sensors = template.shape
    aligned_series = np.zeros((target_length, num_sensors))
    
    # Create a mapping from each template index to a list of corresponding original series indices
    template_to_original_map = [[] for _ in range(target_length)]
    for r, c in path:
        # r is index from series, c is index from template
        template_to_original_map[c].append(r)
        
    for j in range(target_length):
        original_indices = template_to_original_map[j]
        if original_indices:
            # Average the values from the original series that map to this template point
            aligned_series[j, :] = np.mean(series[original_indices, :], axis=0)
        else:
            # If a template point is unmapped (rare, but possible at edges),
            # fill it with the value of the previously aligned point.
            if j > 0:
                aligned_series[j, :] = aligned_series[j-1, :]
            else:
                # For the very first point, if unmapped, use the first point of the original series.
                aligned_series[j, :] = series[0, :]
                
    return aligned_series

def dynamic_time_warping(data: List[np.ndarray]) -> np.ndarray:
    """
    Aligns all time series samples to a common length using DTW Barycenter Averaging.
    This function first computes an average template from all samples and then aligns
    each sample to this template using the DTW path.

    This corrected function resolves several issues from the original version:
    1. Replaces non-existent 'average_ndim' with the correct 'barycenter_average'.
    2. Fixes the incorrect loop structure to be efficient.
    3. Provides a working implementation for alignment, as 'dtw.warp' does not exist.

    Args:
        data: A list of wafer profile arrays, where each array has a shape of 
              (time_steps, num_sensors). time_steps can be variable.

    Returns:
        np.ndarray: A single 3D numpy array of shape (num_samples, target_length, num_sensors)
                    containing all aligned time series.
    """
    if not data:
        return np.array([])

    print("\nAligning time series using DTW Barycenter Averaging...")
    
    # Step 1: Calculate the average template ONCE for the entire dataset.
    # FIX: The function name is `barycenter_average`, not `average_ndim`.
    print("  - Calculating barycenter average template...")
    average_template = dtw_barycenter.barycenter_average(data, use_c=True)
    target_length = len(average_template)
    print(f"  - Alignment target length: {target_length}")

    aligned_dataset = []
    # Step 2: Align each series in the dataset to the template.
    for i, series in enumerate(data):
        # Calculate the warping path between the series and the template
        path = dtw.warping_path(series, average_template)
        
        # Align the series using the calculated path.
        aligned_series = _align_series_with_path(series, average_template, path)
        
        aligned_dataset.append(aligned_series)
        if i < 3:
            print(f"  - Aligned sample {i+1}: original shape {series.shape} -> new shape {aligned_series.shape}")

    # Step 3: Convert the list of aligned series into a single 3D numpy array.
    final_dataset = np.array(aligned_dataset)
    print(f"\nAlignment complete. Final dataset shape: {final_dataset.shape}")
    
    return final_dataset

def standardize_to_common_length(data: List[np.ndarray]) -> np.ndarray:
    """
    Standardizes a list of time series samples to a common length by truncating.

    This function finds the minimum length across all samples and truncates every
    sample to this length. This is a prerequisite for converting the list of 2D
    arrays into a single 3D numpy array.

    Args:
        data: A list of wafer profile arrays, where each array can have a
              variable number of time steps (rows).

    Returns:
        A single 3D numpy array of shape (num_samples, min_length, num_sensors).
    """
    if not data:
        return np.array([])

    # Find the minimum length across all samples
    min_length = min(sample.shape[0] for sample in data)
    print(f"\nStandardizing all samples to the minimum length: {min_length}")

    # Truncate all samples to the minimum length
    standardized_list = [sample[:min_length, :] for sample in data]

    # Convert the list of consistently-shaped arrays to a single 3D array
    data_array = np.array(standardized_list)
    print(f"Data shape after length standardization: {data_array.shape}")
    return data_array

def feature_flatten(data_array):
    n_samples, n_timepoints, n_sensors = data_array.shape

    # Flatten the standardized data
    features_array = data_array.reshape(n_samples, -1)

    print(f"Extracted features shape: {features_array.shape}")
    return features_array

def PCA_preprocessing(features_array, pca_components=0.95):
    pca = PCA(n_components=pca_components)
    features_pca = pca.fit_transform(features_array)
    if isinstance(pca_components, float) and 0 < pca_components < 1.0:
        print(f"PCA selected {pca.n_components_} components to explain {pca_components:.0%} of variance.")
    else:
        print(f"PCA reduced dimensions to: {features_pca.shape[1]}")
    return features_pca
        
def run():
    multi_profile_data = load_wafer_data("data.xlsx")
    # aligned_data = dynamic_time_warping(normalized_data)
    # normalized_data = normalize_profiles_robustly(multi_profile_data)
    aligned_data = standardize_to_common_length(multi_profile_data)
    selected_data, sensor_indices = feature_extraction(aligned_data)
    return selected_data

def run_flatten():
    selected_data = run()
    multi_profile_data = load_wafer_data("data.xlsx")
    aligned_data = standardize_to_common_length(multi_profile_data)
    aligned_data, _= feature_extraction(aligned_data)
    features_array = feature_flatten(aligned_data)
    return features_array


if __name__ == "__main__":
    run()