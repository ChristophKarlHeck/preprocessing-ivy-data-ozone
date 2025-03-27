import argparse
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse, AddNoise

def get_precomputed_path(data_dir: str, name: str):

    return os.path.join(data_dir, name)

def get_data(data_dir: str, normalization: str) -> pd.DataFrame:
    path_pattern = f"{data_dir}/*/preprocessed/training_data_{normalization}.csv"
    training_files = glob.glob(path_pattern)
    df_list = []
    # Loop through each training_data directory
    for file in training_files:
        df = pd.read_csv(file)  # Read CSV file
        df_list.append(df)  # Add to list
    final_df = pd.concat(df_list, ignore_index=True)
    return final_df

def plot_final(df: pd.DataFrame) -> None:

    fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

    for ozone, color in zip([0, 1], ['blue', 'orange']):
        subset = df[(df['channel'] == 0) & (df['ozone'] == ozone)]
        first_entry = True
        for _, row in subset.iterrows():
            # Get the values for columns starting with 'val_'
            value_cols = [col for col in row.index if col.startswith('val_')]
            values = row[value_cols].values
            num_points = len(values)
            # Generate a time range that matches the number of value points
            time_range = pd.date_range(start=row['start_time'], end=row['end_time'], periods=num_points)
            axes[0].plot(time_range, values, color=color, alpha=0.5,
                         label=f'CH0 Ozone {ozone}' if first_entry else "")
            first_entry = False
    axes[0].set_title('CH0 Signal')
    axes[0].set_ylabel('[normalized]')
    axes[0].grid()
    axes[0].legend()

    for ozone, color in zip([0, 1], ['blue', 'orange']):
        subset = df[(df['channel'] == 1) & (df['ozone'] == ozone)]
        first_entry = True
        for _, row in subset.iterrows():
            value_cols = [col for col in row.index if col.startswith('val_')]
            values = row[value_cols].values
            num_points = len(values)
            time_range = pd.date_range(start=row['start_time'], end=row['end_time'], periods=num_points)
            axes[1].plot(time_range, values, color=color, alpha=0.5,
                         label=f'CH1 Ozone {ozone}' if first_entry else "")
            first_entry = False
    axes[1].set_title('CH1 Signal')
    axes[1].set_ylabel('[normalized]')
    axes[1].grid()
    axes[1].legend()

    plt.xlabel('Datetime')
    plt.tight_layout()
    plt.show()

def make_lists(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    no_ozone_df = df[df['ozone'] == 0]
    ozone_df = df[df['ozone'] == 1]

    # Extract the last 600 columns as the time series data (each row is one sample)
    no_ozone_lists = no_ozone_df.iloc[:, -600:].to_numpy()
    ozone_lists = ozone_df.iloc[:, -600:].to_numpy()

    return no_ozone_lists, ozone_lists

def down(arr: np.ndarray, count: int) -> np.ndarray:
    if arr.shape[0] > count:
        indices = np.random.choice(arr.shape[0], count, replace=False)
        arr = arr[indices]
        print("Subsampled shape:", arr.shape)
    return arr

def plot_classifications(no_ozone_lists, ozone_lists) -> None:
   
    # Normalize each time series by subtracting its first value
    no_ozone_norm = no_ozone_lists - no_ozone_lists[:, 0].reshape(-1, 1)
    ozone_norm = ozone_lists - ozone_lists[:, 0].reshape(-1, 1)

    # Compute the mean and standard deviation at each time step across samples
    mean_no_ozone = np.mean(no_ozone_norm, axis=0)
    std_no_ozone = np.std(no_ozone_norm, axis=0)

    mean_ozone = np.mean(ozone_norm, axis=0)
    std_ozone = np.std(ozone_norm, axis=0)

    # Define the time axis (here simply using indices 0 to 599)
    x = np.arange(600)

    # Create subplots for each classification group
    fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

    # Plot for ozone=0 (no ozone)
    axes[0].plot(x, mean_no_ozone, label='Mean (ozone=0)', color='blue')
    axes[0].fill_between(x, mean_no_ozone - std_no_ozone, mean_no_ozone + std_no_ozone,
                         color='blue', alpha=0.3, label='Std Dev')
    axes[0].set_title('No Ozone (ozone = 0) - Normalized by First Value')
    axes[0].set_ylabel('[normalized]')
    axes[0].legend()
    axes[0].grid(True)

    # Plot for ozone=1
    axes[1].plot(x, mean_ozone, label='Mean (ozone=1)', color='orange')
    axes[1].fill_between(x, mean_ozone - std_ozone, mean_ozone + std_ozone,
                         color='orange', alpha=0.3, label='Std Dev')
    axes[1].set_title('Ozone (ozone = 1) - Normalized by First Value')
    axes[1].set_ylabel('[normalized]')
    axes[1].legend()
    axes[1].grid(True)

    plt.xlabel('Time Step')
    plt.tight_layout()
    plt.show()

def split_chunks_in_columns(df: pd.DataFrame) -> pd.DataFrame:

    new_rows = []

    for idx, row in df.iterrows():
        new_row = {
            'ozone': row["ozone"]
        }
        new_row.update({f'val_{i}': row["signal"][i] for i in range(len(row["signal"]))})
        new_rows.append(new_row)

    return pd.DataFrame(new_rows)

def main():
    parser = argparse.ArgumentParser(description="Preprocess CSV files.")
    parser.add_argument("--data-dir", required=True, type=str, help="Directory with raw files.")
    parser.add_argument(
        "--normalization",
        required=False,
        type=str,
        default=None,
        choices=["min-max", "adjusted-min-max", "min-max-chunk", "z-score-chunk", "z-score"],
        help="Normalization method to apply. Options: min-max, adjusted-min-max, min-max-chunk, z-score-chunk, or z-score."
    )
    args = parser.parse_args()

    # Now you can use args.normalization to conditionally apply normalization
    if args.normalization is not None:
        print(f"Normalization method chosen: {args.normalization}")
    else:
        print("No normalization method specified.")

    # Normalize and validate inputs
    data_dir = args.data_dir
    normalization = args.normalization

    real_data_df = get_data(data_dir, normalization)
    #plot_final(real_data_df)

    no_ozone_lists, ozone_lists = make_lists(real_data_df)
    plot_classifications(no_ozone_lists, ozone_lists)

    

    aug_pipeline = AddNoise(scale=0.05) * 10  # This produces 240*10 = 2640 samples

    no_ozone_synthetic = aug_pipeline.augment(no_ozone_lists)
    ozone_synthetic = aug_pipeline.augment(ozone_lists)

    no_ozone_synthetic =  down(no_ozone_synthetic, 2338)
    ozone_synthetic = down(ozone_synthetic, 2338)
    plot_classifications(no_ozone_synthetic, ozone_synthetic)

    print(no_ozone_synthetic.shape)
    print(ozone_synthetic.shape)

    no_ozone_lists = list(no_ozone_lists)
    ozone_lists = list(ozone_lists)

    no_ozone_lists.extend(no_ozone_synthetic.tolist())
    ozone_lists.extend(ozone_synthetic.tolist())

    all_samples = no_ozone_lists + ozone_lists
    all_labels = [0] * len(no_ozone_lists) + [1] * len(ozone_lists)

    df_final_almost = pd.DataFrame({
    "ozone": all_labels,
    "signal": all_samples
    })

    print(df_final_almost.head())

    df_final = split_chunks_in_columns(df_final_almost)

    print(df_final.head())

    augmented_folder = os.path.join(data_dir, "augmented_training_data")
    os.makedirs(augmented_folder, exist_ok=True)
    final_path = get_precomputed_path(augmented_folder, f"augmented_data_{normalization}.csv")
    df_final.to_csv(final_path)




if __name__ == "__main__":
    main()