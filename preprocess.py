"""
Author: Christoph Karl Heck
Date: 2025-03-25
Description: This script preprocesses CSV files for phyto and ozone node data. 
             It includes data filtering, resampling, smoothing, scaling, and visualization.
             The output includes preprocessed CSV files and visual plots.
"""

from datetime import datetime
from rich.console import Console
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import argparse
from scipy.stats import mode
import matplotlib.dates as mdates

from typing import Optional

# Constants
CONFIG = {
    "DATABITS": 8388608,
    "VREF": 2.5,
    "GAIN": 4.0,
    "MIN_VALUE": -200,
    "MAX_VALUE": 200,
    "FACTOR": 1000, # not get numerical differention
    "BEFORE": 30, # minutes before stimulus
    "AFTER": 30, # minutes after stimulus
    "CHUNK_SIZE": 10 ,#min
    "GROUP_SIZE": 6 # 18s merged
}

# Initialize the console
console = Console()

def get_precomputed_path(data_dir: str, name: str):

    return os.path.join(data_dir, name)


def discover_files(data_dir: str, prefix: str) -> list[str]:
    """
    Discover CSV files matching the given prefix in the specified directory.
    Args:
        data_dir (str): Directory to search.
        prefix (str): File prefix to match.
    Returns:
        list[str]: List of matching file paths.
    """
    console.print(f"[bold cyan]Discovering files with prefix '{prefix}' in '{data_dir}'[/bold cyan]")
    files = glob.glob(os.path.join(data_dir, f"{prefix}*.csv"))
    console.print(f"Found [bold yellow]{len(files)}[/bold yellow] matching files.")
    return files


def load_and_merge_data(data_dir: str, files: list, resample_rate: str) -> pd.DataFrame:
    columns = ["differential_potential_pn1", "differential_potential_pn3", "O3_1", "O3_2"]
    combined_df = None  # Start with an empty DataFrame
    list_df=[]

    for column in columns:
        usecols = ["timestamp", column]  # Read only timestamp and the specific column
        df_of_files = []

        for file in files:
            df_f = pd.read_csv(file, parse_dates=["timestamp"], usecols=usecols)
            df_f.rename(columns={"timestamp": "datetime"}, inplace=True)
            df_f = df_f.dropna()
            df_f = df_f.drop_duplicates(subset=["datetime"])
            df_of_files.append(df_f)
        
        
        df = pd.concat(df_of_files)
        df.set_index("datetime", inplace=True)
        df = df.sort_index()
        df = df.resample(resample_rate).mean().interpolate()
        list_df.append(df)

    # Merge all DataFrames on 'datetime' by concatenating along columns
    combined_df = pd.concat(list_df, axis=1).reset_index()
    combined_df['datetime'] = pd.to_datetime(combined_df['datetime'])
    combined_df['datetime'] = combined_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    combined_df.set_index("datetime", inplace=True, drop=False)

    # Write precomputed data to file
    path = get_precomputed_path(data_dir, f"precomputed_experiments_{resample_rate}.csv")
    combined_df.to_csv(path, index=True)

    return combined_df


def convert_to_mv(df: pd.DataFrame, column: str) -> pd.DataFrame:
    
    df[column] = ((df[column] / CONFIG["DATABITS"] - 1) * CONFIG["VREF"] / CONFIG["GAIN"]) * 1000

    return df


def load_times(file: str) -> pd.DataFrame:

    df = pd.read_csv(file)
    df["times"] = pd.to_datetime(df['times'], format="%Y-%m-%d %H:%M:%S")

    return df


def min_max_normalization(df: pd.DataFrame, column: str) -> None:

    df[column] = ((df[column] - CONFIG["MIN_VALUE"]) / (
        CONFIG["MAX_VALUE"] - CONFIG["MIN_VALUE"])) * CONFIG["FACTOR"]
    


def adjusted_min_max_normalization(df: pd.DataFrame, column: str) -> None:

    df[column] = (df[column] - (CONFIG["MIN_VALUE"]/CONFIG["FACTOR"])) / (
        (CONFIG["MAX_VALUE"]/CONFIG["FACTOR"]) - (CONFIG["MIN_VALUE"]/CONFIG["FACTOR"]))
    


def none_1000(df: pd.DataFrame, column: str) -> None:

    df[column] = df[column] *1000


def min_max_important_data(df: pd.DataFrame) -> None:
    columns = ["differential_potential_pn1", "differential_potential_pn3"]
    
    def mm_chunk(chunk):
        # Compute min and max for the list
        chunk_min = min(chunk)
        chunk_max = max(chunk)
        if chunk_max != chunk_min:
            # Normalize each element in the chunk
            return [((x - chunk_min) / (chunk_max - chunk_min))*CONFIG["FACTOR"] if pd.notnull(x) else x for x in chunk]
        else:
            return [0.0 for x in chunk]

    # Apply the normalization function to each cell in the specified columns.
    for column in columns:
        df[column] = df[column].apply(mm_chunk)


def z_score_important_data(df: pd.DataFrame) -> None:
    columns = ["differential_potential_pn1", "differential_potential_pn3"]
    
    def zs_chunk(chunk):
        # Compute min and max for the list
        chunk_mean = np.mean(chunk)
        chunk_std = np.std(chunk)

        return [((x - chunk_mean) / chunk_std) if pd.notnull(x) else x for x in chunk]

    # Apply the normalization function to each cell in the specified columns.
    for column in columns:
        df[column] = df[column].apply(zs_chunk)


def z_score_chunk(df: pd.DataFrame) -> None:
    
    def zs(chunk):
        # Compute min and max for the list
        chunk_mean = np.mean(chunk)
        chunk_std = np.std(chunk)

        return [((x - chunk_mean) / chunk_std) if pd.notnull(x) else x for x in chunk]

    df["chunk"] = df["chunk"].apply(zs)


def extract_important_data(df_phyto: pd.DataFrame, df_times: pd.DataFrame, data_dir) -> pd.DataFrame:
    """
    Important data equals x min before and x min after stimulus
    """

    rows = []
    for stimulus_time in df_times['times'][1:]:
        start_time = stimulus_time - pd.Timedelta(minutes=CONFIG["BEFORE"])
        end_time = stimulus_time + pd.Timedelta(minutes=CONFIG["AFTER"])

        subset = df_phyto.loc[start_time:end_time]

        dp_pn1 = subset['differential_potential_pn1'].tolist()
        dp_pn3 = subset['differential_potential_pn3'].tolist()

        # Check if both lists contain no NaN values
        if not np.any(np.isnan(dp_pn1)) and not np.any(np.isnan(dp_pn3)):
            rows.append({
            'start_time': start_time,
            'stimulus_time': stimulus_time,
            'end_time': end_time,
            'differential_potential_pn1': dp_pn1,
            'differential_potential_pn3': dp_pn3
            })
        
    result_df = pd.DataFrame(rows)

    file_path = os.path.join(data_dir, "important_data.csv")
    result_df.to_csv(file_path, index=False)  
    
    return result_df


def label_ground_truth(df_phyto: pd.DataFrame, df_times: pd.DataFrame) -> pd.DataFrame:
    # Ensure ground_truth column exists and is initialized to 0
    df_phyto['ground_truth'] = 0

    for stimulus_time in df_times['times']:
        start = pd.to_datetime(stimulus_time)
        end = start + pd.Timedelta(minutes=CONFIG["AFTER"])
        mask = (df_phyto['datetime'] >= start) & (df_phyto['datetime'] <= end)
        df_phyto.loc[mask, 'ground_truth'] = 1

    return df_phyto


def extract_simulation_data(df_phyto: pd.DataFrame, minutes: int, nbr_values: int) -> pd.DataFrame:
    print("Extract Simulation data")

    df_phyto = df_phyto.sort_index()
    df_phyto = df_phyto.dropna(subset=["differential_potential_pn1", "differential_potential_pn3"])
    start_time = pd.to_datetime(df_phyto['datetime'].min())
    end_time = pd.to_datetime(df_phyto['datetime'].max())

    seconds = (minutes / nbr_values) * 60

    simulation_data = []

    datetime_end = start_time + pd.Timedelta(minutes=minutes)
    arr = df_phyto[(df_phyto['datetime'] >= start_time) & (df_phyto['datetime'] < datetime_end)]

    signal_ch0 = arr["differential_potential_pn1"].to_numpy()
    resampled_ch0 = np.interp(
            np.linspace(0, len(signal_ch0) - 1, 100),
            np.arange(len(signal_ch0)),
            signal_ch0
        ).tolist()

    signal_ch1 = arr["differential_potential_pn3"].to_numpy()
    resampled_ch1 = np.interp(
            np.linspace(0, len(signal_ch1) - 1, 100),
            np.arange(len(signal_ch1)),
            signal_ch1
        ).tolist()
    
    row = {
        "datetime": datetime_end,
        "input_not_normalized_ch0": resampled_ch0,
        "input_not_normalized_ch1": resampled_ch1,
    }

    simulation_data.append(row)
    
    current = datetime_end
    while current + pd.Timedelta(seconds=seconds) <= end_time:
        print("Current:", current, "| End:", end_time)

        segment = df_phyto[(df_phyto['datetime'] >= current) & (df_phyto['datetime'] < current + pd.Timedelta(seconds=seconds))]

        # Downsample to 100 points using interpolation
        signal_ch0 = segment["differential_potential_pn1"].to_numpy()
        resampled_ch0 = resampled_ch0[1:] + [float(np.mean(signal_ch0))]

        signal_ch1 = segment["differential_potential_pn3"].to_numpy()
        resampled_ch1 = resampled_ch1[1:] + [float(np.mean(signal_ch1))]

        row = {
            "datetime": current + pd.Timedelta(seconds=seconds),
            "input_not_normalized_ch0": resampled_ch0,
            "input_not_normalized_ch1": resampled_ch1,
        }

        simulation_data.append(row)

        current += pd.Timedelta(seconds=6)

    return pd.DataFrame(simulation_data)


def split_data_in_Xmin_chunks(df: pd.DataFrame) -> pd.DataFrame:
    "X min training chunks"

    chunk_seconds = CONFIG["CHUNK_SIZE"] * 60
    nbr_chunks = int(((CONFIG["BEFORE"]*60) + (CONFIG["AFTER"]*60))/chunk_seconds)

    new_rows = []

    # Iterate over each row in result_df
    for idx, row in df.iterrows():
        start_time = row['start_time']
        stimulus_time = row['stimulus_time']
        pn1_list = row['differential_potential_pn1']
        pn3_list = row['differential_potential_pn3']

        for i in range(nbr_chunks):
            chunk_ch0 = pn1_list[i*chunk_seconds:(i+1)*chunk_seconds]
            chunk_ch1 = pn3_list[i*chunk_seconds:(i+1)*chunk_seconds]

            ozone = 0 if i < nbr_chunks/2 else 1 # First half 0, second half 1

            new_rows.append({
                'start_time': start_time,
                'stimulus_time': stimulus_time,
                'end_time': start_time + pd.Timedelta(minutes=CONFIG["CHUNK_SIZE"]),
                'channel': 0,
                'ozone': ozone,
                'chunk': chunk_ch0
            })
        
            # Create a row for the "after" slice (Ozone = 1)
            new_rows.append({
                'start_time': start_time,
                'stimulus_time': stimulus_time,
                'end_time': start_time + pd.Timedelta(minutes=CONFIG["CHUNK_SIZE"]),
                'channel': 1,
                'ozone': ozone,
                'chunk': chunk_ch1
            })

            start_time = start_time + pd.Timedelta(minutes=CONFIG["CHUNK_SIZE"])

    df = pd.DataFrame(new_rows)

    return df

def downsample_by_mean(data):
    data = np.array(data)
    if data.shape[0] % CONFIG["GROUP_SIZE"] != 0:
        raise ValueError("Length of data must be a multiple of 6.")

    # Reshape the data into a 2D array with each row containing 6 values.
    reshaped_data = data.reshape(-1, CONFIG["GROUP_SIZE"])
    
    # Compute the mean along the axis 1 (i.e. for each row)
    downsampled = reshaped_data.mean(axis=1)
    return downsampled


def split_data_in_rolling_chunks(df: pd.DataFrame) -> pd.DataFrame:
    "X min training chunks"

    chunk_seconds = CONFIG["CHUNK_SIZE"] * 60

    group_size = CONFIG["GROUP_SIZE"]

    new_rows = []

    # Iterate over each row in result_df
    for idx, segment in df.iterrows():
        start_time = segment['start_time']
        stimulus_time = segment['stimulus_time']
        end_time = start_time + pd.Timedelta(minutes=CONFIG["CHUNK_SIZE"])
        pn1_list = segment['differential_potential_pn1']
        pn3_list = segment['differential_potential_pn3']

        chunk_ch0 = downsample_by_mean(pn1_list[0:chunk_seconds])
        new_rows.append({
                'start_time': start_time,
                'stimulus_time': stimulus_time,
                'end_time': end_time,
                'channel': 0,
                'ozone': 0,
                'chunk': chunk_ch0
            })


        chunk_ch1 = downsample_by_mean(pn3_list[0:chunk_seconds])
        new_rows.append({
                'start_time': start_time,
                'stimulus_time': stimulus_time,
                'end_time': end_time,
                'channel': 0,
                'ozone': 0,
                'chunk': chunk_ch1
            })

        pn1_list_after_600 = pn1_list[chunk_seconds:]
        pn3_list_after_600 = pn3_list[chunk_seconds:]
        n_groups_ch0 = len(pn1_list) // group_size
        n_groups_ch1 = len(pn3_list) // group_size
        n_groups = min(n_groups_ch0, n_groups_ch1)
        for i in range(n_groups):
            start_time = start_time + pd.Timedelta(seconds=group_size)
            end_time = start_time + pd.Timedelta(minutes=CONFIG["CHUNK_SIZE"])
            ozone_flag = 1 if end_time > stimulus_time and end_time < (stimulus_time + pd.Timedelta(minutes=20)) else 0
            group_ch0 = np.array(pn1_list_after_600[i*group_size:(i+1)*group_size])
            group_ch1 = np.array(pn3_list_after_600[i*group_size:(i+1)*group_size])

            if group_ch0.size > 0 and group_ch1.size > 0:
                mean_ch0 = group_ch0.mean()
                mean_ch1 = group_ch1.mean()

                chunk_ch0 = np.append(chunk_ch0[1:], mean_ch0)
                chunk_ch1 = np.append(chunk_ch1[1:], mean_ch1)

                new_rows.append({
                    'start_time': start_time,
                    'stimulus_time': stimulus_time,
                    'end_time': end_time,
                    'channel': 0,
                    'ozone': ozone_flag,
                    'chunk': chunk_ch0
                })

                new_rows.append({
                    'start_time': start_time,
                    'stimulus_time': stimulus_time,
                    'end_time': end_time,
                    'channel': 1,
                    'ozone': ozone_flag,
                    'chunk': chunk_ch1
                })
        


    df = pd.DataFrame(new_rows)

    # For plotting, you may want to focus on one channel (say, Channel 0).
    df_channel0 = df[df['channel'] == 0]

    # Create a scatter plot showing the heat classification over time.
    plt.figure(figsize=(12, 6))
    # We'll plot 1 for heat and 0 for not heat; using different colors.
    plt.scatter(df_channel0['end_time'], df_channel0['ozone'], 
                c=df_channel0['ozone'], cmap='coolwarm', marker='o')
    plt.xlabel("Datetime")
    plt.ylabel("Ozone Classification (0 = Not Ozone, 1 = Ozone)")
    plt.title("Ozone Classification Over Time (Channel 0)")
    plt.grid(True)
    plt.show()

    # Separate the two classes.
    df_ozone0 = df[df['ozone'] == 0]
    df_ozone1 = df[df['ozone'] == 1]

    # Determine the size of the minority class.
    n_min = min(len(df_ozone0), len(df_ozone1))

    # Downsample both classes to have the same number of rows.
    df_ozone0_balanced = df_ozone0.sample(n=n_min, random_state=42)
    df_ozone1_balanced = df_ozone1.sample(n=n_min, random_state=42)

    # Combine the balanced classes.
    df = pd.concat([df_ozone0_balanced, df_ozone1_balanced]).sort_values('start_time').reset_index(drop=True)

    counts_original = df['ozone'].value_counts().sort_index()
    # Count classes in the balanced dataset
    counts_balanced = df['ozone'].value_counts().sort_index()

    # Create a figure with two subplots side by side.
    plt.figure(figsize=(12, 5))

    # Plot the original distribution.
    plt.subplot(1, 2, 1)
    plt.bar(counts_original.index, counts_original.values, tick_label=["Ozone=0", "Ozone=1"])
    plt.title("Original Class Distribution")
    plt.xlabel("Ozone Class")
    plt.ylabel("Count")

    # Plot the balanced distribution.
    plt.subplot(1, 2, 2)
    plt.bar(counts_balanced.index, counts_balanced.values, tick_label=["Ozone=0", "Ozone=1"])
    plt.title("Balanced Class Distribution")
    plt.xlabel("Ozone Class")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()

    return df


def split_chunks_in_columns(df: pd.DataFrame) -> pd.DataFrame:

    new_rows = []

    for idx, row in df.iterrows():
        new_row = {
            'start_time': row["start_time"],
            'stimulus_time': row['stimulus_time'],
            'end_time': row["end_time"],
            'channel': row["channel"],
            'ozone': row["ozone"]
        }
        new_row.update({f'val_{i}': row["chunk"][i] for i in range(len(row["chunk"]))})
        new_rows.append(new_row)

    return pd.DataFrame(new_rows)


def plot_final(df: pd.DataFrame) -> None:

    fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

    for ozone, color in zip([0, 1], ['blue', 'orange']):
        subset = df[(df['channel'] == 0) & (df['ozone'] == ozone)]
        first_entry = True
        for _, row in subset.iterrows():
            time_range = pd.date_range(start=row['start_time'], end=row['end_time'], periods=len(row)-5)
            values = row[[col for col in row.index if col.startswith('val_')]].values
            axes[0].plot(time_range, values, color=color, alpha=0.5, label=f'CH0 Ozone {ozone}' if first_entry else "")
            first_entry = False
    axes[0].set_title('CH0 Signal')
    axes[0].set_ylabel('[normalized]')
    axes[0].grid()
    axes[0].legend()

    for ozone, color in zip([0, 1], ['blue', 'orange']):
        subset = df[(df['channel'] == 1) & (df['ozone'] == ozone)]
        first_entry = True
        for _, row in subset.iterrows():
            time_range = pd.date_range(start=row['start_time'], end=row['end_time'], periods=len(row)-5)
            values = row[[col for col in row.index if col.startswith('val_')]].values
            axes[1].plot(time_range, values, color=color, alpha=0.5, label=f'CH1 Ozone {ozone}' if first_entry else "")
            first_entry = False
    axes[1].set_title('CH1 Signal')
    axes[1].set_ylabel('[normalized]')
    axes[1].grid()
    axes[1].legend()

    plt.xlabel('Datetime')
    plt.tight_layout()
    plt.show()


def plot_basic_data(df_phyto: pd.DataFrame, df_times: pd.DataFrame, mark_stimulus_window: bool) -> None:


    fig_width = 5.90666  # Width in inches
    aspect_ratio = 0.618  # Example aspect ratio (height/width)
    fig_height = fig_width * aspect_ratio


    # Create subplots
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(fig_width, 6), sharex=True)

    # Plot each column in a separate subplot
    # relevant sind differential_potential_pn1 (leaf), differnetial_potential_pn3 (stem), O3_1 und O3_2 die zwei ozon
    # sensoren (aber ich weiß nicht welcher oben und welcher unten ist)
    plot_info = [
        {"col": "differential_potential_pn1", "label": "CH0 Resampled" ,"title": "CH0 Resampled", "ylabel": "Voltage [mv]", "color": "#6aaed6"},
        {"col": "differential_potential_pn3", "label": "CH1 Resampled" ,"title": "CH1 Resampled", "ylabel": "Voltage [mv]", "color": "#1764ab"},
        {"col": "O3_1", "label": "Ozone Level" ,"title": "O3_1", "ylabel": r'O$_3$ [ppb]', "color": '#1f77b4'},
        #{"col": "O3_2", "title": "O3_2", "ylabel": "Ozone [ppb]"}
]

    for i, info in enumerate(plot_info):
        ax = axes[i]
        # Plot the data using the details from plot_info
        ax.plot(df_phyto["datetime"], df_phyto[info["col"]], label=info["label"], linewidth=1, color=info["color"])
        #ax.set_title(info["title"])
        ax.grid(True)
        ax.set_ylabel(info["ylabel"])
        ax.legend()

        ax.xaxis.set_major_locator(
        mdates.HourLocator(byhour=range(0,24,6))
        )
        ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%H:%M")
        )

        # Loop through each event in df_times to mark the area around each event
        if mark_stimulus_window:
            for start_time in df_times['times']:
                axes[i].axvline(start_time, color='blue', linestyle='--', linewidth=1.5)
                axes[i].axvspan(start_time - pd.Timedelta(minutes=CONFIG["BEFORE"]),
                                start_time + pd.Timedelta(minutes=CONFIG["AFTER"]),
                                color='blue', alpha=0.2)

    # Set common x-label
    plt.xticks(rotation=0)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.savefig(f"OzoneRaw.pgf", format="pgf", bbox_inches="tight", pad_inches=0.05)
    plt.show()


def plot_extracted_data(df: pd.DataFrame) -> None:

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)

    for lst in df["differential_potential_pn1"]:
        axes[0].plot(lst)
        axes[0].set_title("Ozone CH0")
        axes[0].set_ylabel("[normalized]")

    for lst in df["differential_potential_pn3"]:
        axes[1].plot(lst)
        axes[1].set_title("Ozone CH1")
        axes[1].set_ylabel("[normalized]")

    plt.xlabel("[seconds]")
    plt.ylabel("[normalized]")
    plt.show()


def plot_extracted_data_stats(df: pd.DataFrame) -> None:

    channel1_data = np.stack(df["differential_potential_pn1"].values)
    channel3_data = np.stack(df["differential_potential_pn3"].values)
    
    mean_ch0 = np.mean(channel1_data, axis=0)
    std_ch0 = np.std(channel1_data, axis=0)
    mean_ch1 = np.mean(channel3_data, axis=0)
    std_ch1 = np.std(channel3_data, axis=0)
    
    x = np.arange(mean_ch0.shape[0])

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
    
    axes[0].plot(x, mean_ch0, label="Mean CH0", color="blue")
    axes[0].fill_between(x, mean_ch0 - std_ch0, mean_ch0 + std_ch0, color="blue", alpha=0.3, label="Std. Dev.")
    axes[0].set_title("Ozone CH0 - Mean and Std")
    axes[0].set_ylabel("[normalized]")
    axes[0].legend()
    
    axes[1].plot(x, mean_ch1, label="Mean CH1", color="green")
    axes[1].fill_between(x, mean_ch1 - std_ch1, mean_ch1 + std_ch1, color="green", alpha=0.3, label="Std. Dev.")
    axes[1].set_title("Ozone CH1 - Mean and Std")
    axes[1].set_ylabel("[normalized]")
    axes[1].legend()
    
    plt.xlabel("[seconds]")
    plt.show()


def save_config_to_txt(configuration: dict, directory: str, prefix: str) -> None:
    """
    Save the global configuration dictionary to a .txt file in the specified directory
    with a filename based on the given prefix.
    Args:
        configuration (dict): The configuration dictionary to save.
        directory (str): The directory where the file will be saved.
        prefix (str): The prefix for the filename.
    """
    filename = os.path.join(directory, f"{prefix}_config_used_for_preprocessing.txt")

    try:
        # Write the configuration to the file
        with open(filename, "w") as file:
            for key, value in configuration.items():
                file.write(f"{key} = {value}\n")
        
        console.print(f"[bold green]Configuration successfully saved to '{filename}'[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Failed to save configuration to '{filename}': {e}[/bold red]")


def check_for_precomputation(data_dir: str, resample_rate: str) -> pd.DataFrame:
    # Process Phyto Node 
    df_phyto = None
    path = get_precomputed_path(data_dir, f"precomputed_experiments_{resample_rate}.csv")
    if os.path.exists(path):
        df_phyto = pd.read_csv(path)
        df_phyto['datetime'] = pd.to_datetime(df_phyto['datetime'], format="%Y-%m-%d %H:%M:%S.%f")
        df_phyto.set_index("datetime", inplace=True, drop=False)
    else:
        phyto_files = discover_files(data_dir, "experiment")
        df_phyto = load_and_merge_data(data_dir, phyto_files, resample_rate)
    
    return df_phyto


def preprocess(data_dir: str, normalization: str, resample_rate: str):

    df_phyto = check_for_precomputation(data_dir, resample_rate)

    df_phyto = convert_to_mv(df_phyto, "differential_potential_pn1")
    df_phyto = convert_to_mv(df_phyto, "differential_potential_pn3")

    # Read Ozone times file
    times_files = discover_files(data_dir, "times")
    df_times = load_times(times_files[0])

    plot_basic_data(df_phyto, df_times, False)
    plot_basic_data(df_phyto, df_times, True)

    if normalization == "min-max":
        min_max_normalization(df_phyto, "differential_potential_pn1")
        min_max_normalization(df_phyto, "differential_potential_pn3")

    if normalization == "adjusted-min-max":
        adjusted_min_max_normalization(df_phyto, "differential_potential_pn1")
        adjusted_min_max_normalization(df_phyto, "differential_potential_pn3")

    if normalization == "none-1000":
        none_1000(df_phyto, "differential_potential_pn1")
        none_1000(df_phyto, "differential_potential_pn3")

    df_important_data = extract_important_data(df_phyto, df_times, data_dir)
    
    if normalization == "min-max-chunk":
        min_max_important_data(df_important_data)

    if normalization == "z-score-chunk":
        z_score_important_data(df_important_data)

    plot_extracted_data(df_important_data)
    plot_extracted_data_stats(df_important_data)

    # split data in 10min chunks
    #df_training_split = split_data_in_Xmin_chunks(df_important_data)
    df_training_split = split_data_in_rolling_chunks(df_important_data)
    print(df_training_split.describe())

    if normalization == "z-score":
        z_score_chunk(df_training_split)

    df_final = split_chunks_in_columns(df_training_split)

    plot_final(df_final)

    preprocessed_folder = os.path.join(data_dir, "preprocessed")
    os.makedirs(preprocessed_folder, exist_ok=True)
    final_path = get_precomputed_path(preprocessed_folder, f"training_data_{normalization}.csv")

    df_final.to_csv(final_path, index=True)


def create_simulation_files(data_dir: str, resample_rate: str) -> None:
   
    df_phyto = check_for_precomputation(data_dir, resample_rate)

    df_phyto = convert_to_mv(df_phyto, "differential_potential_pn1")
    df_phyto = convert_to_mv(df_phyto, "differential_potential_pn3")

    times_files = discover_files(data_dir, "times")
    df_times = load_times(times_files[0])

    df_phyto = extract_simulation_data(df_phyto, 10, 100)
    df_phyto = label_ground_truth(df_phyto, df_times)

    preprocessed_folder = os.path.join(data_dir, "simulation")
    os.makedirs(preprocessed_folder, exist_ok=True)
    final_path = get_precomputed_path(preprocessed_folder, f"simulation_data.csv")
    df_phyto.to_csv(final_path, index=True)



def main():
    parser = argparse.ArgumentParser(description="Preprocess CSV files.")
    parser.add_argument("--data-dir", required=True, type=str, help="Directory with raw files.")
    parser.add_argument(
        "--normalization",
        required=False,
        type=str,
        default=None,
        choices=["min-max", "adjusted-min-max", "min-max-chunk", "z-score-chunk", "z-score", "none-1000"],
        help="Normalization method to apply. Options: min-max, adjusted-min-max, min-max-chunk, z-score-chunk, or z-score."
    )
    parser.add_argument("--create-simulation-files", required=False, type=bool, default=False, help="Create Simulation Files")
    args = parser.parse_args()

    # Now you can use args.normalization to conditionally apply normalization
    if args.normalization is not None:
        print(f"Normalization method chosen: {args.normalization}")
    else:
        print("No normalization method specified.")

    # Normalize and validate inputs
    data_dir = args.data_dir
    normalization = args.normalization
    create_simulation_files_flag = args.create_simulation_files

    # Print Input Parameters
    console.print(f"[bold green]Data Directory:[/bold green] {data_dir}")

    if create_simulation_files_flag:
        create_simulation_files(data_dir, "10ms")

    else:
        preprocess(data_dir, normalization, "1s")



if __name__ == "__main__":
    main()
