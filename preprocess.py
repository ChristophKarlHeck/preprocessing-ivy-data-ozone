"""
Author: Christoph Karl Heck
Date: 2025-03-25
Description: This script preprocesses CSV files for phyto and ozone node data. 
             It includes data filtering, resampling, smoothing, scaling, and visualization.
             The output includes preprocessed CSV files and visual plots.
"""

from datetime import datetime
from rich.console import Console
import os
import pandas as pd
import glob
import argparse
import matplotlib.pyplot as plt
from typing import Optional

# Constants
CONFIG = {
    "RESAMPLE_RATE": "1s",
    "MIN_VALUE": -200,
    "MAX_VALUE": 200,
    "FACTOR": 1000,
    "BEFORE": 20,
    "AFTER": 20
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


def load_and_merge_data(data_dir: str, files: list) -> pd.DataFrame:
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
        df = df.resample(CONFIG["RESAMPLE_RATE"]).mean().interpolate()
        list_df.append(df)

    # Merge all DataFrames on 'datetime' by concatenating along columns
    combined_df = pd.concat(list_df, axis=1).reset_index()
    combined_df['datetime'] = pd.to_datetime(combined_df['datetime'], format="%Y-%m-%d %H:%M:%S")
    combined_df.set_index("datetime", inplace=True, drop=False)

    # Write precomputed data to file
    path = get_precomputed_path(data_dir, "precomputed_experiments.csv")
    combined_df.to_csv(path, index=True)

    return combined_df


def load_times(file: str) -> pd.DataFrame:

    df = pd.read_csv(file)
    print(df.head())
    df["times"] = pd.to_datetime(df['times'], format="%Y-%m-%d %H:%M:%S")
    print(df.head())

    return df


def min_max_normalization(df: pd.DataFrame, column: str) -> None:

    df[f"{column}_scaled"] = ((df[column] - CONFIG["MIN_VALUE"]) / (
        CONFIG["MAX_VALUE"] - CONFIG["MIN_VALUE"]) * CONFIG["FACTOR"]
    )


def adjusted_min_max_normalization(df: pd.DataFrame, column: str) -> None:

    df[f"{column}_scaled"] = (df[column] - (CONFIG["MIN_VALUE"]/CONFIG["FACTOR"])) / (
        (CONFIG["MAX_VALUE"]/CONFIG["FACTOR"]) - (CONFIG["MIN_VALUE"]/CONFIG["FACTOR"]))


def extract_important_data(df_phyto: pd.DataFrame, df_times: pd.DataFrame, ) -> pd.DataFrame:
    """
    Important data equals x min before and x min after stimulus
    """

    rows = []
    for start_time in df_times['times'][1:]:
        start_time = start_time - pd.Timedelta(minutes=10)
        end_time = start_time + pd.Timedelta(minutes=10)

        subset = df_phyto.loc[start_time:end_time]

        rows.append({
        'time': start_time,
        'differential_potential_pn1': subset['differential_potential_pn1'].tolist(),
        'differential_potential_pn3': subset['differential_potential_pn3'].tolist()
    })
        
    result_df = pd.DataFrame(rows)
    
    return result_df


def plot_data(df_phyto: pd.DataFrame, df_times: pd.DataFrame) -> None:

    # Create subplots
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 8), sharex=True)

    # Plot each column in a separate subplot
    # relevant sind differential_potential_pn1 (leaf), differnetial_potential_pn3 (stem), O3_1 und O3_2 die zwei ozon
    # sensoren (aber ich weiß nicht welcher oben und welcher unten ist)
    columns = ["differential_potential_pn1", "differential_potential_pn3", "O3_1", "O3_2"]
    titles = ["Differential Potential Leaf", "Differential Potential Stem", "O3_1", "O3_2"]

    for i, col in enumerate(columns):
        axes[i].plot(df_phyto["datetime"], df_phyto[col], label=col, linewidth=1)
        axes[i].set_title(titles[i])
        axes[i].grid(True)
        axes[i].legend()

        # Loop through each event in df_times to mark the area around each event
        for start_time in df_times['times']:
            axes[i].axvline(start_time, color='blue', linestyle='--', linewidth=1.5)
            axes[i].axvspan(start_time - pd.Timedelta(minutes=CONFIG["BEFORE"]),
                            start_time + pd.Timedelta(minutes=CONFIG["AFTER"]),
                            color='blue', alpha=0.2)

    # Set common x-label
    plt.xlabel("Datetime")
    plt.xticks(rotation=45)

    # Adjust layout for better spacing
    plt.tight_layout()
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

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Preprocess CSV files.")
    parser.add_argument("--data-dir", required=True, help="Directory with raw files.")
    args = parser.parse_args()

    # Normalize and validate inputs
    data_dir = args.data_dir

    # Print Input Parameters
    console.print(f"[bold green]Data Directory:[/bold green] {data_dir}")

    # Process Phyto Node 
    df_phyto = None
    path = get_precomputed_path(data_dir, "precomputed_experiments.csv")
    if os.path.exists(path):
        df_phyto = pd.read_csv(path)
        df_phyto['datetime'] = pd.to_datetime(df_phyto['datetime'], format="%Y-%m-%d %H:%M:%S")
        df_phyto.set_index("datetime", inplace=True, drop=False)
    else:
        phyto_files = discover_files(data_dir, "experiment")
        df_phyto = load_and_merge_data(data_dir, phyto_files)

    # Read Ozone times file
    times_files = discover_files(data_dir, "times")
    df_times = load_times(times_files[0])

    extract_important_data(df_phyto, df_times)

    plot_data(df_phyto, df_times)





if __name__ == "__main__":
    main()
