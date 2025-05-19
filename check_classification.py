from datetime import datetime
from rich.console import Console
import os
import pandas as pd
import numpy as np
import glob
import argparse
import matplotlib
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional
from scipy.interpolate import make_interp_spline

# Use the PGF backend
# matplotlib.use("pgf")

# # Update rcParams
# plt.rcParams.update({
#     "pgf.texsystem": "xelatex",  # Use XeLaTeX
#     "font.family": "sans-serif",  # Use a sans-serif font
#     "font.sans-serif": ["Arial"],  # Specifically use Arial
#     "font.size": 10,  # Set the font size
#     "text.usetex": True,  # Use LaTeX for text rendering
#     "pgf.rcfonts": False,  # Do not override Matplotlib's rc settings
# })

# Constants

CONFIG = {
    "WINDOW_SIZE": 5,
    "RESAMPLE_RATE": "1s",
    "MIN_VALUE": -0.2,
    "MAX_VALUE": 0.2,
    "AFTER": 30, # minutes after stimulus
}

# Initialize the console
console = Console()

def get_precomputed_path(data_dir: str, name: str):

    return os.path.join(data_dir, name)

def validate_date(date_str: str) -> str:
    try:
        # Parse the input string to a datetime object
        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
        # Return the formatted datetime string
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        raise ValueError("Invalid datetime format. Use 'YYYY-MM-DD HH:MM'.")


def discover_files(data_dir: str, prefix: str) -> list[str]:
    console.print(f"[bold cyan]Discovering files with prefix '{prefix}' in '{data_dir}'[/bold cyan]")
    files = glob.glob(os.path.join(data_dir, f"{prefix}*.csv"))
    console.print(f"Found [bold yellow]{len(files)}[/bold yellow] matching files.")
    return files

def load_and_combine_csv(files: list[str]) -> pd.DataFrame:
    console.print("[bold green]Loading and combining CSV files into a single DataFrame...[/bold green]")
    return pd.concat([pd.read_csv(file) for file in files], ignore_index=True)

def cut_data(df: pd.DataFrame, from_date: str, until_date: str) -> pd.DataFrame:
 
    console.print("[bold yellow]Preprocessing data: filtering and resampling...[/bold yellow]")

    # Add dates to CONFIG
    CONFIG["FROM_DATE"] = from_date
    CONFIG["UNTIL_DATE"] = until_date

    df = df.dropna(subset=["datetime"])
    df = df[(df["datetime"] >= from_date) & (df["datetime"] < until_date)]
    return df

def z_score(arr: np.ndarray, factor: float = 1.0, mean_val: Optional[float] = None, std_val: Optional[float] = None) -> np.ndarray:
    """
    Applies z-score normalization to the array and scales it by the given factor.
    """
    arr = np.array(arr, dtype=np.float32)
    
    if mean_val is None:
        mean_val = np.mean(arr)
    if std_val is None:
        std_val = np.std(arr)
    
    if std_val == 0:
        return np.zeros_like(arr)
    
    return ((arr - mean_val) / std_val) * factor

def adjusted_min_max(data_slice: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """
    Apply Min-Max normalization to a given data slice.

    The normalization transforms the data into the range [0, factor].

    Args:
        data_slice (np.ndarray): The input array to normalize.
        factor (float): Scaling factor to adjust the normalized values (default is 1.0).

    Returns:
        np.ndarray: The min-max normalized array.
    """
    min_val = -0.2
    max_val = +0.2
    range_val = (max_val/factor) - (min_val/factor)

    if range_val == 0:
        return np.zeros_like(data_slice)  # Avoid division by zero if all values are the same

    normalized = (data_slice - (min_val/factor)) / range_val
    return normalized

def normalize_input(df: pd.DataFrame, normalization: str) -> pd.DataFrame:
    # Convert stringified lists to numpy arrays
    df['VoltagesCh0'] = df['VoltagesCh0NotScaled'].apply(lambda x: np.array(eval(x)))
    df['VoltagesCh1'] = df['VoltagesCh1NotScaled'].apply(lambda x: np.array(eval(x)))

    if normalization == "z-score":
        df['input_normalized_ch0'] = df['VoltagesCh0'].apply(lambda x: z_score(x,1000))
        df['input_normalized_ch1'] = df['VoltagesCh1'].apply(lambda x: z_score(x,1000))

    if normalization == "adjusted-min-max":
        df['input_normalized_ch0'] = df['VoltagesCh0'].apply(lambda x: adjusted_min_max(x,1000))
        df['input_normalized_ch1'] = df['VoltagesCh1'].apply(lambda x: adjusted_min_max(x,1000))

     # Drop unnecessary columns
    df.drop(columns=['VoltagesCh0NotScaled', 'VoltagesCh1NotScaled'], inplace=True)

    return df

def extract_classification(df: pd.DataFrame) -> pd.DataFrame:

    df['Ch0'] = df['ClassificationCh0'].apply(lambda x: np.array(eval(x)))
    df['Ch1'] = df['ClassificationCh1'].apply(lambda x: np.array(eval(x)))

    # Extract values from Ch0
    df['classification_ch0_idle'] = df['Ch0'].apply(lambda x: x[0])
    #df['classification_ch0_heat'] = df['Ch0'].apply(lambda x: x[1])
    df['classification_ch0_ozone'] = df['Ch0'].apply(lambda x: x[1])

    # Extract values from Ch1
    df['classification_ch1_idle'] = df['Ch1'].apply(lambda x: x[0])
    #df['classification_ch1_heat'] = df['Ch1'].apply(lambda x: x[1])
    df['classification_ch1_ozone'] = df['Ch1'].apply(lambda x: x[1])

    # Drop unnecessary columns
    df.drop(columns=['Ch0', 'Ch1', 'ClassificationCh0', 'ClassificationCh1'], inplace=True)

    return df

def smooth_classification(df: pd.DataFrame, window_size: int) -> pd.DataFrame:


    df["ch0_smoothed_idle"] = df["classification_ch0_idle"].rolling(window=window_size, min_periods=1).mean()
    #df["ch0_smoothed_heat"] = df["classification_ch0_heat"].rolling(window=window_size, min_periods=1).mean()
    df["ch0_smoothed_ozone"] = df["classification_ch0_ozone"].rolling(window=window_size, min_periods=1).mean()
    df["ch1_smoothed_idle"] = df["classification_ch1_idle"].rolling(window=window_size, min_periods=1).mean()
    #df["ch1_smoothed_heat"] = df["classification_ch1_heat"].rolling(window=window_size, min_periods=1).mean()
    df["ch1_smoothed_ozone"] = df["classification_ch1_ozone"].rolling(window=window_size, min_periods=1).mean()

    df.drop(columns=[
        'classification_ch0_idle',
        #'classification_ch0_heat',
        'classification_ch0_ozone',
        'classification_ch1_idle',
        #'classification_ch1_heat',
        'classification_ch1_ozone'], inplace=True)

    return df

def load_times(file: str) -> pd.DataFrame:

    df = pd.read_csv(file)
    df["times"] = pd.to_datetime(df['times'], format="%Y-%m-%d %H:%M:%S")

    return df

def label_ground_truth(df_phyto: pd.DataFrame, df_times: pd.DataFrame) -> pd.DataFrame:
    # Ensure ground_truth column exists and is initialized to 0
    df_phyto['ground_truth'] = 0

    for stimulus_time in df_times['times']:
        start = pd.to_datetime(stimulus_time)
        end = start + pd.Timedelta(minutes=CONFIG["AFTER"])
        mask = (df_phyto['datetime'] >= start) & (df_phyto['datetime'] <= end)
        df_phyto.loc[mask, 'ground_truth'] = 1

    return df_phyto

def make_ready_for_classification(df: pd.DataFrame, data_dir):

    df['input_not_normalized_ch0'] = df['VoltagesCh0NotScaled'].apply(lambda x: "[" + ",".join(map(str, eval(x))) + "]")
    df['input_not_normalized_ch1'] = df['VoltagesCh1NotScaled'].apply(lambda x: "[" + ",".join(map(str, eval(x))) + "]")

    df.drop(columns=[
        'VoltagesCh0NotScaled', 'ClassificationCh0',
        'VoltagesCh1NotScaled', 'ClassificationCh1'], inplace=True)
    
    df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
    df = df.reset_index(drop=True)
    
    preprocessed_folder = os.path.join(data_dir, "simulation")
    os.makedirs(preprocessed_folder, exist_ok=True)
    final_path = get_precomputed_path(preprocessed_folder, f"simulation_data.csv")
    df.to_csv(final_path, index=True)


def plot_data(df_classified: pd.DataFrame, df_ozone: pd.DataFrame, threshold: float) -> None:
    
    plant_id=3
    #------------------Prepare Data for Plot---------------------------------------#
    window_size = 10 # 100 = 10min
    df_classified['LastVoltageCh0'] = df_classified['input_normalized_ch0'].apply(lambda x: x[-1])
    df_classified['LastVoltageCh1'] = df_classified['input_normalized_ch1'].apply(lambda x: x[-1])
    df_classified["LastVoltageCh0"] = df_classified["LastVoltageCh0"].rolling(window=window_size, min_periods=1).mean()
    df_classified["LastVoltageCh1"] = df_classified["LastVoltageCh1"].rolling(window=window_size, min_periods=1).mean()

    fig_width = 5.90666  # Width in inches
    aspect_ratio = 0.618  # Example aspect ratio (height/width)
    fig_height = fig_width * aspect_ratio


    fig, axs = plt.subplots(3, 1, figsize=(fig_width, 8), sharex=True)

    time_fmt = mdates.DateFormatter('%H:%M')

    for ax in axs:
        ax.grid(True, linestyle='dashed', linewidth=0.5, alpha=0.6)
        ax.xaxis.set_major_formatter(time_fmt)  # Format x-axis as hours
        ax.tick_params(axis='x', labelsize=10)  # Set font size to 10
        plt.setp(ax.get_xticklabels(), fontsize=10, rotation=0, ha='center')


    #df_classified["smoothed_heat_mean"] = (df_classified["ch0_smoothed_heat"] + df_classified["ch1_smoothed_heat"])/2
    df_classified["smoothed_ozone_mean"] = (df_classified["ch0_smoothed_ozone"] + df_classified["ch1_smoothed_ozone"])/2
    df_classified["smoothed_idle_mean"] = (df_classified["ch0_smoothed_idle"] + df_classified["ch1_smoothed_idle"])/2

    df_classified["smoothed_ozone_min"] = (
    df_classified[["ch0_smoothed_ozone", "ch1_smoothed_ozone"]]
    .min(axis=1))

    axs[0].axhline(y=threshold, color="black", linestyle="--", linewidth=1, label=f"Threshold: {threshold}")

    axs[0].fill_between(
        df_classified['datetime'], 0, 1.0, 
        where=(df_classified["ground_truth"] == 1), 
        color='#4169E1', alpha=0.3, label="Stimulus application"
    )

    # CH0: blues
    #axs[0].plot(df_classified['datetime'], df_classified["ch0_smoothed_idle"], label="Idle CH0", color="#add8e6")   # lightblue
   # axs[0].plot(df_classified['datetime'], df_classified["smoothed_heat_mean"], label="Heat CH0", color="#FF0000")  # matplotlib default blue
    #axs[0].plot(df_classified['datetime'], df_classified["ch0_smoothed_ozone"], label="Ozone CH0", color="#007BFF") # darkblue
    #axs[0].plot(df_classified['datetime'], df_classified["ch1_smoothed_ozone"], label="Ozone CH1", color="#012169")
    axs[0].plot(df_classified['datetime'], df_classified["smoothed_ozone_min"], label="Min of CH0 and CH1 Classification", color="#004EB4") 

    # CH1: oranges
    #axs[0].plot(df_classified['datetime'], df_classified["ch1_smoothed_idle"], label="Idle CH1", color="#ffdab9")   # peachpuff (light orange)
    # axs[0].plot(df_classified['datetime'], df_classified["ch1_smoothed_heat_mean"], label="Heat CH1", color="#8B0000")   # matplotlib default orange
    # axs[0].plot(df_classified['datetime'], df_classified["ch1_smoothed_ozone"], label="Ozone CH1", color="#00008B") # dark orange/brown



   

    axs[0].fill_between(df_classified['datetime'], 0, 1.0, 
                    where=(df_classified["ch0_smoothed_ozone"] > threshold),# & (df_classified["ch1_smoothed_heat"] > threshold), 
                    color='#000080', alpha=0.3, label="Stimulus prediction")
    



    # Ensure y-axis limits and set explicit tick marks
    axs[0].set_ylim(0, 1.05)
    axs[0].set_yticks([0, 0.25, 0.5, 0.75, 1])  # Explicitly set y-ticks
    axs[0].set_ylabel("Phase Probability",fontsize=10)
    axs[0].tick_params(axis='y', labelsize=10) 

    axs[0].set_title(f"Online Ozone Phase Classification with PhytoNodeClassifier (Plant ID: {plant_id})", fontsize=10, pad=40)
    axs[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2, framealpha=0.7)


    #Line plot for interpolated electric potential
    #axs[1].set_title(f"Feature-Scaled FCN Input via Adjusted-Min-Max Normalization", fontsize=10, pad=40)
    axs[1].plot(df_classified['datetime'], df_classified['LastVoltageCh0'], label="CH0", color="#90EE90")
    axs[1].plot(df_classified['datetime'], df_classified['LastVoltageCh1'], label="CH1", color="#013220")

    # Labels and Titles
    axs[1].tick_params(axis='y', labelsize=10)
    axs[1].set_ylabel("EDP [scaled]",fontsize=10)
    axs[1].set_title("Feature-Scaled FCN Input via Adjusted-Min-Max Normalization",fontsize=10)
    axs[1].legend(fontsize=8, loc="center right")

    axs[2].plot(df_ozone['datetime'], df_ozone['O3_2'])
    axs[2].set_ylabel('O3 [ppb]', fontsize=10)
    axs[2].set_title('Ozone Data', fontsize=10)
    axs[2].grid(True)

    # Improve spacing to prevent label cutoff
    fig.tight_layout()

    # Save figure in PGF format with proper bounding box
    plt.savefig(f"Ozone.pgf", format="pgf", bbox_inches="tight", pad_inches=0.05)
    #plot_path = os.path.join(save_dir, f"{prefix}_classified_plot.png")
    #plt.savefig(plot_path, dpi=300)
    #plt.show()

def plot_o3(df):
    df['datetime'] = pd.to_datetime(df['datetime'])  # Ensure datetime is parsed

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axs[0].plot(df['datetime'], df['O3_1'], marker='o')
    axs[0].set_ylabel('O3_1')
    axs[0].set_title('O3_1 over Time')
    axs[0].grid(True)

    axs[1].plot(df['datetime'], df['O3_2'], marker='x', color='orange')
    axs[1].set_ylabel('O3_2')
    axs[1].set_title('O3_2 over Time')
    axs[1].set_xlabel('Time')
    axs[1].grid(True)

    plt.tight_layout()

def main():
    parser = argparse.ArgumentParser(description="Preprocess CSV files.")
    parser.add_argument("--data-dir", required=True, help="Directory with raw files.")
    parser.add_argument("--prefix", required=True, help="C1")
    parser.add_argument("--from-date", required=True, help="Cutoff date (YYYY-MM-DD). Cut off data before that date.")
    parser.add_argument("--normalization", required=True, type=str, help="Normalization method.")
    parser.add_argument("--until-date", required=True, help="Cutoff date (YYYY-MM-DD). Cut off data after that date.")
    parser.add_argument("--threshold", type=float, required=True, help="Threshold has to be exceed to classify as ozone phase")
    args = parser.parse_args()

    data_dir = args.data_dir
    prefix = args.prefix.upper()
    threshold = args.threshold
    normalization = args.normalization.lower()
    from_date = validate_date(args.from_date)
    until_date = validate_date(args.until_date)
    
    ozone_files = discover_files(data_dir, "Ozone")
    df_ozone = load_and_combine_csv(ozone_files)
    df_ozone.rename(columns={'timestamp': 'datetime'}, inplace=True)
    df_ozone['datetime'] = pd.to_datetime(df_ozone['datetime'])
    df_ozone = df_ozone.sort_values(by="datetime").reset_index(drop=True)
    df_ozone = cut_data(df_ozone, from_date, until_date)

    plot_o3(df_ozone)

    times_files = discover_files(data_dir, "times")
    df_times = load_times(times_files[0])


    classified_files = discover_files(data_dir, prefix)
    df_classified = load_and_combine_csv(classified_files)
    df_classified.rename(columns={'Datetime': 'datetime'}, inplace=True)
    df_classified['datetime'] = pd.to_datetime(df_classified['datetime'], format='%Y-%m-%d %H:%M:%S:%f', errors='coerce')
    df_classified = df_classified.sort_values(by="datetime").reset_index(drop=True)
    df_classified = cut_data(df_classified, from_date, until_date)
    df_classified = label_ground_truth(df_classified, df_times)
    df_copy = df_classified.copy()
    make_ready_for_classification(df_copy, data_dir)
    # 
    df_classified = extract_classification(df_classified)
    df_classified = normalize_input(df_classified, normalization)
    df_classified = smooth_classification(df_classified, 10)

    df_classified = label_ground_truth(df_classified, df_times)

    plot_data(df_classified, df_ozone, threshold)



if __name__ == "__main__":
    main()
