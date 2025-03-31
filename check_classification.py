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
    "DATABITS": 8388608,
    "VREF": 2.5,
    "GAIN": 4.0,
    "WINDOW_SIZE": 5,
    "RESAMPLE_RATE": "1s",
    "MIN_VALUE": -0.2,
    "MAX_VALUE": 0.2,
}

# Initialize the console
console = Console()

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

def main():
    parser = argparse.ArgumentParser(description="Preprocess CSV files.")
    parser.add_argument("--data_dir", required=True, help="Directory with raw files.")
    parser.add_argument("--prefix", required=True, help="C1")
    parser.add_argument("--from_date", required=True, help="Cutoff date (YYYY-MM-DD). Cut off data before that date.")
    parser.add_argument("--until_date", required=True, help="Cutoff date (YYYY-MM-DD). Cut off data after that date.")
    args = parser.parse_args()

    data_dir = args.data_dir
    prefix = args.prefix.upper()
    from_date = validate_date(args.from_date)
    until_date = validate_date(args.until_date)
    
    ozone_files = discover_files(data_dir, "Ozone")
    df_ozone = load_and_combine_csv(ozone_files)

    classified_files = discover_files(data_dir, prefix)
    df_classified = load_and_combine_csv(ozone_files)



if __name__ == "__main__":
    main()
