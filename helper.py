import pandas as pd
from pathlib import Path
from datetime import timedelta
import sys

def process_csvs(input_dir):
    input_path = Path(input_dir)
    output_path = input_path / "out"
    output_path.mkdir(exist_ok=True)

    csv_files = input_path.glob("Ozone*.csv")
    timestamp_col = "timestamp"

    for file in csv_files:
        print(f"Processing {file.name}...")
        df = pd.read_csv(file)

        if timestamp_col not in df.columns:
            print(f"Skipping {file.name}: no '{timestamp_col}' column.")
            continue

        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
        df[timestamp_col] += timedelta(minutes=10)

        out_file = output_path / file.name
        df.to_csv(out_file, index=False)

    print(f"All files processed. Output saved to: {output_path.resolve()}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python shift_timestamps.py <data_directory>")
    else:
        process_csvs(sys.argv[1])


