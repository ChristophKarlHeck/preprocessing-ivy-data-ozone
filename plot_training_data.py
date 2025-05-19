import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import ast
import glob
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

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


def extract_data():
    # your exact pattern:
    pattern = '/home/chris/experiment_data/ozone_cut/ozone_cut/*/important_data.csv'

    # find all matching files
    file_paths = glob.glob(pattern)
    print(f"Found {len(file_paths)} files:")
    for p in file_paths:
        print(" ", p)

    # read each into a DataFrame
    df_list = [pd.read_csv(fp) for fp in file_paths]

    # concatenate into one big DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)

    all_segments = []
    for _, row in combined_df.iterrows():
        # 1) turn the string "[...]" into a Python list
        list_pn1 = ast.literal_eval(row['differential_potential_pn1'])
        # 2) cast to a NumPy array
        arr_pn1  = np.array(list_pn1, dtype=float)
        print(len(arr_pn1))
        all_segments.append(arr_pn1)

        list_pn3 = ast.literal_eval(row['differential_potential_pn3'])
        arr_pn3  = np.array(list_pn3, dtype=float)
        all_segments.append(arr_pn3)





    return all_segments #segments_ch1, segments_ch2


if __name__ == "__main__":
    # Load configuration from the JSON file

    all_segments_both = []


    all_segments_both = extract_data()


    # Ensure each segment has the same length and set fixed length to 3600 (600*6)
    max_length = 3602
    time_axis = np.linspace(-30, 30, max_length)
    
    # x_values_ch1 = []
    # y_values_ch1 = []
    # x_values_ch2 = []
    # y_values_ch2 = []

    x_values_both = []
    y_values_both = []

    # for not average indent 2 spaces 
    
    # for segments in all_segments_ch1:
    #     for segment in segments:
    #         if len(segment) > max_length:
    #             segment = segment[:max_length]  # Trim longer segments
    #         time_subset = time_axis[:len(segment)]
    #         x_values_ch1.extend(time_subset)
    #         y_values_ch1.extend(segment)
    
    # for segments in all_segments_ch2:
    #     for segment in segments:
    #         if len(segment) > max_length:
    #             segment = segment[:max_length]  # Trim longer segments
    #         time_subset = time_axis[:len(segment)]
    #         x_values_ch2.extend(time_subset)
    #         y_values_ch2.extend(segment)


    for segment in all_segments_both:
        if len(segment) > max_length:
            segment = segment[:max_length]  # Trim longer segments
        time_subset = time_axis[:len(segment)]
        x_values_both.extend(time_subset)
        y_values_both.extend(segment)
    
    # x_values_ch1 = np.array(x_values_ch1)
    # y_values_ch1 = np.array(y_values_ch1)
    # x_values_ch2 = np.array(x_values_ch2)
    # y_values_ch2 = np.array(y_values_ch2)


    plt.figure(figsize=(3,2))

    # Create a hexbin plot using the combined data
    hb = plt.hexbin(x_values_both, y_values_both, gridsize=50, cmap='Blues', mincnt=1)

    plt.xlabel("Minutes", fontsize=10)
    plt.ylabel("EDP [scaled]", fontsize=10)

    # Add a vertical line at time zero to indicate the start of Heating
    plt.axvline(0, color='black', linestyle='--')

    #plt.legend(loc="lower left", fontsize=8)
    #plt.colorbar(hb)
    plt.tight_layout()
    #plt.show()
    plt.savefig("heatMapMMOzone.pgf", format="pgf", bbox_inches="tight", pad_inches=0.05)

    
    # # Create subplots for CH1 and CH2
    # fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    # # yticks = [-3000, -2000, -1000, 0, 1000, 2000, 3000]
    # # ytick_labels = ["-3000", "-2000", "-1000", "0", "1000", "2000", "3000"]
    
    # axes[0].hexbin(x_values_ch1, y_values_ch1, gridsize=50, cmap='Reds', mincnt=1)
    # axes[0].set_title("Training Data Ch0")
    # axes[0].set_xlabel("Time (minutes relative to Increasing start)")
    # axes[0].set_ylabel("Scaled Electrical Potential (mV)")
    # # axes[0].set_ylim([-3000, 3000])
    # #axes[0].set_yticks(yticks)
    # #axes[0].set_yticklabels(ytick_labels)
    # axes[0].axvline(0, color='blue', linestyle='--', label="Start of Increasing")
    # axes[0].legend()
    
    # axes[1].hexbin(x_values_ch2, y_values_ch2, gridsize=50, cmap='Blues', mincnt=1)
    # axes[1].set_title("Training Data Ch1")
    # axes[1].set_xlabel("Time (minutes relative to Increasing start)")
    # axes[1].set_ylabel("Scaled Electrical Potential (mV)")
    # # axes[1].set_ylim([-3000, 3000])
    # #axes[1].set_yticks(yticks)
    # #axes[1].set_yticklabels(ytick_labels)
    # axes[1].axvline(0, color='blue', linestyle='--', label="Start of Increasing")
    # axes[1].legend()
    
    # plt.colorbar(axes[0].collections[0], ax=axes[0], label="Density of Measurements")
    # plt.colorbar(axes[1].collections[0], ax=axes[1], label="Density of Measurements")
    
    # plt.tight_layout()
    # plt.show()