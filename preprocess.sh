#!/bin/bash

# Define an array of data directories
DATA_DIRS=(
    "/home/chris/experiment_data/ozone_cut/ozone_cut/Exp44_Ivy2"
    "/home/chris/experiment_data/ozone_cut/ozone_cut/Exp45_Ivy4"
    "/home/chris/experiment_data/ozone_cut/ozone_cut/Exp46_Ivy0"
    "/home/chris/experiment_data/ozone_cut/ozone_cut/Exp47_Ivy5"
   
)

# Define an array of normalization options (empty string for no normalization)
NORMALIZATIONS=(
    ""
    "min-max"
    "adjusted-min-max"
    "min-max-chunk"
    "z-score-chunk"
    "z-score"
)

# Loop over each data directory
for DATA_DIR in "${DATA_DIRS[@]}"; do
    echo "Processing data directory: $DATA_DIR"
    
    # Loop over each normalization option
    for NORM in "${NORMALIZATIONS[@]}"; do
        if [ -z "$NORM" ]; then
            echo "Running preprocess with no normalization..."
            python3 preprocess.py --data-dir "$DATA_DIR"
        else
            echo "Running preprocess with normalization: $NORM..."
            python3 preprocess.py --data-dir "$DATA_DIR" --normalization "$NORM"
        fi
    done
done