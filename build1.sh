#!/bin/bash
###############################################################################
#  fusion.slurm
#  ------------
#  Batch script for SLURM that:
#    1. Loads required modules (GCC, CMake).
#    2. Builds the project in ./build (Release mode, 24 cores).
#    3. Runs `parallel-test` on every dataset file in DATASET_DIR.
#    4. Stores per-dataset CSV results in RESULTS_DIR with runtime appended.
#
#  Submit with:  sbatch fusion.slurm
###############################################################################

# =======================
# ===  SLURM SETTINGS ===
# =======================
#SBATCH --job-name="fusion"          # Job name visible in squeue
#SBATCH --nodes=1                    # Allocate a single node
#SBATCH --cpus-per-task=24           # 24 CPU threads (for make -j and runtime)
#SBATCH --mem=254000M                # ~254 GB RAM
#SBATCH --constraint=rome            # Prefer AMD Rome nodes (adjust/remove if not needed)
#SBATCH --time=47:59:00              # Wall-time limit hh:mm:ss
#SBATCH --output="fusion.%j.%N.out"  # Stdout filename (%j = job-ID)
#SBATCH --export=ALL                 # Export current env vars to job

# E-mail notifications
#SBATCH --mail-user=
#SBATCH --mail-type=begin            # Send email when job starts
#SBATCH --mail-type=end              # Send email when job ends

# =======================
# ===  MODULE LOADING ===
# =======================
module load StdEnv/2023
module load gcc/13.3
module load cmake/3.27.7
# (Add other `module load` lines here if your program needs them)

# ------------------------------------------------------------------------------
# USER-ADJUSTABLE PATHS
# ------------------------------------------------------------------------------

# DATASET_DIR — absolute path to the folder that stores the input data files
#               Every regular file found here will be fed to `parallel-test`.
DATASET_DIR=""

# RESULTS_DIR — destination directory for the per-dataset CSV outputs produced
#               by the executable.  Will be created if it does not exist.
RESULTS_DIR=""

# CONFIG_FILE — CSV table that defines the clustering.
CONFIG_FILE="./all.csv"

# EXECUTABLE  — relative path to the compiled binary that performs the actual
#               compression and measurement.  Built in the block above.
EXECUTABLE="./build/external_tools/parallel-test"


mkdir -p "$RESULTS_DIR"                    # Ensure results folder exists

# =======================
# ===  BUILD PROJECT  ===
# =======================
mkdir -p build
cd build
# Use a newer standalone CMake binary (adjust path if different)
~/programs/cmake-3.31.0-rc3-linux-x86_64/bin/cmake -DCMAKE_BUILD_TYPE=Release ..
make -j24                                    # Compile with 24 threads
cd ..

# =====  Sanity checks  =====
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable not found: $EXECUTABLE"
    exit 1
fi
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "=====> Starting Dataset Processing <====="

# =======================
# ===  MAIN LOOP      ===
# =======================
for dataset in "$DATASET_DIR"/*; do
    # Skip if entry is not a regular file
    if [ -f "$dataset" ]; then
        # Strip directory and extension for a clean result filename
        base=$(basename "$dataset")
        dataset_name="${base%.*}"
        result_file="$RESULTS_DIR/${dataset_name}.csv"

        echo "Processing dataset: $dataset"
        start_time=$(date +%s.%N)

        # -----------------------
        #  CALL THE EXECUTABLE
        # -----------------------
        "$EXECUTABLE" \
            --dataset "$dataset" \
            --outcsv  "$result_file" \
            --threads 16 \
            --bits    32 \
            --method  zlib \
            --config  "$CONFIG_FILE"

        end_time=$(date +%s.%N)
        elapsed=$(echo "$end_time - $start_time" | bc)

        # Append timing info to the same CSV
        {
            echo "Dataset: $dataset"
            echo "Processing Time: $elapsed sec"
        } >> "$result_file"

        echo "Results saved in: $result_file"
    fi
done

echo "=====> All Datasets Processed Successfully <====="
