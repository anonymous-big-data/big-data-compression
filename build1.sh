#!/bin/bash

#SBATCH --cpus-per-task=24
#SBATCH --export=ALL
#SBATCH --job-name="fusion"
#SBATCH --mail-type=begin  # Email when the job starts
#SBATCH --mail-type=end    # Email when the job finishes
#SBATCH --mail-user=jamalids@mcmaster.ca
#SBATCH --nodes=1
#SBATCH --output="fusion.%j.%N.out"
#SBATCH --constraint=rome
#SBATCH --mem=254000M
#SBATCH -t 47:59:00

module load StdEnv/2023
module load gcc/13.3
module load cmake/3.27.7

# Define directories





DATASET_DIR="/home/Fcbench-dataset/64/64/r1"
RESULTS_DIR="/home/Documents/results2"
EXECUTABLE="./build/external_tools/parallel-test"


mkdir -p "$RESULTS_DIR"

# Build the program
mkdir -p build
cd build
/home/programs/cmake-3.31.0-rc3-linux-x86_64/bin/cmake -DCMAKE_BUILD_TYPE=Release ..
make -j24  # Compile using 24 cores
cd ..

# Check if executable was built
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: Executable not found: $EXECUTABLE"

    exit 1
fi

echo "=====> Starting Dataset Processing <====="

# Process ALL dataset files
for dataset in "$DATASET_DIR"/*; do
    if [ -f "$dataset" ]; then
        dataset_name=$(basename "$dataset")  # Extract dataset filename
        dataset_name="${dataset_name%.*}"  # Remove file extension
        result_file="$RESULTS_DIR/${dataset_name}.csv"  # Save results in CSV format

        echo "Processing dataset: $dataset"
        start_time=$(date +%s.%N)

        "$EXECUTABLE" --dataset "$dataset" --outcsv "$result_file" --threads 16 --bits 64 --method=fastlz

        end_time=$(date +%s.%N)
        elapsed_time=$(echo "$end_time - $start_time" | bc)


        echo "Dataset: $dataset" >> "$result_file"
        echo "Processing Time: $elapsed_time sec" >> "$result_file"

        echo "Results saved in: $result_file"

    fi
done

echo "=====> All Datasets Processed Successfully <====="
