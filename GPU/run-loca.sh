#!/usr/bin/env bash
set -euo pipefail

# ---- CONFIG (edit if your paths differ) ----
DATASET_DIR="
EXECUTABLE="./nvcomp_gds"
RESULTS_DIR="./results1"

# CUDA runtime (match toolkit 12.3) + avoid PTX JIT (your driver is 12.2)
export LD_LIBRARY_PATH="/usr/local/cuda-12.3/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export CUDA_DISABLE_PTX_JIT=1
# Select codec via env var only (no CLI arg). Options: gdeflate | lz4 | zstd
export NVCOMP_CODEC="${NVCOMP_CODEC:-gdeflate}"

# Optional: limit OpenMP threads if you want
export OMP_NUM_THREADS="$(nproc)"

# ---- Sanity checks ----
if [[ ! -x "$EXECUTABLE" ]]; then
  echo "ERROR: Executable not found or not executable: $EXECUTABLE" >&2
  exit 1
fi
if [[ ! -d "$DATASET_DIR" ]]; then
  echo "ERROR: Dataset dir not found: $DATASET_DIR" >&2
  exit 1
fi
mkdir -p "$RESULTS_DIR"

echo "=====> Starting nvcomp_gds execution for all datasets <====="
echo "Using EXECUTABLE : $EXECUTABLE"
echo "Using DATASET_DIR: $DATASET_DIR"
echo "NVCOMP_CODEC     : $NVCOMP_CODEC"
echo "Logs will go to  : $RESULTS_DIR"
echo

shopt -s nullglob
tsv_list=("$DATASET_DIR"/*.tsv)
if (( ${#tsv_list[@]} == 0 )); then
  echo "No .tsv files found in $DATASET_DIR" >&2
  exit 1
fi

for dataset in "${tsv_list[@]}"; do
  dataset_name="$(basename "$dataset" .tsv)"
  log_file="$RESULTS_DIR/${dataset_name}_run.log"
  echo "Processing: $dataset"
  {
    echo "Dataset: $dataset"
    start_time=$(date +%s.%N)

    # Run the program: <file_or_folder> <32|64>
    "$EXECUTABLE" "$dataset" 64

    end_time=$(date +%s.%N)
    elapsed_time=$(echo "$end_time - $start_time" | bc -l)
    printf "Execution Time: %s seconds\n" "$elapsed_time"
  } | tee "$log_file"

  echo "Results saved in: $log_file"
  echo
done

echo "=====> All datasets processed successfully <====="
####NVCOMP_CODEC=lz4 ./run_local.sh
    #

    #NVCOMP_CODEC=zstd ./run_local.sh
