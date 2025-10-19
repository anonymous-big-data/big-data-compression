
===============================================
 Parallel Compression Benchmarking Project
===============================================

This project benchmarks parallel compression using multiple algorithms: 
Zstd, LZ4, Snappy, Bzip2, Zlib, and FastLZ using C++17 and OpenMP.

-----------------------------------------------
 REQUIRED SYSTEM TOOLS AND LIBRARIES
-----------------------------------------------

You MUST install or load the following tools before building the project:

SYSTEM TOOLS:
- GCC (tested with 13.3)
- CMake (tested with 3.27 or newer)
- Git
- SLURM (for HPC clusters)
- OpenMP (comes with modern GCC versions)

SYSTEM LIBRARIES:
- Zlib (install via `sudo apt install zlib1g-dev` or module load)

EXTERNAL LIBRARIES (automatically fetched by CMake):
- Zstd
- LZ4
- Snappy
- Bzip2
- FastLZ (local source)
- cxxopts
- Google Benchmark
- half.hpp (half-precision float utility)

-----------------------------------------------
 SETUP AND USAGE
-----------------------------------------------

1. Clone the repository:
   git clone <your-repo-url>
   cd zstd-test-project

2. Configure the script "build1.sh" before running.
   You MUST edit the following variables:

   DATASET_DIR="/your/path/to/datasets"
   RESULTS_DIR="/your/path/to/output/results"
   EXECUTABLE="./build/external_tools/parallel-test"
   /home/youruser/programs/cmake-x.x.x/bin/cmake -DCMAKE_BUILD_TYPE=Release ..

   Replace paths with values that match your environment.

3. Run the full build + benchmark using:
   bash build1.sh

   This script will:
   - Build the project using 16 threads
   - Process each dataset in DATASET_DIR
   - Save results to RESULTS_DIR as .csv files

4. Sample manual execution:
   ./build/external_tools/parallel-test --dataset ./yourdata.dat --outcsv result.csv --threads 16 --bits 64 --method=zstd

-----------------------------------------------
 OUTPUT
-----------------------------------------------

Each result CSV file includes:
- Dataset name
- Threads used
- Block Size
- Type of runnig
- Compression ratio
- Execution time
 -Throughput


-----------------------------------------------
 CLI PARAMETERS
-----------------------------------------------

--dataset    : Path to dataset file
--outcsv     : Path to save result CSV
--threads    : Number of OpenMP threads
--bits       : Data bit width (e.g., 64)
--method     : One of [zstd, snappy, lz4, bzip2, zlib, fastlz]

