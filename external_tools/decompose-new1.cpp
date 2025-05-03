// ----------------------------------------------------------------------------
// Header Includes
// ----------------------------------------------------------------------------
#include "decompose.h"
#include "bzib_parallel.h"
#include "lz4_parallel.h"
#include "snappy_parallel.h"
#include "zlib-parallel.h"
#include "zstd_parallel.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cxxopts.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <omp.h>
#include <sstream>
#include <string>
#include <vector>
#include "half.hpp"

// Type alias for FP16
using fp16 = half_float::half;

// ----------------------------------------------------------------------------
// Global Variables
// ----------------------------------------------------------------------------
std::vector<uint8_t> globalByteArray;
std::map<std::string, std::vector<std::vector<std::vector<size_t>>>> datasetComponentMap;

// ----------------------------------------------------------------------------
// Parse configuration string into nested vector format
// ----------------------------------------------------------------------------
std::vector<std::vector<size_t>> parseClusterConfig(const std::string& configStr) {
  std::vector<std::vector<size_t>> clusters;
  std::vector<size_t> current;
  std::string num;
  bool inside = false;

  for (char ch : configStr) {
    if (ch == '{') {
      if (!inside) { inside = true; current.clear(); }
    } else if (ch == '}') {
      if (inside) {
        if (!num.empty()) { current.push_back(std::stoul(num)); num.clear(); }
        clusters.push_back(current);
        inside = false;
      }
    } else if (ch == ',' && inside) {
      if (!num.empty()) { current.push_back(std::stoul(num)); num.clear(); }
    } else if (std::isdigit(ch)) {
      num += ch;
    }
  }
  return clusters;
}

// ----------------------------------------------------------------------------
// Split a vector into N nearly equal chunks
// ----------------------------------------------------------------------------
std::vector<std::vector<uint8_t>> splitIntoChunks(const std::vector<uint8_t>& data, int numChunks) {
  std::vector<std::vector<uint8_t>> chunks;
  size_t totalSize = data.size();
  size_t chunkSize = totalSize / numChunks;
  size_t remainder = totalSize % numChunks;
  size_t offset = 0;
  for (int i = 0; i < numChunks; i++) {
    size_t currentSize = chunkSize + (i < remainder ? 1 : 0);
    std::vector<uint8_t> chunk(data.begin() + offset, data.begin() + offset + currentSize);
    chunks.push_back(chunk);
    offset += currentSize;
  }
  return chunks;
}

// ----------------------------------------------------------------------------
// Retrieve component configurations based on dataset name
// ----------------------------------------------------------------------------
std::vector<std::vector<std::vector<size_t>>> getComponentConfigurationsForDataset(const std::string& datasetName) {
  const auto it = datasetComponentMap.find(datasetName);
  if (it != datasetComponentMap.end()) return it->second;

  const auto defaultIt = datasetComponentMap.find("default");
  return (defaultIt != datasetComponentMap.end()) ? defaultIt->second : std::vector<std::vector<std::vector<size_t>>>{};
}

// ----------------------------------------------------------------------------
// Extract dataset name from file path
// ----------------------------------------------------------------------------
std::string extractDatasetName(const std::string& filePath) {
  size_t lastSlashPos = filePath.find_last_of("/\\");
  std::string fileName = (lastSlashPos == std::string::npos) ? filePath : filePath.substr(lastSlashPos + 1);
  size_t lastDotPos = fileName.find_last_of('.');
  return (lastDotPos == std::string::npos) ? fileName : fileName.substr(0, lastDotPos);
}

// ----------------------------------------------------------------------------
// Convert configuration to formatted string for display/logging
// ----------------------------------------------------------------------------
std::string configToString1(const std::vector<std::vector<size_t>>& config) {
  std::stringstream ss;
  ss << "{ ";
  for (size_t i = 0; i < config.size(); ++i) {
    ss << "[";
    for (size_t j = 0; j < config[i].size(); ++j) {
      ss << config[i][j];
      if (j + 1 < config[i].size()) ss << " ";
    }
    ss << "]";
    if (i + 1 < config.size()) ss << "- ";
  }
  ss << " }";
  return ss.str();
}

// ----------------------------------------------------------------------------
// Dataset loading from TSV files
// ----------------------------------------------------------------------------
std::pair<std::vector<float>, size_t> loadTSVDataset(const std::string& filePath) {
  std::vector<float> floatArray;
  std::ifstream file(filePath);
  std::string line;
  size_t rowCount = 0;
  if (file.is_open()) {
    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string value;
      std::getline(ss, value, '\t'); // skip first column
      while (std::getline(ss, value, '\t')) {
        floatArray.push_back(std::stof(value));
      }
      rowCount++;
    }
    file.close();
  } else {
    std::cerr << "Unable to open file: " << filePath << std::endl;
  }
  return {floatArray, rowCount};
}

std::pair<std::vector<double>, size_t> loadTSVDatasetdouble(const std::string& filePath) {
  std::vector<double> doubleArray;
  std::ifstream file(filePath);
  std::string line;
  size_t rowCount = 0;
  if (file.is_open()) {
    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string value;
      std::getline(ss, value, '\t'); // skip first column
      while (std::getline(ss, value, '\t')) {
        doubleArray.push_back(std::stod(value));
      }
      rowCount++;
    }
    file.close();
  } else {
    std::cerr << "Unable to open file: " << filePath << std::endl;
  }
  return {doubleArray, rowCount};
}

// ----------------------------------------------------------------------------
// Data type conversions (float, double, fp16 <-> byte arrays)
// ----------------------------------------------------------------------------
std::vector<uint8_t> convertFloatToBytes(const std::vector<float>& floatArray) {
  std::vector<uint8_t> byteArray(floatArray.size() * 4);
  for (size_t i = 0; i < floatArray.size(); i++) {
    uint8_t* floatBytes = reinterpret_cast<uint8_t*>(const_cast<float*>(&floatArray[i]));
    for (size_t j = 0; j < 4; j++) {
      byteArray[i * 4 + j] = floatBytes[j];
    }
  }
  return byteArray;
}

std::pair<std::vector<fp16>, size_t> loadTSVDatasetFp16(const std::string& filePath) {
  std::vector<fp16> fp16Array;
  std::ifstream file(filePath);
  std::string line;
  size_t rowCount = 0;

  if (file.is_open()) {
    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string value;
      while (std::getline(ss, value, '\t')) {
        try {
          float floatValue = std::stof(value);
          fp16Array.push_back(static_cast<fp16>(floatValue));
        } catch (const std::exception& e) {
          std::cerr << "Error converting value to fp16: " << value << " (" << e.what() << ")\n";
        }
      }
      rowCount++;
    }
    file.close();
  } else {
    std::cerr << "Unable to open file: " << filePath << std::endl;
  }

  return {fp16Array, rowCount};
}

std::vector<uint8_t> convertFp16ToBytes(const std::vector<fp16>& fp16Array) {
  std::vector<uint8_t> byteArray(fp16Array.size() * 2);
  for (size_t i = 0; i < fp16Array.size(); ++i) {
    const uint16_t* raw = reinterpret_cast<const uint16_t*>(&fp16Array[i]);
    byteArray[i * 2]     = static_cast<uint8_t>(*raw & 0xFF);
    byteArray[i * 2 + 1] = static_cast<uint8_t>((*raw >> 8) & 0xFF);
  }
  return byteArray;
}

std::vector<fp16> convertBytesToFp16(const std::vector<uint8_t>& byteArray) {
  if (byteArray.size() % 2 != 0)
    throw std::runtime_error("Byte array size is not a multiple of 2.");

  std::vector<fp16> fp16Array(byteArray.size() / 2);
  for (size_t i = 0; i < fp16Array.size(); ++i) {
    uint16_t value = static_cast<uint16_t>(byteArray[i * 2]) |
                     (static_cast<uint16_t>(byteArray[i * 2 + 1]) << 8);
    std::memcpy(&fp16Array[i], &value, sizeof(uint16_t));
  }
  return fp16Array;
}

std::vector<float> convertBytesToFloat(const std::vector<uint8_t>& byteArray) {
  if (byteArray.size() % 4 != 0)
    throw std::runtime_error("Byte array size is not a multiple of 4.");
  std::vector<float> floatArray(byteArray.size() / 4);
  for (size_t i = 0; i < floatArray.size(); i++) {
    const uint8_t* bytePtr = &byteArray[i * 4];
    float* floatPtr = reinterpret_cast<float*>(const_cast<uint8_t*>(bytePtr));
    floatArray[i] = *floatPtr;
  }
  return floatArray;
}

std::vector<uint8_t> convertDoubleToBytes(const std::vector<double>& doubleArray) {
  std::vector<uint8_t> byteArray(doubleArray.size() * 8);
  for (size_t i = 0; i < doubleArray.size(); i++) {
    uint8_t* doubleBytes = reinterpret_cast<uint8_t*>(const_cast<double*>(&doubleArray[i]));
    for (size_t j = 0; j < 8; j++) {
      byteArray[i * 8 + j] = doubleBytes[j];
    }
  }
  return byteArray;
}

std::vector<double> convertBytesToDouble(const std::vector<uint8_t>& byteArray) {
  if (byteArray.size() % 8 != 0)
    throw std::runtime_error("Byte array size is not a multiple of 8.");
  std::vector<double> doubleArray(byteArray.size() / 8);
  for (size_t i = 0; i < doubleArray.size(); i++) {
    const uint8_t* bytePtr = &byteArray[i * 8];
    const double* doublePtr = reinterpret_cast<const double*>(bytePtr);
    doubleArray[i] = *doublePtr;
  }
  return doubleArray;
}

// ----------------------------------------------------------------------------
// Compute compression and decompression throughput (in GB/s)
// ----------------------------------------------------------------------------
std::pair<double, double> calculateCompDecomThroughput(size_t originalSize, double compressedTime, double decompressedTime) {
  double originalSizeGB = static_cast<double>(originalSize) / 1e9;
  double compressionThroughput = originalSizeGB / static_cast<double>(compressedTime);
  double decompressionThroughput = originalSizeGB / static_cast<double>(decompressedTime);
  return {compressionThroughput, decompressionThroughput};
}

// ----------------------------------------------------------------------------
// Equality checks for vectors (with floating point tolerance)
// ----------------------------------------------------------------------------
bool areVectorsEqual(const std::vector<float>& a, const std::vector<float>& b, float epsilon = 1e-5) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); i++) {
    if (std::fabs(a[i] - b[i]) > epsilon) return false;
  }
  return true;
}

bool areVectorsEqualdouble(const std::vector<double>& a, const std::vector<double>& b, float epsilon = 1e-5) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); i++) {
    if (std::fabs(a[i] - b[i]) > epsilon) return false;
  }
  return true;
}

// ----------------------------------------------------------------------------
// Helper: Block a vector into chunks of a given size
// ----------------------------------------------------------------------------
std::vector<std::vector<uint8_t>> blockData(const std::vector<uint8_t>& data, size_t blockSize) {
  std::vector<std::vector<uint8_t>> blocks;
  size_t totalSize = data.size();
  size_t numBlocks = (totalSize + blockSize - 1) / blockSize;
  blocks.reserve(numBlocks);
  for (size_t i = 0; i < totalSize; i += blockSize) {
    size_t end = std::min(i + blockSize, totalSize);
    blocks.push_back(std::vector<uint8_t>(data.begin() + i, data.begin() + end));
  }
  return blocks;
}

int main(int argc, char* argv[]) {
  // Setup command line options.
  cxxopts::Options options("DataCompressor", "Compress datasets and profile the compression");
  options.add_options()
      ("d,dataset",   "Path to the dataset file", cxxopts::value<std::string>())
          ("o,outcsv",    "Output CSV file path",     cxxopts::value<std::string>())
              ("t,threads",   "Number of threads to use", cxxopts::value<int>()->default_value("10"))
                  ("b,bits",      "Floating-point precision (32 or 64 bits)", cxxopts::value<int>()->default_value("64"))
                      ("m,method",    "Compression method (fastlz or zstd)", cxxopts::value<std::string>()->default_value("fastlz"))
                          ("h,help",      "Print help");
  auto result = options.parse(argc, argv);
  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  // Read command-line arguments.
  std::string datasetPath = result["dataset"].as<std::string>();
  std::string outputCSV   = result["outcsv"].as<std::string>();
  int userThreads         = result["threads"].as<int>();
  int precisionBits       = result["bits"].as<int>();
  std::string method      = result["method"].as<std::string>();
  ///read path to the clustering .csv                                                               /
  std::ifstream file("/home/jamalids/Downloads/64-config-S.csv");
  if (!file.is_open()) {
    std::cerr << "Cannot open cluster config CSV." << std::endl;
    exit(1);
  }

  std::string headerLine;
  std::getline(file, headerLine);  // Skip header

  std::string line;
  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string dataset, clusterStr;
    std::getline(ss, dataset, ',');
    std::getline(ss, clusterStr);

    if (!dataset.empty() && !clusterStr.empty()) {
      auto parsed = parseClusterConfig(clusterStr);
      datasetComponentMap[dataset].push_back(parsed);
    }
  }

  std::vector<int> threadList = { userThreads };
  int runCount = 1;

  size_t rowCount;
  std::string datasetName = extractDatasetName(datasetPath);
  std::cout << "Dataset Name: " << datasetName << std::endl;

  if (precisionBits == 64) {
    auto [doubleArray, rows] = loadTSVDatasetdouble(datasetPath);
    if (doubleArray.empty()) {
      std::cerr << "Failed to load dataset from " << datasetPath << std::endl;
      return 1;
    }
    globalByteArray = convertDoubleToBytes(doubleArray);
    rowCount = rows;
    std::cout << "Loaded " << rows << " rows (64-bit) with "
              << doubleArray.size() << " total values.\n";

  } else if (precisionBits == 32) {
    auto [floatArray, rows] = loadTSVDataset(datasetPath);
    if (floatArray.empty()) {
      std::cerr << "Failed to load dataset from " << datasetPath << std::endl;
      return 1;
    }
    globalByteArray = convertFloatToBytes(floatArray);
    rowCount = rows;
    std::cout << "Loaded " << rows << " rows (32-bit) with "
              << floatArray.size() << " total values.\n";

  } else if (precisionBits == 16) {
    auto [fp16Array, rows] = loadTSVDatasetFp16(datasetPath);

    if (fp16Array.empty()) {
      std::cerr << "Failed to load dataset from " << datasetPath << std::endl;
      return 1;
    }
    globalByteArray = convertFp16ToBytes(fp16Array);

    rowCount = rows;
    std::cout << "Loaded " << rows << " rows (16-bit) with "
              << fp16Array.size() << " total values.\n";

  } else {
    std::cerr << "Unsupported precision: " << precisionBits
              << ". Use 16, 32, or 64." << std::endl;
    return 1;
  }

  size_t totalBytes = globalByteArray.size();

  // Define block sizes (in bytes)
  std::vector<size_t> blockSizes = {
      400 *1024,
      640* 1024,
      768*1024,
      1024*1024,
      10*1024*1024,
      24 *1024 *1024,
      27*1024*1024,
      30 *1024 *1024,

  };

  // Open the CSV output file.

  std::ofstream outputFile(outputCSV);

  if (!outputFile) {
    std::cerr << "Failed to open the file for writing: " << outputCSV << std::endl;
    return 1;
  }
  outputFile << "Index;DatasetName;Threads;BlockSize;ConfigString;RunType;CompressionRatio;"
             << "TotalTimeCompressed;TotalTimeDecompressed;CompressionThroughput;DecompressionThroughput;TotalValues;Num-Block;Compressedsize\n";

  int recordIndex = 1;
  auto componentConfigurationsList = getComponentConfigurationsForDataset(datasetName);

  // ----------------------------------------------------------------------
  // Loop over thread counts and run iterations.
  // ----------------------------------------------------------------------
  for (int currentThreads : threadList) {
    for (int run = 1; run <= runCount; run++) {
      std::cout << "\n[INFO] Starting run " << run << "/" << runCount
                << " with " << currentThreads << " threads.\n";
      int numThreads = currentThreads;

      if (method == "fastlz") {
        // ------------------------------
        // A. FULL COMPRESSION WITH BLOCKING - PARALLEL (FastLZ)
        // ------------------------------
        for (size_t bs : blockSizes) {
          std::cout << "Testing (FastLZ) with block size = " << bs << " bytes." << std::endl;
          size_t totalSize = globalByteArray.size();
          size_t numBlocks = (totalSize + bs - 1) / bs;
          size_t totalCompressedSize = 0;
          double totalCompTime = 0.0, totalDecompTime = 0.0;
          std::vector<std::vector<uint8_t>> compressedBlocks(numBlocks);
          std::vector<double> blockCompTimes(numBlocks, 0.0);
          std::vector<double> blockDecompTimes(numBlocks, 0.0);
          std::vector<size_t> blockCompressedSizes(numBlocks, 0);
          ProfilingInfo pi_parallel;
          pi_parallel.config_string = "N/A";
          omp_set_num_threads(numThreads);

          auto compStartOverall = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
          for (int i = 0; i < static_cast<int>(numBlocks); i++) {
            size_t start = i * bs;
            size_t end = std::min(start + bs, totalSize);
            size_t blockLength = end - start;
            const uint8_t* blockStart = globalByteArray.data() + start;
            auto startTime = std::chrono::high_resolution_clock::now();
            size_t cSize = compressWithFastLZ1(blockStart, blockLength, compressedBlocks[i]);
            auto endTime = std::chrono::high_resolution_clock::now();
            blockCompTimes[i] = std::chrono::duration<double>(endTime - startTime).count();
            blockCompressedSizes[i] = cSize;
#pragma omp atomic
            totalCompressedSize += cSize;
          }
          auto compEndOverall = std::chrono::high_resolution_clock::now();
          totalCompTime = std::chrono::duration<double>(compEndOverall - compStartOverall).count();

          std::vector<uint8_t> finalReconstructed(totalSize, 0);
          double decompStartOverall = omp_get_wtime();
#pragma omp parallel for
          for (int i = 0; i < static_cast<int>(numBlocks); i++) {
            size_t start = i * bs;
            size_t end = std::min(start + bs, totalSize);
            size_t blockLength = end - start;
            uint8_t* dest = finalReconstructed.data() + start;
            double startTime = omp_get_wtime();
            decompressWithFastLZ1(compressedBlocks[i], dest, blockLength);
            double endTime = omp_get_wtime();
            blockDecompTimes[i] = endTime - startTime;
          }
          double decompEndOverall = omp_get_wtime();
          totalDecompTime = decompEndOverall - decompStartOverall;

          if (finalReconstructed == globalByteArray)
            std::cout << "[INFO] (FastLZ) Reconstructed full data matches the original (PARALLEL)." << std::endl;
          else
            std::cerr << "[ERROR] (FastLZ) Reconstructed data does NOT match the original (PARALLEL)." << std::endl;

          double compRatio = calculateCompressionRatio(totalBytes, totalCompressedSize);
          auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(totalBytes, totalCompTime, totalDecompTime);

          outputFile << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << bs << ";" << "N/A" << ";"
                     << "Full_Block_Parallel" << ";" << compRatio << ";" << totalCompTime << ";" << totalDecompTime << ";"
                     << compThroughput << ";" << decompThroughput << ";" << totalBytes << ";" << numBlocks << ";" << totalCompressedSize <<"\n";
        }

        // ------------------------------
        // B. FULL COMPRESSION WITHOUT BLOCKING (non-blocking) - FastLZ
        // ------------------------------
        {
          ProfilingInfo pi_full;
          pi_full.type = "Full Compression (Non-blocking)";
          std::vector<uint8_t> compressedData, decompressedData;
          auto start = std::chrono::high_resolution_clock::now();
          size_t compressedSize = compressWithFastLZ1(globalByteArray.data(), globalByteArray.size(), compressedData);
          auto end = std::chrono::high_resolution_clock::now();
          pi_full.total_time_compressed = std::chrono::duration<double>(end - start).count();

          decompressedData.resize(globalByteArray.size());
          start = std::chrono::high_resolution_clock::now();
          decompressWithFastLZ1(compressedData, decompressedData.data(), globalByteArray.size());
          end = std::chrono::high_resolution_clock::now();
          pi_full.total_time_decompressed = std::chrono::duration<double>(end - start).count();

          double compRatio = calculateCompressionRatio(totalBytes, compressedSize);
          auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(totalBytes, pi_full.total_time_compressed, pi_full.total_time_decompressed);

          outputFile << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << "N/A" << ";" << "N/A" << ";"
                     << "Full" << ";" << compRatio << ";" << pi_full.total_time_compressed << ";" << pi_full.total_time_decompressed << ";"
                     << compThroughput << ";" << decompThroughput << ";" << totalBytes <<";" << 1<< ";" << compressedSize <<"\n";
        }

        // ------------------------------
        // C. DECOMPOSED COMPRESSION (Blocking Parallel and Decompose-Then-Chunk) - FastLZ
        // ------------------------------

        for (const auto& componentConfig : componentConfigurationsList) {
          std::string configStr = configToString1(componentConfig);
          std::cout << "\nConfig with " << componentConfig.size() << " sub-config(s): " << configStr << "\n";
          // (i) Blocking Parallel Version:
          for (size_t bs : blockSizes) {
            std::cout << "Testing (FastLZ) with block size = " << bs << " bytes." << std::endl;
            size_t totalSize = globalByteArray.size();
            size_t numBlocks = (totalSize + bs - 1) / bs;
            struct BlockView { const uint8_t* data; size_t size; };
            std::vector<BlockView> fullBlocks;
            fullBlocks.reserve(numBlocks);
            for (size_t i = 0; i < numBlocks; i++) {
              size_t start = i * bs;
              size_t end = std::min(start + bs, totalSize);
              fullBlocks.push_back({ globalByteArray.data() + start, end - start });
            }
            size_t totalCompressedSize = 0;
            double totalCompTime = 0.0, totalDecompTime = 0.0;
            std::vector<std::vector<std::vector<uint8_t>>> compressedBlocks(numBlocks);
            std::vector<double> blockCompTimes(numBlocks, 0.0);
            std::vector<double> blockDecompTimes(numBlocks, 0.0);
            std::vector<size_t> blockCompressedSizes(numBlocks, 0);
            ProfilingInfo pi_parallel;
            pi_parallel.config_string = configStr;
            omp_set_num_threads(numThreads);
            auto compStartOverall = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
            for (int i = 0; i < static_cast<int>(numBlocks); i++) {
              auto start = std::chrono::high_resolution_clock::now();
              double cSize = fastlzFusedDecomposedParallel(fullBlocks[i].data, fullBlocks[i].size,
                                                           pi_parallel, compressedBlocks[i],
                                                           componentConfig, 4);
              auto end = std::chrono::high_resolution_clock::now();
              blockCompTimes[i] = std::chrono::duration<double>(end - start).count();
              blockCompressedSizes[i] = cSize;
#pragma omp atomic
              totalCompressedSize += cSize;
            }
            auto compEndOverall = std::chrono::high_resolution_clock::now();
            totalCompTime = std::chrono::duration<double>(compEndOverall - compStartOverall).count();
            std::vector<uint8_t> fullReconstructed(totalSize);
            double decompStartOverall = omp_get_wtime();
#pragma omp parallel for
            for (int i = 0; i < static_cast<int>(numBlocks); i++) {
              double start = omp_get_wtime();
              uint8_t* dest = fullReconstructed.data() + i * bs;
              fastlzDecomposedParallelDecompression(compressedBlocks[i], pi_parallel, componentConfig,
                                                    4, fullBlocks[i].size, dest);
              double end = omp_get_wtime();
              blockDecompTimes[i] = end - start;
            }
            double decompEndOverall = omp_get_wtime();
            totalDecompTime = decompEndOverall - decompStartOverall;
            if (fullReconstructed == globalByteArray)
              std::cout << "[INFO] (FastLZ) Full reconstructed data matches the original data." << std::endl;
            else
              std::cerr << "[ERROR] (FastLZ) Full reconstructed data does NOT match the original data." << std::endl;
            double compRatio = calculateCompressionRatio(totalBytes, totalCompressedSize);
            auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(totalBytes, totalCompTime, totalDecompTime);
            outputFile << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << bs << ";" << configStr << ";"
                       << "Decompose_Block_Parallel" << ";" << compRatio << ";" << totalCompTime << ";" << totalDecompTime << ";"
                       << compThroughput << ";" << decompThroughput << ";" << totalBytes << ";" << numBlocks<< ";" << totalCompressedSize <<"\n";
          }
          // (ii) Decompose-Then-Chunk Version:
          for (size_t bs : blockSizes) {
            std::cout << "Testing (FastLZ) with chunk block size = " << bs << " bytes." << std::endl;
            size_t totalSize = globalByteArray.size();
            ProfilingInfo pi_chunk;
            pi_chunk.config_string = configStr;
            size_t numBlocks = (totalSize + bs - 1) / bs;
            std::vector<std::vector<std::vector<uint8_t>>> compressedBlocks;
            omp_set_num_threads(numThreads);
            double compStartOverall = omp_get_wtime();
            size_t totalCompressedSize = fastlzDecomposedThenChunkedParallelCompression(
                globalByteArray.data(), globalByteArray.size(),
                pi_chunk,
                compressedBlocks,
                componentConfig,
                numThreads,
                bs);
            double compEndOverall = omp_get_wtime();
            pi_chunk.total_time_compressed = compEndOverall - compStartOverall;
            std::vector<uint8_t> finalReconstructed(totalSize);
            double decompStartOverall = omp_get_wtime();
            fastlzDecomposedThenChunkedParallelDecompression(
                compressedBlocks,
                pi_chunk,
                componentConfig,
                numThreads,
                globalByteArray.size(),
                bs,
                finalReconstructed.data());
            double decompEndOverall = omp_get_wtime();
            pi_chunk.total_time_decompressed = decompEndOverall - decompStartOverall;
            if (finalReconstructed == globalByteArray)
              std::cout << "[INFO] (FastLZ) Final reconstructed data matches the original data." << std::endl;
            else
              std::cerr << "[ERROR] (FastLZ) Final reconstructed data does NOT match the original data." << std::endl;
            double compRatio = calculateCompressionRatio(totalSize, totalCompressedSize);
            auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(totalSize, pi_chunk.total_time_compressed, pi_chunk.total_time_decompressed);
            outputFile << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << bs << ";" << configStr << ";"
                       << "Decompose_Chunk_Parallel" << ";" << compRatio << ";" << pi_chunk.total_time_compressed << ";" << pi_chunk.total_time_decompressed << ";"
                       << compThroughput << ";" << decompThroughput << ";" << totalBytes << ";" << numBlocks <<";" << totalCompressedSize <<"\n";
          }
        }
      } else if (method == "zstd") {
        //------------------------------
        // A. FULL COMPRESSION WITH BLOCKING - PARALLEL (Zstd)
        // ------------------------------
        for (size_t bs : blockSizes) {
          std::cout << "Testing (Zstd) with block size = " << bs << " bytes." << std::endl;
          size_t totalSize = globalByteArray.size();
          size_t numBlocks = (totalSize + bs - 1) / bs;
          size_t totalCompressedSize = 0;
          double totalCompTime = 0.0, totalDecompTime = 0.0;
          std::vector<std::vector<uint8_t>> compressedBlocks(numBlocks);
          std::vector<double> blockCompTimes(numBlocks, 0.0);
          std::vector<double> blockDecompTimes(numBlocks, 0.0);
          std::vector<size_t> blockCompressedSizes(numBlocks, 0);
          ProfilingInfo pi_parallel;
          pi_parallel.config_string = "N/A";
          omp_set_num_threads(numThreads);
          auto compStartOverall = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
          for (int i = 0; i < static_cast<int>(numBlocks); i++) {
            size_t start = i * bs;
            size_t end = std::min(start + bs, totalSize);
            size_t blockLength = end - start;
            std::vector<uint8_t> blockData(globalByteArray.begin() + start, globalByteArray.begin() + end);
            auto startTime = std::chrono::high_resolution_clock::now();
            size_t cSize = compressWithZstd(blockData, compressedBlocks[i], 3);
            auto endTime = std::chrono::high_resolution_clock::now();
            blockCompTimes[i] = std::chrono::duration<double>(endTime - startTime).count();
            blockCompressedSizes[i] = cSize;
#pragma omp atomic
            totalCompressedSize += cSize;
          }
          auto compEndOverall = std::chrono::high_resolution_clock::now();
          totalCompTime = std::chrono::duration<double>(compEndOverall - compStartOverall).count();
          std::vector<uint8_t> finalReconstructed(totalSize, 0);
          double decompStartOverall = omp_get_wtime();
#pragma omp parallel for
          for (int i = 0; i < static_cast<int>(numBlocks); i++) {
            size_t start = i * bs;
            size_t end = std::min(start + bs, totalSize);
            size_t blockLength = end - start;
            std::vector<uint8_t> decompBlock;
            double startTime = omp_get_wtime();
            decompressWithZstd(compressedBlocks[i], decompBlock, blockLength);
            double endTime = omp_get_wtime();
            blockDecompTimes[i] = endTime - startTime;
            std::copy(decompBlock.begin(), decompBlock.end(), finalReconstructed.begin() + start);
          }
          double decompEndOverall = omp_get_wtime();
          totalDecompTime = decompEndOverall - decompStartOverall;
          if (finalReconstructed == globalByteArray)
            std::cout << "[INFO] (Zstd) Reconstructed data matches the original (PARALLEL)." << std::endl;
          else
            std::cerr << "[ERROR] (Zstd) Reconstructed data does NOT match the original (PARALLEL)." << std::endl;
          double compRatio = calculateCompressionRatio(totalBytes, totalCompressedSize);
          auto [compThroughput, decompThroughput] =
              calculateCompDecomThroughput(totalBytes, totalCompTime, totalDecompTime);
          outputFile << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << bs << ";"
                     << "N/A" << ";" << "Chunked_parallel" << ";" << compRatio << ";"
                     << totalCompTime << ";" << totalDecompTime << ";"
                     << compThroughput << ";" << decompThroughput << ";" << totalBytes << ";" << numBlocks<< ";" << totalCompressedSize <<"\n";
        }

        // ------------------------------
        // B. FULL COMPRESSION WITHOUT BLOCKING (Non-blocking)
        // ------------------------------
        {
          ProfilingInfo pi_full;
          pi_full.type = "Full Compression (Non-blocking)";
          std::vector<uint8_t> compressedData, decompressedData;
          auto start = std::chrono::high_resolution_clock::now();
          size_t compressedSize = zstdCompression(globalByteArray, pi_full, compressedData);
          auto end = std::chrono::high_resolution_clock::now();
          pi_full.total_time_compressed = std::chrono::duration<double>(end - start).count();
          decompressedData.resize(globalByteArray.size());
          start = std::chrono::high_resolution_clock::now();
          zstdDecompression(compressedData, decompressedData, pi_full);
          end = std::chrono::high_resolution_clock::now();
          pi_full.total_time_decompressed = std::chrono::duration<double>(end - start).count();
          double compRatio = calculateCompressionRatio(totalBytes, compressedSize);
          auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(
              totalBytes, pi_full.total_time_compressed, pi_full.total_time_decompressed);
          outputFile << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << "N/A" << ";" << "N/A" << ";"
                     << "Full" << ";" << compRatio << ";" << pi_full.total_time_compressed << ";"
                     << pi_full.total_time_decompressed << ";" << compThroughput << ";" << decompThroughput
                     << ";" << totalBytes << ";" << 1 << ";" << compressedSize <<"\n";
        }

        // ------------------------------
        // C. Decomposed Compression with Blocking Parallel (Zstd)
        // ------------------------------
        for (const auto& componentConfig : componentConfigurationsList) {
          std::string configStr = configToString1(componentConfig);
          std::cout << "\nConfig with " << componentConfig.size()
                    << " sub-config(s): " << configStr << "\n";
          for (size_t bs : blockSizes) {
            std::cout << "Testing (Zstd) with block size = " << bs << " bytes." << std::endl;
            size_t totalSize = globalByteArray.size();
            size_t numBlocks = (totalSize + bs - 1) / bs;

            // Prepare a vector of block views.
            struct BlockView { const uint8_t* data; size_t size; };
            std::vector<BlockView> fullBlocks;
            fullBlocks.reserve(numBlocks);
            for (size_t i = 0; i < numBlocks; i++) {
              size_t start = i * bs;
              size_t end = std::min(start + bs, totalSize);
              fullBlocks.push_back({ globalByteArray.data() + start, end - start });
            }

            size_t totalCompressedSize = 0;
            double totalCompTime = 0.0, totalDecompTime = 0.0;
            std::vector<double> blockCompTimes(numBlocks, 0.0);
            std::vector<double> blockDecompTimes(numBlocks, 0.0);
            std::vector<size_t> blockCompressedSizes(numBlocks, 0);
            ProfilingInfo pi_parallel;
            pi_parallel.config_string = configStr;
            omp_set_num_threads(numThreads);

            // This vector will hold, for each block, the per–component compressed data.
            std::vector<std::vector<std::vector<uint8_t>>> allCompressedBlocks(numBlocks);

            // --- Compression Phase: Use the fused Zstd decomposed parallel compression ---
            auto compStartOverall = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(numBlocks); i++) {
              auto startTime = std::chrono::high_resolution_clock::now();

              std::vector<std::vector<uint8_t>> compComponents;
              size_t cSize = zstdFusedDecomposedParallel(fullBlocks[i].data, fullBlocks[i].size,
                                                         pi_parallel, compComponents,
                                                         componentConfig, numThreads);
              auto endTime = std::chrono::high_resolution_clock::now();
              blockCompTimes[i] = std::chrono::duration<double>(endTime - startTime).count();
              blockCompressedSizes[i] = cSize;
#pragma omp atomic
              totalCompressedSize += cSize;
              allCompressedBlocks[i] = std::move(compComponents);
            }
            auto compEndOverall = std::chrono::high_resolution_clock::now();
            totalCompTime = std::chrono::duration<double>(compEndOverall - compStartOverall).count();

            // --- Decompression Phase: Use the Zstd decomposed parallel decompression function ---
            std::vector<uint8_t> finalReconstructed(totalSize, 0);
            double decompStartOverall = omp_get_wtime();
#pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(numBlocks); i++) {
              // Allocate a temporary vector to hold the decompressed block.
              std::vector<uint8_t> decompBlock(fullBlocks[i].size, 0);
              auto startTime = omp_get_wtime();
              zstdDecomposedParallelDecompression(allCompressedBlocks[i], pi_parallel,
                                                  componentConfig, numThreads,
                                                  fullBlocks[i].size, decompBlock.data());
              auto endTime = omp_get_wtime();
              blockDecompTimes[i] = endTime - startTime;
              // Copy the decompressed block into its proper position.
              std::copy(decompBlock.begin(), decompBlock.end(),
                        finalReconstructed.begin() + i * bs);
            }
            double decompEndOverall = omp_get_wtime();
            totalDecompTime = decompEndOverall - decompStartOverall;

            if (finalReconstructed == globalByteArray)
              std::cout << "[INFO] (Zstd) Reconstructed data matches the original (PARALLEL)." << std::endl;
            else
              std::cerr << "[ERROR] (Zstd) Reconstructed data does NOT match the original (PARALLEL)." << std::endl;

            double compRatio = calculateCompressionRatio(totalBytes, totalCompressedSize);
            auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(totalBytes, totalCompTime, totalDecompTime);
            outputFile << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << bs << ";"
                       << configStr << ";" << "Chunked_Decompose_Parallel" << ";" << compRatio << ";"
                       << totalCompTime << ";" << totalDecompTime << ";"
                       << compThroughput << ";" << decompThroughput << ";" << totalBytes <<";" << numBlocks << ";" << totalCompressedSize << "\n";
          }
        }
        //----------------------------------------
        //--- Decomposed Compression without Chunking (Zstd) ---
        //---------------------------------------------
        //
        for (const auto& componentConfig : componentConfigurationsList) {
          std::string configStr = configToString1(componentConfig);
          std::cout << "\nConfig with " << componentConfig.size()
                    << " sub-config(s): " << configStr << "\n";
          std::cout << "Testing (Zstd) without chunking." << std::endl;

          size_t totalSize = globalByteArray.size();
          size_t totalCompressedSize = 0;
          double totalCompTime = 0.0, totalDecompTime = 0.0;

          // Create a separate ProfilingInfo structure for the non-chunked test.
          ProfilingInfo pi_nonchunked;
          pi_nonchunked.config_string = configStr;

          // --- Compression Phase: Compress the entire data in one go ---
          std::vector<std::vector<uint8_t>> compComponents; // Per–component compressed data.
          auto compStartOverall = std::chrono::high_resolution_clock::now();
          size_t cSize = zstdFusedDecomposedParallel(globalByteArray.data(), totalSize,
                                                     pi_nonchunked, compComponents,
                                                     componentConfig, numThreads);
          auto compEndOverall = std::chrono::high_resolution_clock::now();
          totalCompTime = std::chrono::duration<double>(compEndOverall - compStartOverall).count();
          totalCompressedSize = cSize;

          // --- Decompression Phase: Decompress the entire data ---
          std::vector<uint8_t> finalReconstructed(totalSize, 0);
          auto decompStartOverall = std::chrono::high_resolution_clock::now();
          zstdDecomposedParallelDecompression(compComponents, pi_nonchunked,
                                              componentConfig, numThreads,
                                              totalSize, finalReconstructed.data());
          auto decompEndOverall = std::chrono::high_resolution_clock::now();
          totalDecompTime = std::chrono::duration<double>(decompEndOverall - decompStartOverall).count();

          // Verify that the decompressed data matches the original.
          if (finalReconstructed == globalByteArray)
            std::cout << "[INFO] (Zstd) Reconstructed data matches the original (NON-CHUNKED)." << std::endl;
          else
            std::cerr << "[ERROR] (Zstd) Reconstructed data does NOT match the original (NON-CHUNKED)." << std::endl;

          // Calculate the compression ratio and throughput.
          double compRatio = calculateCompressionRatio(totalSize, totalCompressedSize);
          auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(totalSize, totalCompTime, totalDecompTime);

          // Record the results (using "N/A" or similar for block size since no chunking is used).
          outputFile << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << "N/A" << ";"
                     << configStr << ";" << "Decompose_NonChunked" << ";" << compRatio << ";"
                     << totalCompTime << ";" << totalDecompTime << ";"
                     << compThroughput << ";" << decompThroughput << ";"  << totalBytes <<";" << 1 << ";" << totalCompressedSize << "\n";
        }

        // ------------------------------
        // D. Decompose-Then-Chunk Parallel Compression (Zstd)
        // ------------------------------
        size_t totalSize = globalByteArray.size();
        for (const auto& componentConfig : componentConfigurationsList) {
          std::string configStr = configToString1(componentConfig);
          std::cout << "\nConfig with " << componentConfig.size() << " sub-config(s): " << configStr << "\n";

          for (size_t bs : blockSizes) {
            size_t numBlocks = (totalSize + bs - 1) / bs;
            std::cout << "Testing (Zstd) with chunk block size = " << bs << " bytes." << std::endl;
            size_t totalSize = globalByteArray.size();
            ProfilingInfo pi_chunk;
            pi_chunk.config_string = configStr;
            std::vector<std::vector<std::vector<uint8_t>>> compressedBlocks;
            omp_set_num_threads(numThreads);
            double compStartOverall = omp_get_wtime();
            size_t totalCompressedSize = zstdDecomposedThenChunkedParallelCompression(
                globalByteArray.data(), globalByteArray.size(),
                pi_chunk,
                compressedBlocks,
                componentConfig,
                numThreads,
                bs);
            double compEndOverall = omp_get_wtime();
            pi_chunk.total_time_compressed = compEndOverall - compStartOverall;
            std::vector<uint8_t> finalReconstructed(totalSize);
            double decompStartOverall = omp_get_wtime();
            zstdDecomposedThenChunkedParallelDecompression(
                compressedBlocks,
                pi_chunk,
                componentConfig,
                numThreads,
                globalByteArray.size(),
                bs,
                finalReconstructed.data());
            double decompEndOverall = omp_get_wtime();
            pi_chunk.total_time_decompressed = decompEndOverall - decompStartOverall;
            if (finalReconstructed == globalByteArray)
              std::cout << "[INFO] (Zstd) Final reconstructed data matches the original." << std::endl;
            else
              std::cerr << "[ERROR] (Zstd) Final reconstructed data does NOT match the original." << std::endl;
            double compRatio = calculateCompressionRatio(totalSize, totalCompressedSize);
            auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(
                totalSize, pi_chunk.total_time_compressed, pi_chunk.total_time_decompressed);
            outputFile << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << bs << ";"
                       << configStr << ";" << "Decompose_Chunk_Parallel" << ";" << compRatio << ";"
                       << pi_chunk.total_time_compressed << ";" << pi_chunk.total_time_decompressed << ";"
                       << compThroughput << ";" << decompThroughput << ";"  << totalBytes <<";" << numBlocks << ";" << totalCompressedSize <<  "\n";
          }
        }
      }

      ///bzip///////////////////////////////////////////////////////////

      else if (method == "bzip") {
         // ------------------------------
        // A. FULL COMPRESSION WITH BLOCKING - PARALLEL (bzip)
        // ------------------------------
        for (size_t bs : blockSizes) {
          std::cout << "Testing (bzip) with block size = " << bs << " bytes." << std::endl;
          size_t totalSize = globalByteArray.size();
          size_t numBlocks = (totalSize + bs - 1) / bs;
          size_t totalCompressedSize = 0;
          double totalCompTime = 0.0, totalDecompTime = 0.0;
          std::vector<std::vector<uint8_t>> compressedBlocks(numBlocks);
          std::vector<double> blockCompTimes(numBlocks, 0.0);
          std::vector<double> blockDecompTimes(numBlocks, 0.0);
          std::vector<size_t> blockCompressedSizes(numBlocks, 0);
          ProfilingInfo pi_parallel;
          pi_parallel.config_string = "N/A";
          omp_set_num_threads(numThreads);
          auto compStartOverall = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
          for (int i = 0; i < static_cast<int>(numBlocks); i++) {
            size_t start = i * bs;
            size_t end = std::min(start + bs, totalSize);
            size_t blockLength = end - start;
            std::vector<uint8_t> blockData(globalByteArray.begin() + start, globalByteArray.begin() + end);
            auto startTime = std::chrono::high_resolution_clock::now();
            size_t cSize = compressWithbzip2(blockData, compressedBlocks[i], 3);
            auto endTime = std::chrono::high_resolution_clock::now();
            blockCompTimes[i] = std::chrono::duration<double>(endTime - startTime).count();
            blockCompressedSizes[i] = cSize;
#pragma omp atomic
            totalCompressedSize += cSize;
          }
          auto compEndOverall = std::chrono::high_resolution_clock::now();
          totalCompTime = std::chrono::duration<double>(compEndOverall - compStartOverall).count();
          std::vector<uint8_t> finalReconstructed(totalSize, 0);
          double decompStartOverall = omp_get_wtime();
#pragma omp parallel for
          for (int i = 0; i < static_cast<int>(numBlocks); i++) {
            size_t start = i * bs;
            size_t end = std::min(start + bs, totalSize);
            size_t blockLength = end - start;
            std::vector<uint8_t> decompBlock;
            double startTime = omp_get_wtime();
            decompressWithbzip2(compressedBlocks[i], decompBlock, blockLength);
            double endTime = omp_get_wtime();
            blockDecompTimes[i] = endTime - startTime;
            std::copy(decompBlock.begin(), decompBlock.end(), finalReconstructed.begin() + start);
          }
          double decompEndOverall = omp_get_wtime();
          totalDecompTime = decompEndOverall - decompStartOverall;
          if (finalReconstructed == globalByteArray)
            std::cout << "[INFO] (bzip) Reconstructed data matches the original (PARALLEL)." << std::endl;
          else
            std::cerr << "[ERROR] (bzip) Reconstructed data does NOT match the original (PARALLEL)." << std::endl;
          double compRatio = calculateCompressionRatio(totalBytes, totalCompressedSize);
          auto [compThroughput, decompThroughput] =
              calculateCompDecomThroughput(totalBytes, totalCompTime, totalDecompTime);
          outputFile << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << bs << ";"
                     << "N/A" << ";" << "Chunked_parallel" << ";" << compRatio << ";"
                     << totalCompTime << ";" << totalDecompTime << ";"
                     << compThroughput << ";" << decompThroughput << ";"  << totalBytes <<";" << numBlocks << ";" << totalCompressedSize <<  "\n";
        }

        // ------------------------------
        // B. FULL COMPRESSION WITHOUT BLOCKING (Non-blocking) using bzip
        // ------------------------------
        {
          ProfilingInfo pi_full;
          pi_full.type = "Full Compression (Non-blocking)";
          std::vector<uint8_t> compressedData, decompressedData;
          auto start = std::chrono::high_resolution_clock::now();
          size_t compressedSize = bzibCompression(globalByteArray, pi_full, compressedData);
          auto end = std::chrono::high_resolution_clock::now();
          pi_full.total_time_compressed = std::chrono::duration<double>(end - start).count();
          decompressedData.resize(globalByteArray.size());
          start = std::chrono::high_resolution_clock::now();
          size_t totalSize = globalByteArray.size();
          bzibDecompression(compressedData, decompressedData,totalSize);
          end = std::chrono::high_resolution_clock::now();
          pi_full.total_time_decompressed = std::chrono::duration<double>(end - start).count();
          double compRatio = calculateCompressionRatio(totalBytes, compressedSize);
          auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(
              totalBytes, pi_full.total_time_compressed, pi_full.total_time_decompressed);
          outputFile << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << "N/A" << ";" << "N/A" << ";"
                     << "Full" << ";" << compRatio << ";" << pi_full.total_time_compressed << ";"
                     << pi_full.total_time_decompressed << ";" << compThroughput << ";" << decompThroughput
                     << ";" << totalBytes <<";" << 1 << ";" << compressedSize <<  "\n";
        }

        // ------------------------------
        // C. Decomposed Compression with Blocking Parallel (bzip)
        // ------------------------------
        for (const auto& componentConfig : componentConfigurationsList) {
          std::string configStr = configToString1(componentConfig);
          std::cout << "\nConfig with " << componentConfig.size()
                    << " sub-config(s): " << configStr << "\n";
          for (size_t bs : blockSizes) {
            std::cout << "Testing (bzip) with block size = " << bs << " bytes." << std::endl;
            size_t totalSize = globalByteArray.size();
            size_t numBlocks = (totalSize + bs - 1) / bs;

            // Prepare a vector of block views.
            struct BlockView { const uint8_t* data; size_t size; };
            std::vector<BlockView> fullBlocks;
            fullBlocks.reserve(numBlocks);
            for (size_t i = 0; i < numBlocks; i++) {
              size_t start = i * bs;
              size_t end = std::min(start + bs, totalSize);
              fullBlocks.push_back({ globalByteArray.data() + start, end - start });
            }

            size_t totalCompressedSize = 0;
            double totalCompTime = 0.0, totalDecompTime = 0.0;
            std::vector<double> blockCompTimes(numBlocks, 0.0);
            std::vector<double> blockDecompTimes(numBlocks, 0.0);
            std::vector<size_t> blockCompressedSizes(numBlocks, 0);
            ProfilingInfo pi_parallel;
            pi_parallel.config_string = configStr;
            omp_set_num_threads(numThreads);
            std::vector<std::vector<std::vector<uint8_t>>> allCompressedBlocks(numBlocks);
            auto compStartOverall = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(numBlocks); i++) {
              auto startTime = std::chrono::high_resolution_clock::now();

              std::vector<std::vector<uint8_t>> compComponents;
              size_t cSize = bzipFusedDecomposedParallel(fullBlocks[i].data, fullBlocks[i].size,
                                                         pi_parallel, compComponents,
                                                         componentConfig, numThreads);
              auto endTime = std::chrono::high_resolution_clock::now();
              blockCompTimes[i] = std::chrono::duration<double>(endTime - startTime).count();
              blockCompressedSizes[i] = cSize;
#pragma omp atomic
              totalCompressedSize += cSize;
              allCompressedBlocks[i] = std::move(compComponents);
            }
            auto compEndOverall = std::chrono::high_resolution_clock::now();
            totalCompTime = std::chrono::duration<double>(compEndOverall - compStartOverall).count();

            // --- Decompression Phase: Use the bzip decomposed parallel decompression function ---
            std::vector<uint8_t> finalReconstructed(totalSize, 0);
            double decompStartOverall = omp_get_wtime();
#pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(numBlocks); i++) {
              // Allocate a temporary vector to hold the decompressed block.
              std::vector<uint8_t> decompBlock(fullBlocks[i].size, 0);
              auto startTime = omp_get_wtime();
              bzipDecomposedParallelDecompression(allCompressedBlocks[i], pi_parallel,
                                                  componentConfig, numThreads,
                                                  fullBlocks[i].size, decompBlock.data());
              auto endTime = omp_get_wtime();
              blockDecompTimes[i] = endTime - startTime;
              // Copy the decompressed block into its proper position.
              std::copy(decompBlock.begin(), decompBlock.end(),
                        finalReconstructed.begin() + i * bs);
            }
            double decompEndOverall = omp_get_wtime();
            totalDecompTime = decompEndOverall - decompStartOverall;

            if (finalReconstructed == globalByteArray)
              std::cout << "[INFO] (bzip) Reconstructed data matches the original (PARALLEL)." << std::endl;
            else
              std::cerr << "[ERROR] (bzip) Reconstructed data does NOT match the original (PARALLEL)." << std::endl;

            double compRatio = calculateCompressionRatio(totalBytes, totalCompressedSize);
            auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(totalBytes, totalCompTime, totalDecompTime);
            outputFile << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << bs << ";"
                       << configStr << ";" << "Chunked_Decompose_Parallel" << ";" << compRatio << ";"
                       << totalCompTime << ";" << totalDecompTime << ";"
                       << compThroughput << ";" << decompThroughput << ";" << totalBytes <<";" << numBlocks << ";" << totalCompressedSize <<  "\n";
          }
        }
        //----------------------------------------
        //--- Decomposed Compression without Chunking (Zstd)
        //---------------------------------------------

        for (const auto& componentConfig : componentConfigurationsList) {
          std::string configStr = configToString1(componentConfig);
          std::cout << "\nConfig with " << componentConfig.size()
                    << " sub-config(s): " << configStr << "\n";
          std::cout << "Testing (Zstd) without chunking." << std::endl;

          size_t totalSize = globalByteArray.size();
          size_t totalCompressedSize = 0;
          double totalCompTime = 0.0, totalDecompTime = 0.0;
          ProfilingInfo pi_nonchunked;
          pi_nonchunked.config_string = configStr;

          std::vector<std::vector<uint8_t>> compComponents;
          auto compStartOverall = std::chrono::high_resolution_clock::now();
          size_t cSize = bzipFusedDecomposedParallel(globalByteArray.data(), totalSize,
                                                     pi_nonchunked, compComponents,
                                                     componentConfig, numThreads);
          auto compEndOverall = std::chrono::high_resolution_clock::now();
          totalCompTime = std::chrono::duration<double>(compEndOverall - compStartOverall).count();
          totalCompressedSize = cSize;

          // --- Decompression Phase: Decompress the entire data ---
          std::vector<uint8_t> finalReconstructed(totalSize, 0);
          auto decompStartOverall = std::chrono::high_resolution_clock::now();
          bzipDecomposedParallelDecompression(compComponents, pi_nonchunked,
                                              componentConfig, numThreads,
                                              totalSize, finalReconstructed.data());
          auto decompEndOverall = std::chrono::high_resolution_clock::now();
          totalDecompTime = std::chrono::duration<double>(decompEndOverall - decompStartOverall).count();

          // Verify that the decompressed data matches the original.
          if (finalReconstructed == globalByteArray)
            std::cout << "[INFO] (bzip) Reconstructed data matches the original (NON-CHUNKED)." << std::endl;
          else
            std::cerr << "[ERROR] (bzip) Reconstructed data does NOT match the original (NON-CHUNKED)." << std::endl;

          // Calculate the compression ratio and throughput.
          double compRatio = calculateCompressionRatio(totalSize, totalCompressedSize);
          auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(totalSize, totalCompTime, totalDecompTime);

          // Record the results (using "N/A" or similar for block size since no chunking is used).
          outputFile << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << "N/A" << ";"
                     << configStr << ";" << "Decompose_NonChunked" << ";" << compRatio << ";"
                     << totalCompTime << ";" << totalDecompTime << ";"
                     << compThroughput << ";" << decompThroughput << ";" << totalBytes <<";" << 1 << ";" << totalCompressedSize <<  "\n";
        }

        // ------------------------------
        // D. Decompose-Then-Chunk Parallel Compression (bzip)
        // ------------------------------
        for (const auto& componentConfig : componentConfigurationsList) {
          std::string configStr = configToString1(componentConfig);
          std::cout << "\nConfig with " << componentConfig.size() << " sub-config(s): " << configStr << "\n";
          for (size_t bs : blockSizes) {
            std::cout << "Testing (bzip) with chunk block size = " << bs << " bytes." << std::endl;
            size_t totalSize = globalByteArray.size();
            ProfilingInfo pi_chunk;
            pi_chunk.config_string = configStr;
            std::vector<std::vector<std::vector<uint8_t>>> compressedBlocks;
            omp_set_num_threads(numThreads);
            double compStartOverall = omp_get_wtime();
            size_t totalCompressedSize = bzipDecomposedThenChunkedParallelCompression(
                globalByteArray.data(), globalByteArray.size(),
                pi_chunk,
                compressedBlocks,
                componentConfig,
                numThreads,
                bs);
            double compEndOverall = omp_get_wtime();
            pi_chunk.total_time_compressed = compEndOverall - compStartOverall;
            std::vector<uint8_t> finalReconstructed(totalSize);
            double decompStartOverall = omp_get_wtime();
            bzipDecomposedThenChunkedParallelDecompression(
                compressedBlocks,
                pi_chunk,
                componentConfig,
                numThreads,
                globalByteArray.size(),
                bs,
                finalReconstructed.data());
            double decompEndOverall = omp_get_wtime();
            pi_chunk.total_time_decompressed = decompEndOverall - decompStartOverall;
            if (finalReconstructed == globalByteArray)
              std::cout << "[INFO] (bzip) Final reconstructed data matches the original." << std::endl;
            else
              std::cerr << "[ERROR] (bzip) Final reconstructed data does NOT match the original." << std::endl;
            double compRatio = calculateCompressionRatio(totalSize, totalCompressedSize);
            auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(
                totalSize, pi_chunk.total_time_compressed, pi_chunk.total_time_decompressed);
            outputFile << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << bs << ";"
                       << configStr << ";" << "Decompose_Chunk_Parallel" << ";" << compRatio << ";"
                       << pi_chunk.total_time_compressed << ";" << pi_chunk.total_time_decompressed << ";"
                       << compThroughput << ";" << decompThroughput << ";" << totalBytes <<";" << -1 << ";" << totalCompressedSize <<  "\n";
          }
        }
      }
      else if (method == "snappy") {
        // ================================
        // Snappy tests
        // ================================
        // [A] Full Compression with Blocking Parallel (Snappy)
        for (size_t bs : blockSizes) {
          std::cout << "Testing (Snappy) with block size = " << bs << " bytes." << std::endl;
          size_t totalSize = globalByteArray.size();
          size_t numBlocks = (totalSize + bs - 1) / bs;
          size_t totalCompressedSize = 0;
          double totalCompTime = 0.0, totalDecompTime = 0.0;
          std::vector<std::vector<uint8_t>> compressedBlocks(numBlocks);
          std::vector<double> blockCompTimes(numBlocks, 0.0);
          std::vector<double> blockDecompTimes(numBlocks, 0.0);
          std::vector<size_t> blockCompressedSizes(numBlocks, 0);
          ProfilingInfo pi_parallel;
          pi_parallel.config_string = "N/A";
          omp_set_num_threads(numThreads);
          auto compStartOverall = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
          for (int i = 0; i < static_cast<int>(numBlocks); i++) {
            size_t start = i * bs;
            size_t end = std::min(start + bs, totalSize);
            std::vector<uint8_t> blockData(globalByteArray.begin() + start, globalByteArray.begin() + end);
            auto startTime = std::chrono::high_resolution_clock::now();
            size_t cSize = compressWithSnappy(blockData, compressedBlocks[i]);
            auto endTime = std::chrono::high_resolution_clock::now();
            blockCompTimes[i] = std::chrono::duration<double>(endTime - startTime).count();
            blockCompressedSizes[i] = cSize;
#pragma omp atomic
            totalCompressedSize += cSize;
          }
          auto compEndOverall = std::chrono::high_resolution_clock::now();
          totalCompTime = std::chrono::duration<double>(compEndOverall - compStartOverall).count();

          std::vector<uint8_t> finalReconstructed(totalSize, 0);
          double decompStartOverall = omp_get_wtime();
#pragma omp parallel for
          for (int i = 0; i < static_cast<int>(numBlocks); i++) {
            size_t start = i * bs;
            size_t end = std::min(start + bs, totalSize);
            size_t blockLength = end - start;
            std::vector<uint8_t> decompBlock;
            double startTime = omp_get_wtime();
            decompressWithSnappy(compressedBlocks[i], decompBlock, blockLength);
            double endTime = omp_get_wtime();
            blockDecompTimes[i] = endTime - startTime;
            std::copy(decompBlock.begin(), decompBlock.end(), finalReconstructed.begin() + start);
          }
          double decompEndOverall = omp_get_wtime();
          totalDecompTime = decompEndOverall - decompStartOverall;
          if (finalReconstructed == globalByteArray)
            std::cout << "[INFO] (Snappy) Reconstructed data matches the original (PARALLEL)." << std::endl;
          else
            std::cerr << "[ERROR] (Snappy) Reconstructed data does NOT match the original (PARALLEL)." << std::endl;
          double compRatio = calculateCompressionRatio(totalBytes, totalCompressedSize);
          auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(totalBytes, totalCompTime, totalDecompTime);
          outputFile << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << bs << ";" << "N/A" << ";"
                     << "Full_Block_Parallel" << ";" << compRatio << ";" << totalCompTime << ";"
                     << totalDecompTime << ";" << compThroughput << ";" << decompThroughput << ";"<< totalBytes <<";" << numBlocks << ";" << totalCompressedSize <<  "\n";
        }

        // [B] Full Compression Without Blocking (Non-blocking) - Snappy
        {
          ProfilingInfo pi_full;
          pi_full.type = "Full Compression (Non-blocking)";
          std::vector<uint8_t> compressedData, decompressedData;
          auto start = std::chrono::high_resolution_clock::now();
          size_t compressedSize = compressWithSnappy(globalByteArray, compressedData);
          auto end = std::chrono::high_resolution_clock::now();
          pi_full.total_time_compressed = std::chrono::duration<double>(end - start).count();

          decompressedData.resize(globalByteArray.size());
          start = std::chrono::high_resolution_clock::now();
          decompressWithSnappy(compressedData, decompressedData, globalByteArray.size());
          end = std::chrono::high_resolution_clock::now();
          pi_full.total_time_decompressed = std::chrono::duration<double>(end - start).count();

          double compRatio = calculateCompressionRatio(totalBytes, compressedSize);
          auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(
              totalBytes, pi_full.total_time_compressed, pi_full.total_time_decompressed);
          outputFile << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << "N/A" << ";" << "N/A" << ";"
                     << "Full" << ";" << compRatio << ";" << pi_full.total_time_compressed << ";"
                     << pi_full.total_time_decompressed << ";" << compThroughput << ";" << decompThroughput << ";" << totalBytes <<";" << 1 << ";" << compressedSize << "\n";
        }

        // [C] Decomposed Compression with Blocking Parallel (Snappy)
        for (const auto& componentConfig : componentConfigurationsList) {
          std::string configStr = configToString1(componentConfig);
          std::cout << "\nConfig with " << componentConfig.size()
                    << " sub-config(s): " << configStr << "\n";
          for (size_t bs : blockSizes) {
            std::cout << "Testing (Snappy) with block size = " << bs << " bytes." << std::endl;
            size_t totalSize = globalByteArray.size();
            size_t numBlocks = (totalSize + bs - 1) / bs;
            struct BlockView { const uint8_t* data; size_t size; };
            std::vector<BlockView> fullBlocks;
            fullBlocks.reserve(numBlocks);
            for (size_t i = 0; i < numBlocks; i++) {
              size_t start = i * bs;
              size_t end = std::min(start + bs, totalSize);
              fullBlocks.push_back({ globalByteArray.data() + start, end - start });
            }
            size_t totalCompressedSize = 0;
            double totalCompTime = 0.0, totalDecompTime = 0.0;
            std::vector<double> blockCompTimes(numBlocks, 0.0);
            std::vector<double> blockDecompTimes(numBlocks, 0.0);
            std::vector<size_t> blockCompressedSizes(numBlocks, 0);
            ProfilingInfo pi_parallel;
            pi_parallel.config_string = configStr;
            omp_set_num_threads(numThreads);
            std::vector<std::vector<std::vector<uint8_t>>> allCompressedBlocks(numBlocks);
            auto compStartOverall = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(numBlocks); i++) {
              auto startTime = std::chrono::high_resolution_clock::now();
              std::vector<std::vector<uint8_t>> compComponents;
              size_t cSize = snappyFusedDecomposedParallel(fullBlocks[i].data, fullBlocks[i].size,
                                                           pi_parallel, compComponents,
                                                           componentConfig, numThreads);
              auto endTime = std::chrono::high_resolution_clock::now();
              blockCompTimes[i] = std::chrono::duration<double>(endTime - startTime).count();
              blockCompressedSizes[i] = cSize;
#pragma omp atomic
              totalCompressedSize += cSize;
              allCompressedBlocks[i] = std::move(compComponents);
            }
            auto compEndOverall = std::chrono::high_resolution_clock::now();
            totalCompTime = std::chrono::duration<double>(compEndOverall - compStartOverall).count();
            std::vector<uint8_t> finalReconstructed(totalSize, 0);
            double decompStartOverall = omp_get_wtime();
#pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(numBlocks); i++) {
              std::vector<uint8_t> decompBlock(fullBlocks[i].size, 0);
              auto startTime = omp_get_wtime();
              snappyDecomposedParallelDecompression(allCompressedBlocks[i], pi_parallel,
                                                    componentConfig, numThreads,
                                                    fullBlocks[i].size, decompBlock.data());
              auto endTime = omp_get_wtime();
              blockDecompTimes[i] = endTime - startTime;
              std::copy(decompBlock.begin(), decompBlock.end(),
                        finalReconstructed.begin() + i * bs);
            }
            double decompEndOverall = omp_get_wtime();
            totalDecompTime = decompEndOverall - decompStartOverall;
            if (finalReconstructed == globalByteArray)
              std::cout << "[INFO] (Snappy) Reconstructed data matches the original (PARALLEL, decomposed)." << std::endl;
            else
              std::cerr << "[ERROR] (Snappy) Reconstructed data does NOT match the original (PARALLEL, decomposed)." << std::endl;
            double compRatio = calculateCompressionRatio(totalBytes, totalCompressedSize);
            auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(totalBytes, totalCompTime, totalDecompTime);
            outputFile << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << bs << ";" << configStr << ";"
                       << "Chunk-decompose_Parallel" << ";" << compRatio << ";" << totalCompTime << ";" << totalDecompTime << ";"
                       << compThroughput << ";" << decompThroughput << ";" << totalBytes <<";" << numBlocks << ";" << totalCompressedSize << "\n";
          }
        }
        // [D] Decomposed Compression without Chunking (Snappy)
        for (const auto& componentConfig : componentConfigurationsList) {
          std::string configStr = configToString1(componentConfig);
          std::cout << "\nConfig with " << componentConfig.size()
                    << " sub-config(s): " << configStr << "\n";
          std::cout << "Testing (Snappy) without chunking." << std::endl;
          size_t totalSize = globalByteArray.size();
          size_t totalCompressedSize = 0;
          double totalCompTime = 0.0, totalDecompTime = 0.0;
          ProfilingInfo pi_nonchunked;
          pi_nonchunked.config_string = configStr;
          std::vector<std::vector<uint8_t>> compComponents; // Per–component compressed data.
          auto compStartOverall = std::chrono::high_resolution_clock::now();
          size_t cSize = snappyFusedDecomposedParallel(globalByteArray.data(), totalSize,
                                                       pi_nonchunked, compComponents,
                                                       componentConfig, numThreads);
          auto compEndOverall = std::chrono::high_resolution_clock::now();
          totalCompTime = std::chrono::duration<double>(compEndOverall - compStartOverall).count();
          totalCompressedSize = cSize;
          std::vector<uint8_t> finalReconstructed(totalSize, 0);
          auto decompStartOverall = std::chrono::high_resolution_clock::now();
          snappyDecomposedParallelDecompression(compComponents, pi_nonchunked,
                                                componentConfig, numThreads,
                                                totalSize, finalReconstructed.data());
          auto decompEndOverall = std::chrono::high_resolution_clock::now();
          totalDecompTime = std::chrono::duration<double>(decompEndOverall - decompStartOverall).count();
          if (finalReconstructed == globalByteArray)
            std::cout << "[INFO] (Snappy) Reconstructed data matches the original (Non-chunked decomposed)." << std::endl;
          else
            std::cerr << "[ERROR] (Snappy) Reconstructed data does NOT match the original (Non-chunked decomposed)." << std::endl;
          double compRatio = calculateCompressionRatio(totalSize, totalCompressedSize);
          auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(totalSize, totalCompTime, totalDecompTime);
          outputFile << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << "N/A" << ";" << configStr << ";"
                     << "Decompose_NonChunked" << ";" << compRatio << ";" << totalCompTime << ";" << totalDecompTime << ";"
                     << compThroughput << ";" << decompThroughput << ";" << totalBytes <<";" << 1 << ";" << totalCompressedSize << "\n";
        }
        // [E] Decomposed Then Chunked Parallel Compression (Snappy)
        for (const auto& componentConfig : componentConfigurationsList) {
          std::string configStr = configToString1(componentConfig);
          std::cout << "\nConfig with " << componentConfig.size() << " sub-config(s): " << configStr << "\n";
          for (size_t bs : blockSizes) {
            std::cout << "Testing (Snappy) with chunk block size = " << bs << " bytes." << std::endl;
            size_t totalSize = globalByteArray.size();
            ProfilingInfo pi_chunk;
            pi_chunk.config_string = configStr;
            std::vector<std::vector<std::vector<uint8_t>>> compressedBlocks;
            omp_set_num_threads(numThreads);
            double compStartOverall = omp_get_wtime();
            size_t totalCompressedSize = snappyDecomposedThenChunkedParallelCompression(
                globalByteArray.data(), globalByteArray.size(),
                pi_chunk,
                compressedBlocks,
                componentConfig,
                numThreads,
                bs);
            double compEndOverall = omp_get_wtime();
            pi_chunk.total_time_compressed = compEndOverall - compStartOverall;
            std::vector<uint8_t> finalReconstructed(totalSize);
            double decompStartOverall = omp_get_wtime();
            snappyDecomposedThenChunkedParallelDecompression(
                compressedBlocks,
                pi_chunk,
                componentConfig,
                numThreads,
                globalByteArray.size(),
                bs,
                finalReconstructed.data());
            double decompEndOverall = omp_get_wtime();
            pi_chunk.total_time_decompressed = decompEndOverall - decompStartOverall;
            if (finalReconstructed == globalByteArray)
              std::cout << "[INFO] (Snappy) Final reconstructed data matches the original." << std::endl;
            else
              std::cerr << "[ERROR] (Snappy) Final reconstructed data does NOT match the original." << std::endl;
            double compRatio = calculateCompressionRatio(totalSize, totalCompressedSize);
            auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(
                totalSize, pi_chunk.total_time_compressed, pi_chunk.total_time_decompressed);
            outputFile << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << bs << ";" << configStr << ";"
                       << "Decompose_Chunk_Parallel" << ";" << compRatio << ";" << pi_chunk.total_time_compressed << ";" << pi_chunk.total_time_decompressed << ";"
                       << compThroughput << ";" << decompThroughput << ";" << totalBytes <<";" << -1 << ";" << totalCompressedSize << "\n";
          }
        }
      }
      else if (method == "zlib") {
         // ------------------------------
        // A. FULL COMPRESSION WITH BLOCKING - PARALLEL (Zlib)
        // ------------------------------
        for (size_t bs : blockSizes) {
          std::cout << "Testing (Zlib) with block size = " << bs << " bytes." << std::endl;
          size_t totalSize = globalByteArray.size();
          size_t numBlocks = (totalSize + bs - 1) / bs;
          size_t totalCompressedSize = 0;
          double totalCompTime = 0.0, totalDecompTime = 0.0;
          std::vector<std::vector<uint8_t>> compressedBlocks(numBlocks);
          std::vector<double> blockCompTimes(numBlocks, 0.0);
          std::vector<double> blockDecompTimes(numBlocks, 0.0);
          std::vector<size_t> blockCompressedSizes(numBlocks, 0);
          ProfilingInfo pi_parallel;
          pi_parallel.config_string = "N/A";
          omp_set_num_threads(numThreads);
          auto compStartOverall = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
          for (int i = 0; i < static_cast<int>(numBlocks); i++) {
            size_t start = i * bs;
            size_t end = std::min(start + bs, totalSize);
            size_t blockLength = end - start;
            std::vector<uint8_t> blockData(globalByteArray.begin() + start, globalByteArray.begin() + end);
            auto startTime = std::chrono::high_resolution_clock::now();
            size_t cSize = compressWithZlib(blockData, compressedBlocks[i], Z_DEFAULT_COMPRESSION);
            auto endTime = std::chrono::high_resolution_clock::now();
            blockCompTimes[i] = std::chrono::duration<double>(endTime - startTime).count();
            blockCompressedSizes[i] = cSize;
#pragma omp atomic
            totalCompressedSize += cSize;
          }
          auto compEndOverall = std::chrono::high_resolution_clock::now();
          totalCompTime = std::chrono::duration<double>(compEndOverall - compStartOverall).count();
          std::vector<uint8_t> finalReconstructed(totalSize, 0);
          double decompStartOverall = omp_get_wtime();
#pragma omp parallel for
          for (int i = 0; i < static_cast<int>(numBlocks); i++) {
            size_t start = i * bs;
            size_t end = std::min(start + bs, totalSize);
            size_t blockLength = end - start;
            std::vector<uint8_t> decompBlock;
            double startTime = omp_get_wtime();
            decompressWithZlib(compressedBlocks[i], decompBlock, blockLength);
            double endTime = omp_get_wtime();
            blockDecompTimes[i] = endTime - startTime;
            std::copy(decompBlock.begin(), decompBlock.end(), finalReconstructed.begin() + start);
          }
          double decompEndOverall = omp_get_wtime();
          totalDecompTime = decompEndOverall - decompStartOverall;
          if (finalReconstructed == globalByteArray)
            std::cout << "[INFO] (Zlib) Reconstructed data matches the original (PARALLEL)." << std::endl;
          else
            std::cerr << "[ERROR] (Zlib) Reconstructed data does NOT match the original (PARALLEL)." << std::endl;
          double compRatio = calculateCompressionRatio(totalBytes, totalCompressedSize);
          auto [compThroughput, decompThroughput] =
              calculateCompDecomThroughput(totalBytes, totalCompTime, totalDecompTime);
          outputFile << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << bs << ";"
                     << "N/A" << ";" << "Chunked_parallel" << ";" << compRatio << ";"
                     << totalCompTime << ";" << totalDecompTime << ";"
                     << compThroughput << ";" << decompThroughput << ";"<< totalBytes <<";" << numBlocks << ";" << totalCompressedSize << "\n";
        }

        // ------------------------------
        // B. FULL COMPRESSION WITHOUT BLOCKING (Non-blocking) using Zlib
        // ------------------------------
        {
          ProfilingInfo pi_full;
          pi_full.type = "Full Compression (Non-blocking)";
          std::vector<uint8_t> compressedData, decompressedData;
          auto start = std::chrono::high_resolution_clock::now();
          size_t compressedSize = zlibCompression(globalByteArray, pi_full, compressedData);
          auto end = std::chrono::high_resolution_clock::now();
          pi_full.total_time_compressed = std::chrono::duration<double>(end - start).count();
          decompressedData.resize(globalByteArray.size());
          start = std::chrono::high_resolution_clock::now();
          zlibDecompression(compressedData, decompressedData, pi_full);
          end = std::chrono::high_resolution_clock::now();
          pi_full.total_time_decompressed = std::chrono::duration<double>(end - start).count();
          double compRatio = calculateCompressionRatio(totalBytes, compressedSize);
          auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(
              totalBytes, pi_full.total_time_compressed, pi_full.total_time_decompressed);
          outputFile << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << "N/A" << ";" << "N/A" << ";"
                     << "Full" << ";" << compRatio << ";" << pi_full.total_time_compressed << ";"
                     << pi_full.total_time_decompressed << ";" << compThroughput << ";" << decompThroughput
                     << ";" << totalBytes <<";" << 1 << ";" << compressedSize << "\n";
        }

        // ------------------------------
        // C. Decomposed Compression with Blocking Parallel (Zlib)
        // ------------------------------
        for (const auto& componentConfig : componentConfigurationsList) {
          std::string configStr = configToString1(componentConfig);
          std::cout << "\nConfig with " << componentConfig.size()
                    << " sub-config(s): " << configStr << "\n";
          for (size_t bs : blockSizes) {
            std::cout << "Testing (Zlib) with block size = " << bs << " bytes." << std::endl;
            size_t totalSize = globalByteArray.size();
            size_t numBlocks = (totalSize + bs - 1) / bs;

            struct BlockView { const uint8_t* data; size_t size; };
            std::vector<BlockView> fullBlocks;
            fullBlocks.reserve(numBlocks);
            for (size_t i = 0; i < numBlocks; i++) {
              size_t start = i * bs;
              size_t end = std::min(start + bs, totalSize);
              fullBlocks.push_back({ globalByteArray.data() + start, end - start });
            }

            size_t totalCompressedSize = 0;
            double totalCompTime = 0.0, totalDecompTime = 0.0;
            std::vector<double> blockCompTimes(numBlocks, 0.0);
            std::vector<double> blockDecompTimes(numBlocks, 0.0);
            std::vector<size_t> blockCompressedSizes(numBlocks, 0);
            ProfilingInfo pi_parallel;
            pi_parallel.config_string = configStr;
            omp_set_num_threads(numThreads);

            std::vector<std::vector<std::vector<uint8_t>>> allCompressedBlocks(numBlocks);

            auto compStartOverall = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(numBlocks); i++) {
              auto startTime = std::chrono::high_resolution_clock::now();
              std::vector<std::vector<uint8_t>> compComponents;
              size_t cSize = zlibFusedDecomposedParallel(fullBlocks[i].data, fullBlocks[i].size,
                                                         pi_parallel, compComponents,
                                                         componentConfig, numThreads);
              auto endTime = std::chrono::high_resolution_clock::now();
              blockCompTimes[i] = std::chrono::duration<double>(endTime - startTime).count();
              blockCompressedSizes[i] = cSize;
#pragma omp atomic
              totalCompressedSize += cSize;
              allCompressedBlocks[i] = std::move(compComponents);
            }
            auto compEndOverall = std::chrono::high_resolution_clock::now();
            totalCompTime = std::chrono::duration<double>(compEndOverall - compStartOverall).count();

            std::vector<uint8_t> finalReconstructed(totalSize, 0);
            double decompStartOverall = omp_get_wtime();
#pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(numBlocks); i++) {
              // Allocate a temporary vector to hold the decompressed block.
              std::vector<uint8_t> decompBlock(fullBlocks[i].size, 0);
              auto startTime = omp_get_wtime();
              zlibDecomposedParallelDecompression(allCompressedBlocks[i], pi_parallel,
                                                  componentConfig, numThreads,
                                                  fullBlocks[i].size, decompBlock.data());
              auto endTime = omp_get_wtime();
              blockDecompTimes[i] = endTime - startTime;
              std::copy(decompBlock.begin(), decompBlock.end(),
                        finalReconstructed.begin() + i * bs);
            }
            double decompEndOverall = omp_get_wtime();
            totalDecompTime = decompEndOverall - decompStartOverall;

            if (finalReconstructed == globalByteArray)
              std::cout << "[INFO] (Zlib) Reconstructed data matches the original (PARALLEL)." << std::endl;
            else
              std::cerr << "[ERROR] (Zlib) Reconstructed data does NOT match the original (PARALLEL)." << std::endl;

            double compRatio = calculateCompressionRatio(totalBytes, totalCompressedSize);
            auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(totalBytes, totalCompTime, totalDecompTime);
            outputFile << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << bs << ";"
                       << configStr << ";" << "Chunked_Decompose_Parallel" << ";" << compRatio << ";"
                       << totalCompTime << ";" << totalDecompTime << ";"
                       << compThroughput << ";" << decompThroughput << ";"<< totalBytes <<";" << numBlocks << ";" << totalCompressedSize << "\n";
          }
        }

        // ------------------------------
        // D. Decomposed Compression without Chunking (Zlib)
        // ------------------------------
        for (const auto& componentConfig : componentConfigurationsList) {
          std::string configStr = configToString1(componentConfig);
          std::cout << "\nConfig with " << componentConfig.size()
                    << " sub-config(s): " << configStr << "\n";
          std::cout << "Testing (Zlib) without chunking." << std::endl;

          size_t totalSize = globalByteArray.size();
          size_t totalCompressedSize = 0;
          double totalCompTime = 0.0, totalDecompTime = 0.0;

          // Create a separate ProfilingInfo structure for the non-chunked test.
          ProfilingInfo pi_nonchunked;
          pi_nonchunked.config_string = configStr;

          // --- Compression Phase: Compress the entire data in one go ---
          std::vector<std::vector<uint8_t>> compComponents;
          auto compStartOverall = std::chrono::high_resolution_clock::now();
          size_t cSize = zlibFusedDecomposedParallel(globalByteArray.data(), totalSize,
                                                     pi_nonchunked, compComponents,
                                                     componentConfig, numThreads);
          auto compEndOverall = std::chrono::high_resolution_clock::now();
          totalCompTime = std::chrono::duration<double>(compEndOverall - compStartOverall).count();
          totalCompressedSize = cSize;

          // --- Decompression Phase: Decompress the entire data ---
          std::vector<uint8_t> finalReconstructed(totalSize, 0);
          auto decompStartOverall = std::chrono::high_resolution_clock::now();
          zlibDecomposedParallelDecompression(compComponents, pi_nonchunked,
                                              componentConfig, numThreads,
                                              totalSize, finalReconstructed.data());
          auto decompEndOverall = std::chrono::high_resolution_clock::now();
          totalDecompTime = std::chrono::duration<double>(decompEndOverall - decompStartOverall).count();

          // Verify that the decompressed data matches the original.
          if (finalReconstructed == globalByteArray)
            std::cout << "[INFO] (Zlib) Reconstructed data matches the original (NON-CHUNKED)." << std::endl;
          else
            std::cerr << "[ERROR] (Zlib) Reconstructed data does NOT match the original (NON-CHUNKED)." << std::endl;

          double compRatio = calculateCompressionRatio(totalSize, totalCompressedSize);
          auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(totalSize, totalCompTime, totalDecompTime);
          outputFile << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << "N/A" << ";"
                     << configStr << ";" << "Decompose_NonChunked" << ";" << compRatio << ";"
                     << totalCompTime << ";" << totalDecompTime << ";"
                     << compThroughput << ";" << decompThroughput << ";" << totalBytes <<";" << 1 << ";" << totalCompressedSize << "\n";
        }

        // ------------------------------
        // E. Decompose-Then-Chunk Parallel Compression (Zlib)
        // ------------------------------
        for (const auto& componentConfig : componentConfigurationsList) {
          std::string configStr = configToString1(componentConfig);
          std::cout << "\nConfig with " << componentConfig.size() << " sub-config(s): " << configStr << "\n";
          for (size_t bs : blockSizes) {
            std::cout << "Testing (Zlib) with chunk block size = " << bs << " bytes." << std::endl;
            size_t totalSize = globalByteArray.size();
            ProfilingInfo pi_chunk;
            pi_chunk.config_string = configStr;
            std::vector<std::vector<std::vector<uint8_t>>> compressedBlocks;
            omp_set_num_threads(numThreads);
            double compStartOverall = omp_get_wtime();
            size_t totalCompressedSize = zlibDecomposedThenChunkedParallelCompression(
                globalByteArray.data(), globalByteArray.size(),
                pi_chunk,
                compressedBlocks,
                componentConfig,
                numThreads,
                bs);
            double compEndOverall = omp_get_wtime();
            pi_chunk.total_time_compressed = compEndOverall - compStartOverall;
            std::vector<uint8_t> finalReconstructed(totalSize, 0);
            double decompStartOverall = omp_get_wtime();
            zlibDecomposedThenChunkedParallelDecompression(
                compressedBlocks,
                pi_chunk,
                componentConfig,
                numThreads,
                globalByteArray.size(),
                bs,
                finalReconstructed.data());
            double decompEndOverall = omp_get_wtime();
            pi_chunk.total_time_decompressed = decompEndOverall - decompStartOverall;
            if (finalReconstructed == globalByteArray)
              std::cout << "[INFO] (Zlib) Final reconstructed data matches the original." << std::endl;
            else
              std::cerr << "[ERROR] (Zlib) Final reconstructed data does NOT match the original." << std::endl;
            double compRatio = calculateCompressionRatio(totalSize, totalCompressedSize);
            auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(
                totalSize, pi_chunk.total_time_compressed, pi_chunk.total_time_decompressed);
            outputFile << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << bs << ";"
                       << configStr << ";" << "Decompose_Chunk_Parallel" << ";" << compRatio << ";"
                       << pi_chunk.total_time_compressed << ";" << pi_chunk.total_time_decompressed << ";"
                       << compThroughput << ";" << decompThroughput << ";" << totalBytes <<";" <<-1 << ";" << totalCompressedSize << "\n";
          }
        }
      }
      else if (method == "lz4") {
         // ------------------------------
        // A. FULL COMPRESSION WITH BLOCKING - PARALLEL (LZ4)
        // ------------------------------
        for (size_t bs : blockSizes) {
          std::cout << "Testing (LZ4) with block size = " << bs << " bytes." << std::endl;
          size_t totalSize = globalByteArray.size();
          size_t numBlocks = (totalSize + bs - 1) / bs;
          size_t totalCompressedSize = 0;
          double totalCompTime = 0.0, totalDecompTime = 0.0;
          std::vector<std::vector<uint8_t>> compressedBlocks(numBlocks);
          std::vector<double> blockCompTimes(numBlocks, 0.0);
          std::vector<double> blockDecompTimes(numBlocks, 0.0);
          std::vector<size_t> blockCompressedSizes(numBlocks, 0);
          ProfilingInfo pi_parallel;
          pi_parallel.config_string = "N/A";
          omp_set_num_threads(numThreads);
          auto compStartOverall = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
          for (int i = 0; i < static_cast<int>(numBlocks); i++) {
            size_t start = i * bs;
            size_t end = std::min(start + bs, totalSize);
            size_t blockLength = end - start;
            std::vector<uint8_t> blockData(globalByteArray.begin() + start, globalByteArray.begin() + end);
            auto startTime = std::chrono::high_resolution_clock::now();
            size_t cSize = compressWithLZ4(blockData, compressedBlocks[i], 3);
            auto endTime = std::chrono::high_resolution_clock::now();
            blockCompTimes[i] = std::chrono::duration<double>(endTime - startTime).count();
            blockCompressedSizes[i] = cSize;
#pragma omp atomic
            totalCompressedSize += cSize;
          }
          auto compEndOverall = std::chrono::high_resolution_clock::now();
          totalCompTime = std::chrono::duration<double>(compEndOverall - compStartOverall).count();
          std::vector<uint8_t> finalReconstructed(totalSize, 0);
          double decompStartOverall = omp_get_wtime();
#pragma omp parallel for
          for (int i = 0; i < static_cast<int>(numBlocks); i++) {
            size_t start = i * bs;
            size_t end = std::min(start + bs, totalSize);
            size_t blockLength = end - start;
            std::vector<uint8_t> decompBlock;
            double startTime = omp_get_wtime();
            decompressWithLZ4(compressedBlocks[i], decompBlock, blockLength);
            double endTime = omp_get_wtime();
            blockDecompTimes[i] = endTime - startTime;
            std::copy(decompBlock.begin(), decompBlock.end(), finalReconstructed.begin() + start);
          }
          double decompEndOverall = omp_get_wtime();
          totalDecompTime = decompEndOverall - decompStartOverall;
          if (finalReconstructed == globalByteArray)
            std::cout << "[INFO] (LZ4) Reconstructed data matches the original (PARALLEL)." << std::endl;
          else
            std::cerr << "[ERROR] (LZ4) Reconstructed data does NOT match the original (PARALLEL)." << std::endl;
          double compRatio = calculateCompressionRatio(totalBytes, totalCompressedSize);
          auto [compThroughput, decompThroughput] =
              calculateCompDecomThroughput(totalBytes, totalCompTime, totalDecompTime);
          outputFile << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << bs << ";"
                     << "N/A" << ";" << "Chunked_parallel" << ";" << compRatio << ";"
                     << totalCompTime << ";" << totalDecompTime << ";"
                     << compThroughput << ";" << decompThroughput << ";" << totalBytes <<";" << numBlocks << ";" << totalCompressedSize << "\n";
        }

        // ------------------------------
        // B. FULL COMPRESSION WITHOUT BLOCKING (Non-blocking) using LZ4
        // ------------------------------
        {
          ProfilingInfo pi_full;
          pi_full.type = "Full Compression (Non-blocking)";
          std::vector<uint8_t> compressedData, decompressedData;
          auto start = std::chrono::high_resolution_clock::now();
          size_t compressedSize = lz4Compression(globalByteArray, pi_full, compressedData);
          auto end = std::chrono::high_resolution_clock::now();
          pi_full.total_time_compressed = std::chrono::duration<double>(end - start).count();
          decompressedData.resize(globalByteArray.size());
          start = std::chrono::high_resolution_clock::now();
          lz4Decompression(compressedData, decompressedData, pi_full);
          end = std::chrono::high_resolution_clock::now();
          pi_full.total_time_decompressed = std::chrono::duration<double>(end - start).count();
          double compRatio = calculateCompressionRatio(totalBytes, compressedSize);
          auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(
              totalBytes, pi_full.total_time_compressed, pi_full.total_time_decompressed);
          outputFile << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << "N/A" << ";" << "N/A" << ";"
                     << "Full" << ";" << compRatio << ";" << pi_full.total_time_compressed << ";"
                     << pi_full.total_time_decompressed << ";" << compThroughput << ";" << decompThroughput
                     << ";" << rowCount << "\n";
        }

        // ------------------------------
        // C. Decomposed Compression with Blocking Parallel (LZ4)
        // ------------------------------
        for (const auto& componentConfig : componentConfigurationsList) {
          std::string configStr = configToString1(componentConfig);
          std::cout << "\nConfig with " << componentConfig.size()
                    << " sub-config(s): " << configStr << "\n";
          for (size_t bs : blockSizes) {
            std::cout << "Testing (LZ4) with block size = " << bs << " bytes." << std::endl;
            size_t totalSize = globalByteArray.size();
            size_t numBlocks = (totalSize + bs - 1) / bs;

            // Prepare a vector of block views.
            struct BlockView { const uint8_t* data; size_t size; };
            std::vector<BlockView> fullBlocks;
            fullBlocks.reserve(numBlocks);
            for (size_t i = 0; i < numBlocks; i++) {
              size_t start = i * bs;
              size_t end = std::min(start + bs, totalSize);
              fullBlocks.push_back({ globalByteArray.data() + start, end - start });
            }

            size_t totalCompressedSize = 0;
            double totalCompTime = 0.0, totalDecompTime = 0.0;
            std::vector<double> blockCompTimes(numBlocks, 0.0);
            std::vector<double> blockDecompTimes(numBlocks, 0.0);
            std::vector<size_t> blockCompressedSizes(numBlocks, 0);
            ProfilingInfo pi_parallel;
            pi_parallel.config_string = configStr;
            omp_set_num_threads(numThreads);

            std::vector<std::vector<std::vector<uint8_t>>> allCompressedBlocks(numBlocks);

            auto compStartOverall = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(numBlocks); i++) {
              auto startTime = std::chrono::high_resolution_clock::now();
              std::vector<std::vector<uint8_t>> compComponents;
              size_t cSize = lz4FusedDecomposedParallel(fullBlocks[i].data, fullBlocks[i].size,
                                                        pi_parallel, compComponents,
                                                        componentConfig, numThreads);
              auto endTime = std::chrono::high_resolution_clock::now();
              blockCompTimes[i] = std::chrono::duration<double>(endTime - startTime).count();
              blockCompressedSizes[i] = cSize;
#pragma omp atomic
              totalCompressedSize += cSize;
              allCompressedBlocks[i] = std::move(compComponents);
            }
            auto compEndOverall = std::chrono::high_resolution_clock::now();
            totalCompTime = std::chrono::duration<double>(compEndOverall - compStartOverall).count();

            std::vector<uint8_t> finalReconstructed(totalSize, 0);
            double decompStartOverall = omp_get_wtime();
#pragma omp parallel for schedule(static)
            for (int i = 0; i < static_cast<int>(numBlocks); i++) {
              // Allocate a temporary vector to hold the decompressed block.
              std::vector<uint8_t> decompBlock(fullBlocks[i].size, 0);
              auto startTime = omp_get_wtime();
              lz4DecomposedParallelDecompression(allCompressedBlocks[i], pi_parallel,
                                                 componentConfig, numThreads,
                                                 fullBlocks[i].size, decompBlock.data());
              auto endTime = omp_get_wtime();
              blockDecompTimes[i] = endTime - startTime;
              std::copy(decompBlock.begin(), decompBlock.end(),
                        finalReconstructed.begin() + i * bs);
            }
            double decompEndOverall = omp_get_wtime();
            totalDecompTime = decompEndOverall - decompStartOverall;

            if (finalReconstructed == globalByteArray)
              std::cout << "[INFO] (LZ4) Reconstructed lz4 data matches the original (PARALLEL)." << std::endl;
            else
              std::cerr << "[ERROR] (LZ4) Reconstructed data does NOT match the original (PARALLEL)." << std::endl;

            double compRatio = calculateCompressionRatio(totalBytes, totalCompressedSize);
            auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(totalBytes, totalCompTime, totalDecompTime);
            outputFile << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << bs << ";"
                       << configStr << ";" << "Chunked_Decompose_Parallel" << ";" << compRatio << ";"
                       << totalCompTime << ";" << totalDecompTime << ";"
                       << compThroughput << ";" << decompThroughput << ";" << totalBytes <<";" << numBlocks << ";" << totalCompressedSize << "\n";
          }
        }

        // ------------------------------
        // D. Decomposed Compression without Chunking (LZ4)
        // ------------------------------
        for (const auto& componentConfig : componentConfigurationsList) {
          std::string configStr = configToString1(componentConfig);
          std::cout << "\nConfig with " << componentConfig.size()
                    << " sub-config(s): " << configStr << "\n";
          std::cout << "Testing (LZ4) without chunking." << std::endl;

          size_t totalSize = globalByteArray.size();
          size_t totalCompressedSize = 0;
          double totalCompTime = 0.0, totalDecompTime = 0.0;

          ProfilingInfo pi_nonchunked;
          pi_nonchunked.config_string = configStr;

          std::vector<std::vector<uint8_t>> compComponents; // Per–component compressed data.
          auto compStartOverall = std::chrono::high_resolution_clock::now();
          size_t cSize = lz4FusedDecomposedParallel(globalByteArray.data(), totalSize,
                                                    pi_nonchunked, compComponents,
                                                    componentConfig, numThreads);
          auto compEndOverall = std::chrono::high_resolution_clock::now();
          totalCompTime = std::chrono::duration<double>(compEndOverall - compStartOverall).count();
          totalCompressedSize = cSize;

          std::vector<uint8_t> finalReconstructed(totalSize, 0);
          auto decompStartOverall = std::chrono::high_resolution_clock::now();
          lz4DecomposedParallelDecompression(compComponents, pi_nonchunked,
                                             componentConfig, numThreads,
                                             totalSize, finalReconstructed.data());
          auto decompEndOverall = std::chrono::high_resolution_clock::now();
          totalDecompTime = std::chrono::duration<double>(decompEndOverall - decompStartOverall).count();

          // Verify that the decompressed data matches the original.
          if (finalReconstructed == globalByteArray)
            std::cout << "[INFO] (LZ4) Reconstructed data matches the original (NON-CHUNKED)." << std::endl;
          else
            std::cerr << "[ERROR] (LZ4) Reconstructed data does NOT match the original (NON-CHUNKED)." << std::endl;

          double compRatio = calculateCompressionRatio(totalSize, totalCompressedSize);
          auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(totalSize, totalCompTime, totalDecompTime);
          outputFile << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << "N/A" << ";"
                     << configStr << ";" << "Decompose_NonChunked" << ";" << compRatio << ";"
                     << totalCompTime << ";" << totalDecompTime << ";"
                     << compThroughput << ";" << decompThroughput << ";" << totalBytes <<";" << 1 << ";" << totalCompressedSize <<"\n";
        }

        // ------------------------------
        // E. Decompose-Then-Chunk Parallel Compression (LZ4)
        // ------------------------------
        for (const auto& componentConfig : componentConfigurationsList) {
          std::string configStr = configToString1(componentConfig);
          std::cout << "\nConfig with " << componentConfig.size() << " sub-config(s): " << configStr << "\n";
          for (size_t bs : blockSizes) {
            std::cout << "Testing (LZ4) with chunk block size = " << bs << " bytes." << std::endl;
            size_t totalSize = globalByteArray.size();
            ProfilingInfo pi_chunk;
            pi_chunk.config_string = configStr;
            std::vector<std::vector<std::vector<uint8_t>>> compressedBlocks;
            omp_set_num_threads(numThreads);
            double compStartOverall = omp_get_wtime();
            size_t totalCompressedSize = lz4DecomposedThenChunkedParallelCompression(
                globalByteArray.data(), globalByteArray.size(),
                pi_chunk,
                compressedBlocks,
                componentConfig,
                numThreads,
                bs);
            double compEndOverall = omp_get_wtime();
            pi_chunk.total_time_compressed = compEndOverall - compStartOverall;
            std::vector<uint8_t> finalReconstructed(totalSize, 0);
            double decompStartOverall = omp_get_wtime();
            lz4DecomposedThenChunkedParallelDecompression(
                compressedBlocks,
                pi_chunk,
                componentConfig,
                numThreads,
                globalByteArray.size(),
                bs,
                finalReconstructed.data());
            double decompEndOverall = omp_get_wtime();
            pi_chunk.total_time_decompressed = decompEndOverall - decompStartOverall;
            if (finalReconstructed == globalByteArray)
              std::cout << "[INFO] (LZ4) Final reconstructed data matches the original." << std::endl;
            else
              std::cerr << "[ERROR] (LZ4) Final reconstructed data does NOT match the original." << std::endl;
            double compRatio = calculateCompressionRatio(totalSize, totalCompressedSize);
            auto [compThroughput, decompThroughput] = calculateCompDecomThroughput(
                totalSize, pi_chunk.total_time_compressed, pi_chunk.total_time_decompressed);
            outputFile << recordIndex++ << ";" << datasetName << ";" << numThreads << ";" << bs << ";"
                       << configStr << ";" << "Decompose_Chunk_Parallel" << ";" << compRatio << ";"
                       << pi_chunk.total_time_compressed << ";" << pi_chunk.total_time_decompressed << ";"
                       << compThroughput << ";" << decompThroughput << ";"<< totalBytes <<";" << -1 << ";" << totalCompressedSize <<"\n";
          }
        }
      }

    }
  }


  outputFile.close();
  std::cout << "Profiling results saved to " << outputCSV << "\n";
  return 0;
}
