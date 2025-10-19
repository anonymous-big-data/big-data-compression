
#ifndef BZ2_PARALLEL_H
#define BZ2_PARALLEL_H

#include <vector>
#include <iostream>
#include <cstring>
#include <bzlib.h>
#include "profiling_info.h"
#include <omp.h>
#include <numeric>


// Declare globalByteArray as an external variable
extern std::vector<uint8_t> globalByteArray;
size_t compressWithbzip2(const std::vector<uint8_t>& data, std::vector<uint8_t>& compressedData, int compressionLevel);

size_t decompressWithbzip2(const std::vector<uint8_t>& compressedData, std::vector<uint8_t>& decompressedData, size_t originalSize);

// Verify if original and reconstructed data match
bool verifyDataMatch(const std::vector<uint8_t>& original, const std::vector<uint8_t>& reconstructed) {
  if (original.size() != reconstructed.size()) {
    std::cerr << "Size mismatch: Original size = " << original.size() << ", Reconstructed size = " << reconstructed.size() << std::endl;
    return false;
  }

  for (size_t i = 0; i < original.size(); i++) {
    if (original[i] != reconstructed[i]) {
      std::cerr << "Data mismatch at index " << i << ": Original = " << static_cast<int>(original[i]) << ", Reconstructed = " << static_cast<int>(reconstructed[i]) << std::endl;
      return false;
    }
  }
  return true;
}
void splitBytesIntoComponents(const std::vector<uint8_t>& byteArray,
                              std::vector<std::vector<uint8_t>>& components,
                              const std::vector<size_t>& componentSizes,int numThreads) {
  size_t numComponents = componentSizes.size();
  size_t totalBytes = std::accumulate(componentSizes.begin(), componentSizes.end(), 0);
  size_t numElements = byteArray.size() / totalBytes;

  // Resize components to hold the split data
  components.resize(numComponents);
  for (size_t i = 0; i < numComponents; ++i) {
    components[i].resize(numElements * componentSizes[i]);
  }

  // Use OpenMP to parallelize the component processing
#pragma omp parallel for num_threads(numThreads)
  for (size_t i = 0; i < numComponents; ++i) {
    size_t offset = std::accumulate(componentSizes.begin(), componentSizes.begin() + i, 0);
    for (size_t j = 0; j < numElements; ++j) {
      std::copy(byteArray.begin() + j * totalBytes + offset,
                byteArray.begin() + j * totalBytes + offset + componentSizes[i],
                components[i].begin() + j * componentSizes[i]);
    }
  }
}
// Compress with BZip2
size_t compressWithbzip2(const std::vector<uint8_t>& data, std::vector<uint8_t>& compressedData, int compressionLevel) {
  if (compressionLevel < 1 || compressionLevel > 9) {
    compressionLevel = 9; // Default to maximum compression
  }

  unsigned int maxCompressedSize = data.size() * 1.01 + 600;
  compressedData.resize(maxCompressedSize);

  unsigned int compressedSize = maxCompressedSize;
  int result = BZ2_bzBuffToBuffCompress(
      reinterpret_cast<char*>(compressedData.data()),
      &compressedSize,
      const_cast<char*>(reinterpret_cast<const char*>(data.data())), // Fix: Use const_cast
      data.size(),
      compressionLevel,
      0,
      30
  );

  if (result != BZ_OK) {
    std::cerr << "BZip2 compression error: " << result << std::endl;
    return 0;
  }

  compressedData.resize(compressedSize);
  return compressedSize;
}

// Decompress with BZip2
size_t decompressWithbzip2(const std::vector<uint8_t>& compressedData, std::vector<uint8_t>& decompressedData, size_t originalSize) {
  decompressedData.resize(originalSize);

  unsigned int decompressedSize = originalSize;
  int result = BZ2_bzBuffToBuffDecompress(
      reinterpret_cast<char*>(decompressedData.data()), // Output buffer
      &decompressedSize,
      const_cast<char*>(reinterpret_cast<const char*>(compressedData.data())), // Fix: Use const_cast
      compressedData.size(),                           // Compressed size
      0,                                               // Verbosity
      0                                                // Small mode (set to 0 for normal decompression)
  );

  if (result != BZ_OK) {
    std::cerr << "BZip2 decompression error: " << result << std::endl;
    return 0;
  }

  decompressedData.resize(decompressedSize);
  return decompressedSize;
}

// Full compression without decomposition
size_t bzibCompression(const std::vector<uint8_t>& data, ProfilingInfo &pi, std::vector<uint8_t>& compressedData) {

    size_t compressedSize = compressWithbzip2(data, compressedData, 3);


    pi.type = "LZ4 Full Compression";

    return compressedSize;
}

// Full decompression without decomposition
void bzibDecompression(const std::vector<uint8_t>& compressedData, std::vector<uint8_t>& decompressedData, ProfilingInfo &pi, size_t originalSize) {

    decompressWithbzip2(compressedData, decompressedData, originalSize);

  // Verify decompressed data
  if (!verifyDataMatch(globalByteArray, decompressedData)) {
    std::cerr << "Error: Decompressed data doesn't match the original." << std::endl;
  }

}
// Sequential compression with decomposition that takes dynamic byte sizes as parameters
size_t bzipDecomposedSequential(const std::vector<uint8_t>& data, ProfilingInfo &pi,
                                std::vector<std::vector<uint8_t>>& compressedComponents,
                                const std::vector<size_t>& componentSizes) {
  // Split data into components
  std::vector<std::vector<uint8_t>> components(componentSizes.size());
  splitBytesIntoComponents(data, components, componentSizes,1);

  size_t compressedSizeTotal = 0;

  pi.component_times.assign(componentSizes.size(), 0.0);

  // Compress each component sequentially and record the compression time
  for (size_t i = 0; i < componentSizes.size(); ++i) {
      compressedSizeTotal += compressWithbzip2(components[i], compressedComponents[i], 3);
    }
  return compressedSizeTotal;

}

// Sequential decompression with decomposition
void bzibDecomposedSequentialDecompression(const std::vector<std::vector<uint8_t>>& compressedComponents,
                                           ProfilingInfo &pi,
                                           const std::vector<size_t>& componentBytes) {
  size_t totalSize = globalByteArray.size();
  size_t numComponents = componentBytes.size();
  size_t totalBytesPerElement = std::accumulate(componentBytes.begin(), componentBytes.end(), 0);
  size_t floatCount = totalSize / totalBytesPerElement;
  std::vector<uint8_t> reconstructedData(totalSize);

  size_t baseOffset = 0;

  for (size_t compIdx = 0; compIdx < numComponents; ++compIdx) {

    std::vector<uint8_t> tempComponent(floatCount * componentBytes[compIdx]);

    // Decompress the current component

    decompressWithbzip2(compressedComponents[compIdx], tempComponent, floatCount * componentBytes[compIdx]);


    // Reassemble the decompressed data
    for (size_t i = 0; i < floatCount; ++i) {
      std::copy(tempComponent.begin() + i * componentBytes[compIdx],
                tempComponent.begin() + (i + 1) * componentBytes[compIdx],
                reconstructedData.begin() + i * totalBytesPerElement + baseOffset);
    }

    baseOffset += componentBytes[compIdx];
  }

   // Verify decompressed data
   if (!verifyDataMatch(globalByteArray, reconstructedData)) {
     std::cerr << "Error: Decompressed data doesn't match the original." << std::endl;
   }
  }
// Parallel compression with decomposition

  size_t bzibDecomposedParallel(const std::vector<uint8_t>& data, ProfilingInfo &pi,
                                std::vector<std::vector<uint8_t>>& compressedComponents,
                                const std::vector<size_t>& componentSizes, int numThreads) {
    // Split data into components
    std::vector<std::vector<uint8_t>> components(componentSizes.size());
    splitBytesIntoComponents(data, components, componentSizes, numThreads);

    // omp_set_num_threads(numThreads);

    size_t compressedSizeTotal = 0;

    pi.component_times.assign(componentSizes.size(), 0.0);

    // #pragma omp parallel  for num_threads(numThreads)
#pragma omp parallel for schedule(dynamic) num_threads(numThreads)
    for (size_t i = 0; i < componentSizes.size(); ++i) {

      compressedSizeTotal += compressWithbzip2(components[i], compressedComponents[i], 3);


    }

    return compressedSizeTotal;
  }

std::vector<uint8_t> bzibDecomposedParallelDecompression(
    const std::vector<std::vector<uint8_t>>& compressedComponents,
    ProfilingInfo& pi,
    const std::vector<size_t>& componentBytes,
    int numThreads) {
  size_t totalSize = globalByteArray.size(); // Size of the original data
  size_t numComponents = componentBytes.size();
  size_t totalBytesPerElement = std::accumulate(componentBytes.begin(), componentBytes.end(), 0);
  size_t floatCount = totalSize / totalBytesPerElement;
  std::vector<uint8_t> reconstructedData(totalSize);

  pi.component_times.resize(numComponents);
  omp_set_num_threads(numThreads);

#pragma omp parallel for
  for (size_t compIdx = 0; compIdx < numComponents; ++compIdx) {


    // Temporary buffer for the decompressed component
    std::vector<uint8_t> tempComponent(floatCount * componentBytes[compIdx]);

    // Decompress the current component
    decompressWithbzip2(compressedComponents[compIdx], tempComponent, floatCount * componentBytes[compIdx]);

    // Calculate the base offset for this component
    size_t baseOffset = std::accumulate(componentBytes.begin(), componentBytes.begin() + compIdx, 0);

    // Reassemble the decompressed data with unrolling
    size_t comByteComp = componentBytes[compIdx];

#pragma omp parallel for schedule(static, 100000)
    for (size_t i = 0; i < floatCount; ++i) {
      size_t baseIndex = i * totalBytesPerElement + baseOffset;
      size_t tempIndex = i * comByteComp;

#pragma omp simd
      for (size_t j = 0; j < comByteComp; ++j) {
        reconstructedData[baseIndex + j] = tempComponent[tempIndex + j];
      }
    }


  }

   // Verify decompressed data
   if (!verifyDataMatch(globalByteArray, reconstructedData)) {
     std::cerr << "Error: Decompressed data doesn't match the original." << std::endl;
   }

  return reconstructedData;
}

double calculateCompressionRatio(size_t originalSize, size_t compressedSize) {
  return static_cast<double>(originalSize) / static_cast<double>(compressedSize);
}



#endif //BZ2_PARALLEL_H

