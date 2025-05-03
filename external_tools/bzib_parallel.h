#ifndef BZIB_PARALLEL_H
#define BZIB_PARALLEL_H

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
bool verifyDataMatch2(const std::vector<uint8_t>& original, const std::vector<uint8_t>& reconstructed) {
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
///////////////////////
// Splitting function: decomposes full (interleaved) data into its components.
inline void splitBytesIntoComponentsNested2(
    const std::vector<uint8_t>& byteArray,
    std::vector<std::vector<uint8_t>>& outputComponents,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads
) {
    size_t totalBytesPerElement = 0;
    for (const auto& group : allComponentSizes) {
        totalBytesPerElement += group.size();
    }
    size_t numElements = byteArray.size() / totalBytesPerElement;
    outputComponents.resize(allComponentSizes.size());

    // Make a temporary copy (if needed)
    std::vector<uint8_t> temp(byteArray);

    // Allocate space for each component.
    for (size_t i = 0; i < allComponentSizes.size(); i++) {
        outputComponents[i].resize(numElements * allComponentSizes[i].size());
    }

    #pragma omp parallel for num_threads(numThreads)
    for (size_t elem = 0; elem < numElements; elem++) {
        for (size_t compIdx = 0; compIdx < allComponentSizes.size(); compIdx++) {
            const auto& groupIndices = allComponentSizes[compIdx];
            size_t groupSize = groupIndices.size();
            size_t writePos = elem * groupSize;
            for (size_t sub = 0; sub < groupSize; sub++) {
                // Adjust from 1-based index if necessary (here we subtract 1)
                size_t idxInElem = groupIndices[sub] - 1;
                size_t globalSrcIdx = elem * totalBytesPerElement + idxInElem;
                outputComponents[compIdx][writePos + sub] = temp[globalSrcIdx];
            }
        }
    }
}

// Reassembly function: rebuilds the full interleaved data from its components.
inline void reassembleBytesFromComponentsNested2(
    const std::vector<std::vector<uint8_t>>& inputComponents,
    uint8_t* byteArray,           // destination buffer pointer
    size_t byteArraySize,         // total size of the destination buffer
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads
) {
    size_t totalBytesPerElement = 0;
    for (const auto& group : allComponentSizes) {
        totalBytesPerElement += group.size();
    }
    size_t numElements = byteArraySize / totalBytesPerElement;

    #pragma omp parallel for num_threads(numThreads)
    for (size_t compIdx = 0; compIdx < inputComponents.size(); compIdx++) {
        const auto& groupIndices = allComponentSizes[compIdx];
        const auto& componentData = inputComponents[compIdx];
        size_t groupSize = groupIndices.size();

        for (size_t elem = 0; elem < numElements; elem++) {
            size_t readPos = elem * groupSize;
            for (size_t sub = 0; sub < groupSize; sub++) {
                size_t idxInElem = groupIndices[sub] - 1;
                size_t globalIndex = elem * totalBytesPerElement + idxInElem;
                byteArray[globalIndex] = componentData[readPos + sub];
            }
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

//=============================================================================
// Full (Non-decomposed) Compression/Decompression
//=============================================================================
size_t bzibCompression(const std::vector<uint8_t>& data, ProfilingInfo &pi, std::vector<uint8_t>& compressedData) {

  size_t compressedSize = compressWithbzip2(data, compressedData, 3);


  return compressedSize;
}

// Full decompression without decomposition
void bzibDecompression(const std::vector<uint8_t>& compressedData, std::vector<uint8_t>& decompressedData, size_t originalSize) {

  decompressWithbzip2(compressedData, decompressedData, originalSize);

  // Verify decompressed data
  if (!verifyDataMatch2(globalByteArray, decompressedData)) {
    std::cerr << "Error: Decompressed data doesn't match the original." << std::endl;
  }

}

//=============================================================================
// Decomposed Compression/Decompression
//
//=============================================================================
// Fused bzip Decomposed Parallel Compression
inline size_t bzipFusedDecomposedParallel(
    const uint8_t* data,
    size_t dataSize,
    ProfilingInfo &pi,
    std::vector<std::vector<uint8_t>>& compressedComponents,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads
) {
    // Compute total bytes per element.
    const size_t totalBytesPerElement = [&]() -> size_t {
        size_t sum = 0;
        for (const auto &group : allComponentSizes)
            sum += group.size();
        return sum;
    }();

    // Compute the number of elements in the interleaved data.
    const size_t numElements = dataSize / totalBytesPerElement;

    // Resize the output container so that each component will have its own compressed output.
    compressedComponents.resize(allComponentSizes.size());

    // Vectors for per–component timing measurements.
    std::vector<double> splitTimes(allComponentSizes.size(), 0.0);
    std::vector<double> compressTimes(allComponentSizes.size(), 0.0);

    size_t totalCompressedSize = 0;
    omp_set_num_threads(numThreads);

    // Start overall timer.
    double overallStart = omp_get_wtime();

    // Parallel region over the number of components.
    #pragma omp parallel
    {
        size_t threadCompressedSize = 0;  // Each thread’s local accumulation.

        #pragma omp for schedule(static)
        for (int compIdx = 0; compIdx < static_cast<int>(allComponentSizes.size()); compIdx++) {
            double localSplitTime = 0.0, localCompTime = 0.0;
            const auto &groupIndices = allComponentSizes[compIdx];
            const size_t groupSize = groupIndices.size();

            // --- Splitting Phase ---
            double t1 = omp_get_wtime();
            // Allocate a buffer to hold the extracted bytes for this component.
            std::vector<uint8_t> componentBuffer(numElements * groupSize);
            for (size_t elem = 0; elem < numElements; elem++) {
                const size_t baseIndex = elem * totalBytesPerElement;
                const size_t writePos = elem * groupSize;
                for (size_t sub = 0; sub < groupSize; sub++) {
                    // Convert 1-based index to 0-based.
                    const size_t idxInElem = groupIndices[sub] - 1;
                    componentBuffer[writePos + sub] = data[baseIndex + idxInElem];
                }
            }
            double t2 = omp_get_wtime();
            localSplitTime = t2 - t1;

            // --- Compression Phase ---
            double t3 = omp_get_wtime();
            std::vector<uint8_t> compData;
            // Use the bzipcompression routine on the component buffer.
            size_t cSize =compressWithbzip2(componentBuffer, compData, 3);
            double t4 = omp_get_wtime();
            localCompTime = t4 - t3;

            // Save the compressed component.
            compressedComponents[compIdx] = std::move(compData);
            threadCompressedSize += cSize;

            // Record per–component timings.
            splitTimes[compIdx] = localSplitTime;
            compressTimes[compIdx] = localCompTime;
        }
        // Safely update the global total compressed size.
        #pragma omp critical
        {
            totalCompressedSize += threadCompressedSize;
        }
    }

    double overallEnd = omp_get_wtime();
    double overallTime = overallEnd - overallStart;


    return totalCompressedSize;
}
//////////////////////////////////////////////////

inline void bzipDecomposedParallelDecompression(
    const std::vector<std::vector<uint8_t>>& compressedComponents,
    ProfilingInfo &pi,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads,
    size_t originalBlockSize,    // Original block size before compression
    uint8_t* finalReconstructed  // Preallocated destination buffer
) {
  auto startAll = std::chrono::high_resolution_clock::now();

  // Determine total bytes per element.
  size_t totalBytesPerElement = 0;
  for (const auto &group : allComponentSizes) {
    totalBytesPerElement += group.size();
  }
  size_t numElements = originalBlockSize / totalBytesPerElement;

  // Setup parallel decompression
  omp_set_num_threads(numThreads);
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(compressedComponents.size()); i++) {
    std::vector<uint8_t> temp(numElements * allComponentSizes[i].size());
    decompressWithbzip2(compressedComponents[i], temp, temp.size());

    // Interleave the decompressed data directly into the final buffer
    for (size_t elem = 0; elem < numElements; elem++) {
      size_t readPos = elem * allComponentSizes[i].size();
      for (size_t sub = 0; sub < allComponentSizes[i].size(); sub++) {
        size_t idxInElem = allComponentSizes[i][sub] - 1;
        size_t globalIndex = elem * totalBytesPerElement + idxInElem;
        finalReconstructed[globalIndex] = temp[readPos + sub];
      }
    }
  }

  auto endAll = std::chrono::high_resolution_clock::now();
  pi.total_time_decompressed = std::chrono::duration<double>(endAll - startAll).count();
}
//=============================================================================
// New Functions: Decomposed Then Chunked Parallel Compression/Decompression
//
//=============================================================================
inline size_t bzipDecomposedThenChunkedParallelCompression(
    const uint8_t* data,
    size_t dataSize,
    ProfilingInfo &pi,
    std::vector<std::vector<std::vector<uint8_t>>>& compressedBlocks,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads,
    size_t chunkBlockSize  // desired chunk size for each component
) {
    // 1. Decompose the full data into its components.
    std::vector<uint8_t> inputData(data, data + dataSize);
    std::vector<std::vector<uint8_t>> decomposedComponents;
    splitBytesIntoComponentsNested(inputData, decomposedComponents, allComponentSizes, numThreads);

    // 2. For each component, divide its data into chunks and compress each chunk.
    compressedBlocks.resize(decomposedComponents.size());
    size_t totalCompressedSize = 0;
    double overallStart = omp_get_wtime();

    #pragma omp parallel for num_threads(numThreads)
    for (int compIdx = 0; compIdx < static_cast<int>(decomposedComponents.size()); compIdx++) {
        const auto& compData = decomposedComponents[compIdx];
        size_t compDataSize = compData.size();
        size_t numChunks = (compDataSize + chunkBlockSize - 1) / chunkBlockSize;
        std::vector<std::vector<uint8_t>> compCompressed(numChunks);

        for (size_t chunk = 0; chunk < numChunks; chunk++) {
            size_t offset = chunk * chunkBlockSize;
            size_t currentBlockSize = std::min(chunkBlockSize, compDataSize - offset);
            std::vector<uint8_t> chunkData(compData.begin() + offset, compData.begin() + offset + currentBlockSize);
            std::vector<uint8_t> compBlock;
            size_t cSize = compressWithbzip2(chunkData, compBlock, 3);
            compCompressed[chunk] = compBlock;
            totalCompressedSize += cSize;
        }
        compressedBlocks[compIdx] = compCompressed;
    }
    double overallEnd = omp_get_wtime();
    pi.total_time_compressed = overallEnd - overallStart;
    return totalCompressedSize;
}

inline void bzipDecomposedThenChunkedParallelDecompression(
    const std::vector<std::vector<std::vector<uint8_t>>>& compressedBlocks,
    ProfilingInfo &pi,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads,
    size_t originalBlockSize, // original full data size (before decomposition)
    size_t chunkBlockSize,    // must match the size used during compression
    uint8_t* finalReconstructed // pointer to preallocated destination buffer
) {
    // Determine the per-element size.
    size_t totalBytesPerElement = 0;
    for (const auto &group : allComponentSizes)
        totalBytesPerElement += group.size();
    size_t numElements = originalBlockSize / totalBytesPerElement;

    // For each component, decompress each chunk sequentially.
    std::vector<std::vector<uint8_t>> decompressedComponents(compressedBlocks.size());
    double overallStart = omp_get_wtime();

    #pragma omp parallel for num_threads(numThreads)
    for (int compIdx = 0; compIdx < static_cast<int>(compressedBlocks.size()); compIdx++) {
        size_t compElementSize = allComponentSizes[compIdx].size();
        size_t expectedCompSize = numElements * compElementSize;
        std::vector<uint8_t> compDecompressed(expectedCompSize);
        size_t offset = 0;

        for (const auto &chunkCompressed : compressedBlocks[compIdx]) {
            size_t remaining = expectedCompSize - offset;
            size_t originalChunkSize = std::min(chunkBlockSize, remaining);
            std::vector<uint8_t> chunkDecompressed;
            decompressWithbzip2(chunkCompressed, chunkDecompressed, originalChunkSize);
            std::copy(chunkDecompressed.begin(), chunkDecompressed.end(),
                      compDecompressed.begin() + offset);
            offset += originalChunkSize;
        }
        decompressedComponents[compIdx] = compDecompressed;
    }
    double overallEnd = omp_get_wtime();
    pi.total_time_decompressed = overallEnd - overallStart;

    // Reassemble the full data from the decomposed components.
    // Here we create a temporary vector for the reassembled data.
    std::vector<uint8_t> reassembled(originalBlockSize, 0);
    reassembleBytesFromComponentsNested2(decompressedComponents,
                                         reassembled.data(),   // destination buffer pointer
                                         originalBlockSize,    // total size of the destination buffer
                                         allComponentSizes,
                                         numThreads);

    // Copy reassembled data into the preallocated destination buffer.
    std::memcpy(finalReconstructed, reassembled.data(), originalBlockSize);
}

#endif // BZIB_PARALLEL_H
