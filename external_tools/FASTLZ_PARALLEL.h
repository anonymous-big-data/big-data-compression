#ifndef FASTLZ_PARALLEL_H
#define FASTLZ_PARALLEL_H

#include <fstream>
#include <iostream>
#include <vector>

#include <chrono>
#include <cstdint>
#include <omp.h>
#include <numeric>
#include "fastlz.h">
#include"fastlz.c"


#include "profiling_info.h"

extern std::vector<uint8_t> globalByteArray;

inline size_t compressWithFastLZ1(
    const uint8_t* data,
    size_t dataSize,
    std::vector<uint8_t>& compressedData
) {
  // Slightly larger buffer for worst-case compression output.
  size_t cBuffSize = static_cast<size_t>(dataSize * 1.05 + 16);
  compressedData.resize(cBuffSize);

  size_t cSize = fastlz_compress(
      data,
      dataSize,
      compressedData.data()
  );

  compressedData.resize(cSize);
  return cSize;
}

// Overload: decompress directly into a preallocated buffer.
inline void decompressWithFastLZ1(
    const std::vector<uint8_t>& compressedData,
    uint8_t* decompressedData, // destination pointer
    size_t originalSize        // expected uncompressed block size
) {
  size_t dSize = fastlz_decompress(
      compressedData.data(),
      compressedData.size(),
      decompressedData,
      originalSize
  );
  if (dSize == 0) {
    std::cerr << "FastLZ decompression error: Invalid input data" << std::endl;
  }
}

//-----------------------------------------------------------------------------
// Optimized In-Place Reordering for Decomposition-Based Compression
// Uses SIMD (AVX2) for faster memory operations and OpenMP for parallel execution
//-----------------------------------------------------------------------------
inline void splitBytesIntoComponentsNested1(
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
    std::vector<uint8_t> temp(byteArray);

    // Resize
    for (size_t i = 0; i < allComponentSizes.size(); i++) {
        outputComponents[i].resize(numElements * allComponentSizes[i].size());
    }
#pragma omp parallel for num_threads(numThreads)
    for (size_t elem = 0; elem < numElements; elem++) {
        for (size_t compIdx = 0; compIdx < allComponentSizes.size(); compIdx++) {
            const auto& groupIndices = allComponentSizes[compIdx];
            size_t groupSize = groupIndices.size();
            size_t writePos = elem * groupSize;

            size_t sub = 0;
#ifdef __AVX2__

#endif
            for (; sub < groupSize; sub++) {
                size_t idxInElem = groupIndices[sub] - 1;
                size_t globalSrcIdx = elem * totalBytesPerElement + idxInElem;
                outputComponents[compIdx][writePos + sub] = temp[globalSrcIdx];
            }
        }
    }
}


inline void reassembleBytesFromComponentsNested(
    const std::vector<std::vector<uint8_t>>& inputComponents,
    uint8_t* byteArray,           // destination buffer pointer
    size_t byteArraySize,         // total size of the destination buffer
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads
) {
  // 1. Compute the total number of bytes per element.
  size_t totalBytesPerElement = 0;
  for (const auto& group : allComponentSizes) {
    totalBytesPerElement += group.size();
  }

  // 2. Compute the number of elements stored in the destination buffer.
  size_t numElements = byteArraySize / totalBytesPerElement;

  // 3. For each component  reassemble its part into the destination.
#pragma omp parallel for num_threads(numThreads)
  for (size_t compIdx = 0; compIdx < inputComponents.size(); compIdx++) {
    const auto& groupIndices = allComponentSizes[compIdx];
    const auto& componentData = inputComponents[compIdx];
    size_t groupSize = groupIndices.size();

    // For each element, copy the corresponding bytes.
    for (size_t elem = 0; elem < numElements; elem++) {
      size_t readPos = elem * groupSize;
      for (size_t sub = 0; sub < groupSize; sub++) {
        // Adjust the index (subtract one if your indices are 1-based).
        size_t idxInElem = groupIndices[sub] - 1;
        size_t globalIndex = elem * totalBytesPerElement + idxInElem;
        byteArray[globalIndex] = componentData[readPos + sub];
      }
    }
  }
}

//////////////////////////////
inline void fastlzDecomposedParallelDecompression(
    const std::vector<std::vector<uint8_t>>& compressedComponents,
    ProfilingInfo &pi,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads,
    size_t originalBlockSize,
    uint8_t* finalReconstructed
) {
    auto startAll = std::chrono::high_resolution_clock::now();

    // 1) Determine total bytes per element (sum of component sizes).
    size_t totalBytesPerElement = 0;
    for (const auto &group : allComponentSizes) {
        totalBytesPerElement += group.size();
    }
    // Compute the number of elements based on the original block size.
    size_t numElements = originalBlockSize / totalBytesPerElement;

    // 2) Compute the expected uncompressed size for each component.
    std::vector<size_t> chunkSizes;
    chunkSizes.reserve(allComponentSizes.size());
    for (const auto &group : allComponentSizes) {
        chunkSizes.push_back(numElements * group.size());
    }

    // 3) Decompress each compressed component in parallel.
    std::vector<std::vector<uint8_t>> decompressedSubChunks(compressedComponents.size());
    omp_set_num_threads(numThreads);
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(compressedComponents.size()); i++) {
        std::vector<uint8_t> temp(chunkSizes[i]);
        decompressWithFastLZ1(compressedComponents[i], temp.data(), chunkSizes[i]);
        decompressedSubChunks[i] = temp;
    }

    // 4) Reassemble the full block from the decompressed sub-components.

    reassembleBytesFromComponentsNested(
        decompressedSubChunks,
        finalReconstructed,       // Pointer to preallocated destination buffer
        originalBlockSize,        // Total size of the destination buffer
        allComponentSizes,
        numThreads
    );

    auto endAll = std::chrono::high_resolution_clock::now();
    pi.total_time_decompressed = std::chrono::duration<double>(endAll - startAll).count();
}


/////////////////////////////////////////////
// Fused Splitting and Compression
// This function fuses the splitting (reordering) and compression steps
// into one routine for potential performance benefits.
/////////////////////////////////////////////
inline size_t fastlzFusedDecomposedParallel1(
    const uint8_t* data, size_t dataSize,
    ProfilingInfo &pi,
    std::vector<std::vector<uint8_t>>& compressedComponents,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads
) {
    // Precompute total bytes per element (invariant).
    const size_t totalBytesPerElement = [&]() -> size_t {
        size_t sum = 0;
        for (const auto& group : allComponentSizes)
            sum += group.size();
        return sum;
    }();

    // Determine the number of elements in the interleaved data.
    const size_t numElements = dataSize / totalBytesPerElement;

    // Resize the output container.
    compressedComponents.resize(allComponentSizes.size());

    // Prepare containers for per-component timing measurements.
    std::vector<double> splitTimes(allComponentSizes.size(), 0.0);
    std::vector<double> compressTimes(allComponentSizes.size(), 0.0);

    size_t totalCompressedSize = 0;
    omp_set_num_threads(numThreads);

    // Start overall timer.
    double overallStart = omp_get_wtime();

    // Parallel region with thread-local accumulation.
    #pragma omp parallel
    {
        size_t threadCompressedSize = 0;

        #pragma omp for schedule(static)
        for (int compIdx = 0; compIdx < static_cast<int>(allComponentSizes.size()); compIdx++) {
            double localSplitTime = 0.0, localCompTime = 0.0;
            const auto& groupIndices = allComponentSizes[compIdx];
            const size_t groupSize = groupIndices.size();

            // --- Splitting Phase ---
            double t1 = omp_get_wtime();
            // Allocate an uncompressed buffer for this component.
            std::vector<uint8_t> componentBuffer(numElements * groupSize);

            // Loop over each element and extract the bytes for this component.
            for (size_t elem = 0; elem < numElements; elem++) {
                const size_t baseIndex = elem * totalBytesPerElement;
                const size_t writePos = elem * groupSize;
                for (size_t sub = 0; sub < groupSize; sub++) {
                    // Adjust from 1-based to 0-based index.
                    const size_t idxInElem = groupIndices[sub] - 1;
                    componentBuffer[writePos + sub] = data[baseIndex + idxInElem];
                }
            }
            double t2 = omp_get_wtime();
            localSplitTime = t2 - t1;

            // --- Compression Phase ---
            double t3 = omp_get_wtime();
            std::vector<uint8_t> compData;
            // Call the pointer-based compress function.
            size_t cSize = compressWithFastLZ1(componentBuffer.data(), componentBuffer.size(), compData);
            double t4 = omp_get_wtime();
            localCompTime = t4 - t3;

            // Save the compressed component.
            compressedComponents[compIdx] = std::move(compData);
            threadCompressedSize += cSize;

            // Store per-component timings.
            splitTimes[compIdx] = localSplitTime;
            compressTimes[compIdx] = localCompTime;
        }

        // Safely update the global totalCompressedSize.
        #pragma omp critical
        {
            totalCompressedSize += threadCompressedSize;
        }
    }

    // Stop overall timer.
    double overallEnd = omp_get_wtime();
    double overallTime = overallEnd - overallStart;
    double maxSplitTime = 0.0;
    double maxCompressTime = 0.0;
    for (size_t i = 0; i < allComponentSizes.size(); i++) {
        if (splitTimes[i] > maxSplitTime)
            maxSplitTime = splitTimes[i];
        if (compressTimes[i] > maxCompressTime)
            maxCompressTime = compressTimes[i];
    }

    // Update profiling information.
    pi.split_time = maxSplitTime;
    pi.compress_time = maxCompressTime;
    pi.total_time_compressed = overallTime;

    return totalCompressedSize;
}

//
inline size_t fastlzFusedDecomposedParallel(
    const uint8_t* data,
    size_t dataSize,
    ProfilingInfo &pi,
    std::vector<std::vector<uint8_t>>& compressedComponents,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads
) {
    // ----------------------------------------------------------------------
    // 1. Calculate the "bytes per element" and "numElements."
    // ----------------------------------------------------------------------
    size_t totalBytesPerElement = 0;
    for (const auto& group : allComponentSizes) {
        totalBytesPerElement += group.size();
    }
    const size_t numElements = dataSize / totalBytesPerElement;

    // We only want ONE final compressed vector, so resize accordingly.
    compressedComponents.resize(1);

    // ----------------------------------------------------------------------
    // 2. Prepare for parallel splitting into a SINGLE large buffer.
    //    We want to place each component’s data in a back-to-back fashion.
    // ----------------------------------------------------------------------

    std::vector<size_t> componentOffsets(allComponentSizes.size());
    {
        size_t currentOffset = 0;
        for (size_t i = 0; i < allComponentSizes.size(); i++) {
            size_t compUncompressedSize = allComponentSizes[i].size() * numElements;
            componentOffsets[i] = currentOffset;
            currentOffset += compUncompressedSize;
        }
    }
    // The final concatenated buffer will be size = dataSize.
    // Because sum_of(allComponentSizes[i].size()) * numElements = dataSize.
    std::vector<uint8_t> concatenatedUncompressed(dataSize);
    std::vector<double> splitTimes(allComponentSizes.size(), 0.0);

    // ----------------------------------------------------------------------
    // 3. Parallel splitting: fill the big "concatenatedUncompressed" buffer.
    // ----------------------------------------------------------------------
    double overallSplitStart = omp_get_wtime();
    omp_set_num_threads(numThreads);

#pragma omp parallel for schedule(static)
    for (int compIdx = 0; compIdx < static_cast<int>(allComponentSizes.size()); compIdx++) {
        double t1 = omp_get_wtime();

        const auto& groupIndices = allComponentSizes[compIdx];
        const size_t groupSize = groupIndices.size();

        // Where this component’s data begins in the final concatenated buffer.
        size_t compOffset = componentOffsets[compIdx];

        // Extract data from `data` -> write into concatenatedUncompressed[compOffset + ...]
        for (size_t elem = 0; elem < numElements; elem++) {
            size_t writePos = compOffset + elem * groupSize;
            // The base index of the 'elem'-th element in the original interleaved data
            size_t baseIndex = elem * totalBytesPerElement;

            for (size_t sub = 0; sub < groupSize; sub++) {
                size_t idxInElem = groupIndices[sub] - 1;  // if 1-based
                concatenatedUncompressed[writePos + sub] = data[baseIndex + idxInElem];
            }
        }

        double t2 = omp_get_wtime();
        splitTimes[compIdx] = (t2 - t1);
    }
    double overallSplitEnd = omp_get_wtime();
    double maxSplitTime = 0.0;
    for (auto st : splitTimes) {
        if (st > maxSplitTime) maxSplitTime = st;
    }

    // ----------------------------------------------------------------------
    // 4. Compress the entire "concatenatedUncompressed" as ONE component.
    // ----------------------------------------------------------------------
    double compStart = omp_get_wtime();
    std::vector<uint8_t> compData;
    size_t cSize = compressWithFastLZ1(
        concatenatedUncompressed.data(), // pointer to big uncompressed buffer
        concatenatedUncompressed.size(),
        compData
    );
    double compEnd = omp_get_wtime();
    double compressTime = compEnd - compStart;

    compressedComponents[0] = std::move(compData);

    // ----------------------------------------------------------------------
    // 5. Calculate times and return total compressed size
    // ----------------------------------------------------------------------
    double overallEnd = omp_get_wtime();
    double overallTime = overallEnd - overallSplitStart;
    pi.split_time = maxSplitTime;
    pi.compress_time = compressTime;
    pi.total_time_compressed = overallTime;

    return cSize; // total compressed bytes
}

//-----------------------------------------------------------------------------
// Compute Overall Compression Ratio
//-----------------------------------------------------------------------------
inline double calculateCompressionRatio(size_t originalSize, size_t compressedSize) {
    return (compressedSize == 0)
        ? 0.0
        : static_cast<double>(originalSize) / static_cast<double>(compressedSize);
}

#endif
