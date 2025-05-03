#ifndef SNAPPY_PARALLEL_H
#define SNAPPY_PARALLEL_H

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <snappy.h>
#include <chrono>
#include <cstdint>
#include <omp.h>
#include <numeric>
#include <algorithm>
#include <cstring>

#include "profiling_info.h"

// Global data (defined elsewhere)
extern std::vector<uint8_t> globalByteArray;

//=============================================================================
// Basic Snappy Compression/Decompression
//=============================================================================

inline size_t compressWithSnappy(
    const std::vector<uint8_t>& data,
    std::vector<uint8_t>& compressedData
) {
    size_t maxCompressedSize = snappy::MaxCompressedLength(data.size());
    compressedData.resize(maxCompressedSize);

    size_t compressedSize = 0;
    snappy::RawCompress(reinterpret_cast<const char*>(data.data()),
                        data.size(),
                        reinterpret_cast<char*>(compressedData.data()),
                        &compressedSize);
    compressedData.resize(compressedSize);
    return compressedSize;
}

// Decompress data using Snappy.
inline size_t decompressWithSnappy(
    const std::vector<uint8_t>& compressedData,
    std::vector<uint8_t>& decompressedData,
    size_t originalSize
) {
    decompressedData.resize(originalSize);
    bool success = snappy::RawUncompress(reinterpret_cast<const char*>(compressedData.data()),
                                         compressedData.size(),
                                         reinterpret_cast<char*>(decompressedData.data()));
    if (!success) {
        std::cerr << "Snappy decompression error." << std::endl;
        return 0;
    }
    return originalSize;
}

//=============================================================================
// Splitting and Reassembly Functions (Nested Decomposition)
//=============================================================================

inline void splitBytesIntoComponentsNestedsanppy(
    const std::vector<uint8_t>& byteArray,
    std::vector<std::vector<uint8_t>>& outputComponents,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads
) {
    // Compute total bytes per element from all groups.
    size_t totalBytesPerElement = 0;
    for (const auto& group : allComponentSizes) {
        totalBytesPerElement += group.size();
    }
    size_t numElements = byteArray.size() / totalBytesPerElement;
    outputComponents.resize(allComponentSizes.size());
    std::vector<uint8_t> temp(byteArray);

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
                // Adjust from 1-based index if necessary.
                size_t idxInElem = groupIndices[sub] - 1;
                size_t globalSrcIdx = elem * totalBytesPerElement + idxInElem;
                outputComponents[compIdx][writePos + sub] = temp[globalSrcIdx];
            }
        }
    }
}

inline void reassembleBytesFromComponentsNestedsnappy(
    const std::vector<std::vector<uint8_t>>& inputComponents,
    uint8_t* byteArray,
    size_t byteArraySize,
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

//=============================================================================
// Full (Non-decomposed) Compression/Decompression
//=============================================================================

inline size_t snappyCompression(
    const std::vector<uint8_t>& data,
    ProfilingInfo &pi,
    std::vector<uint8_t>& compressedData
) {
    size_t cSize = compressWithSnappy(data, compressedData);
    pi.type = "FullCompression";
    return cSize;
}

inline void snappyDecompression(
    const std::vector<uint8_t>& compressedData,
    std::vector<uint8_t>& decompressedData,
    ProfilingInfo &pi
) {
    decompressWithSnappy(compressedData, decompressedData, globalByteArray.size());
}

//=============================================================================
// Decomposed Compression/Decompression
//=============================================================================

// Fused Snappy Decomposed Parallel Compression
inline size_t snappyFusedDecomposedParallel(
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
    // Number of elements in the interleaved data.
    const size_t numElements = dataSize / totalBytesPerElement;

    // Resize the output container.
    compressedComponents.resize(allComponentSizes.size());

    // Vectors for per–component timing measurements.
    std::vector<double> splitTimes(allComponentSizes.size(), 0.0);
    std::vector<double> compressTimes(allComponentSizes.size(), 0.0);

    size_t totalCompressedSize = 0;
    omp_set_num_threads(numThreads);
    double overallStart = omp_get_wtime();

    #pragma omp parallel
    {
        size_t threadCompressedSize = 0;
        #pragma omp for schedule(static)
        for (int compIdx = 0; compIdx < static_cast<int>(allComponentSizes.size()); compIdx++) {
            double localSplitTime = 0.0, localCompTime = 0.0;
            const auto &groupIndices = allComponentSizes[compIdx];
            const size_t groupSize = groupIndices.size();

            // --- Splitting Phase ---
            double t1 = omp_get_wtime();
            std::vector<uint8_t> componentBuffer(numElements * groupSize);
            for (size_t elem = 0; elem < numElements; elem++) {
                const size_t baseIndex = elem * totalBytesPerElement;
                const size_t writePos = elem * groupSize;
                for (size_t sub = 0; sub < groupSize; sub++) {
                    const size_t idxInElem = groupIndices[sub] - 1;
                    componentBuffer[writePos + sub] = data[baseIndex + idxInElem];
                }
            }
            double t2 = omp_get_wtime();
            localSplitTime = t2 - t1;

            // --- Compression Phase ---
            double t3 = omp_get_wtime();
            std::vector<uint8_t> compData;
            size_t cSize = compressWithSnappy(componentBuffer, compData);
            double t4 = omp_get_wtime();
            localCompTime = t4 - t3;

            compressedComponents[compIdx] = std::move(compData);
            threadCompressedSize += cSize;

            // Record per–component timings.
            splitTimes[compIdx] = localSplitTime;
            compressTimes[compIdx] = localCompTime;
        }
        #pragma omp critical
        {
            totalCompressedSize += threadCompressedSize;
        }
    }

    double overallEnd = omp_get_wtime();
    double overallTime = overallEnd - overallStart;

    double maxSplitTime = 0.0, maxCompressTime = 0.0;
    for (size_t i = 0; i < allComponentSizes.size(); i++) {
        if (splitTimes[i] > maxSplitTime)
            maxSplitTime = splitTimes[i];
        if (compressTimes[i] > maxCompressTime)
            maxCompressTime = compressTimes[i];
    }

    pi.split_time = maxSplitTime;
    pi.compress_time = maxCompressTime;
    pi.total_time_compressed = overallTime;

    return totalCompressedSize;
}

// Parallel Decompression with Decomposition using Snappy.

inline void snappyDecomposedParallelDecompression(
    const std::vector<std::vector<uint8_t>>& compressedComponents,
    ProfilingInfo &pi,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads,
    size_t originalBlockSize,    // original block size before compression
    uint8_t* finalReconstructed  // preallocated destination buffer
) {
    auto startAll = std::chrono::high_resolution_clock::now();

    // Determine total bytes per element.
    size_t totalBytesPerElement = 0;
    for (const auto &group : allComponentSizes) {
        totalBytesPerElement += group.size();
    }
    size_t numElements = originalBlockSize / totalBytesPerElement;

    omp_set_num_threads(numThreads);
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(compressedComponents.size()); i++) {
        std::vector<uint8_t> temp(numElements * allComponentSizes[i].size());
        decompressWithSnappy(compressedComponents[i], temp, temp.size());

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
// Decomposed Then Chunked Parallel Compression/Decompression
//=============================================================================

inline size_t snappyDecomposedThenChunkedParallelCompression(
    const uint8_t* data,
    size_t dataSize,
    ProfilingInfo &pi,
    std::vector<std::vector<std::vector<uint8_t>>>& compressedBlocks,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads,
    size_t chunkBlockSize  // desired chunk size for each component
) {
    // 1) Decompose the full data into its components.
    std::vector<uint8_t> inputData(data, data + dataSize);
    std::vector<std::vector<uint8_t>> decomposedComponents;
    splitBytesIntoComponentsNestedsanppy(inputData, decomposedComponents, allComponentSizes, numThreads);

    // 2) For each component, divide its data into chunks and compress each chunk.
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
            size_t cSize = compressWithSnappy(chunkData, compBlock);
            compCompressed[chunk] = compBlock;
            #pragma omp atomic
            totalCompressedSize += cSize;
        }
        compressedBlocks[compIdx] = compCompressed;
    }
    double overallEnd = omp_get_wtime();
    pi.total_time_compressed = overallEnd - overallStart;
    return totalCompressedSize;
}

// Decomposed then Chunked Parallel Decompression using Snappy.
inline void snappyDecomposedThenChunkedParallelDecompression(
    const std::vector<std::vector<std::vector<uint8_t>>>& compressedBlocks,
    ProfilingInfo &pi,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads,
    size_t originalBlockSize,
    size_t chunkBlockSize,
    uint8_t* finalReconstructed
) {
    // Determine per-element size.
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
            decompressWithSnappy(chunkCompressed, chunkDecompressed, originalChunkSize);
            std::copy(chunkDecompressed.begin(), chunkDecompressed.end(),
                      compDecompressed.begin() + offset);
            offset += originalChunkSize;
        }
        decompressedComponents[compIdx] = compDecompressed;
    }
    double overallEnd = omp_get_wtime();
    pi.total_time_decompressed = overallEnd - overallStart;

    // Reassemble the full data from the decomposed components.
    std::vector<uint8_t> reassembled(originalBlockSize, 0);
    reassembleBytesFromComponentsNestedsnappy(decompressedComponents,
                                         reassembled.data(),
                                         originalBlockSize,
                                         allComponentSizes,
                                         numThreads);
    std::memcpy(finalReconstructed, reassembled.data(), originalBlockSize);
}

#endif
