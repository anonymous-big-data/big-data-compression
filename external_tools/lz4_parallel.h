#ifndef LZ4_PARALLEL_H
#define LZ4_PARALLEL_H

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <lz4.h>
#include <lz4hc.h>
#include <chrono>
#include <cstdint>
#include <omp.h>
#include <numeric>
#include <algorithm>
#include <cstring>

#include "profiling_info.h"

extern std::vector<uint8_t> globalByteArray;

//=============================================================================
// Basic LZ4 Compression/Decompression
//=============================================================================
inline size_t compressWithLZ4(
    const std::vector<uint8_t>& data,
    std::vector<uint8_t>& compressedData,
    int compressionLevel = 3
) {
    int maxCompressedSize = LZ4_compressBound(data.size());
    compressedData.resize(maxCompressedSize);

    int compressedSize;
    if (compressionLevel > 0) {
        // Use high-compression mode.
        compressedSize = LZ4_compress_HC(
            reinterpret_cast<const char*>(data.data()),
            reinterpret_cast<char*>(compressedData.data()),
            data.size(),
            maxCompressedSize,
            compressionLevel
        );
    } else {
        // Use fast compression mode.
        compressedSize = LZ4_compress_default(
            reinterpret_cast<const char*>(data.data()),
            reinterpret_cast<char*>(compressedData.data()),
            data.size(),
            maxCompressedSize
        );
    }

    if (compressedSize <= 0) {
        std::cerr << "LZ4 compression error." << std::endl;
        return 0;
    }
    compressedData.resize(compressedSize);
    return compressedSize;
}

inline size_t decompressWithLZ4(
    const std::vector<uint8_t>& compressedData,
    std::vector<uint8_t>& decompressedData,
    size_t originalSize
) {
    decompressedData.resize(originalSize);
    int decompressedSize = LZ4_decompress_safe(
        reinterpret_cast<const char*>(compressedData.data()),
        reinterpret_cast<char*>(decompressedData.data()),
        compressedData.size(),
        originalSize
    );
    if (decompressedSize < 0) {
        std::cerr << "LZ4 decompression error." << std::endl;
        return 0;
    }
    return decompressedSize;
}

//=============================================================================
// Splitting and Reassembly Functions
//=============================================================================
inline void splitBytesIntoComponentsNestedlz4(
    const std::vector<uint8_t>& byteArray,
    std::vector<std::vector<uint8_t>>& outputComponents,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads
) {
    size_t totalBytesPerElement = 0;
    for (const auto& group : allComponentSizes)
        totalBytesPerElement += group.size();

    size_t numElements = byteArray.size() / totalBytesPerElement;
    outputComponents.resize(allComponentSizes.size());

    // Make a temporary copy.
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
                // Adjust from 1-based index if needed.
                size_t idxInElem = groupIndices[sub] - 1;
                size_t globalSrcIdx = elem * totalBytesPerElement + idxInElem;
                outputComponents[compIdx][writePos + sub] = temp[globalSrcIdx];
            }
        }
    }
}

inline void reassembleBytesFromComponentsNestedlz4(
    const std::vector<std::vector<uint8_t>>& inputComponents,
    uint8_t* byteArray,           // destination buffer pointer
    size_t byteArraySize,         // total size of the destination buffer
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads
) {
    size_t totalBytesPerElement = 0;
    for (const auto& group : allComponentSizes)
        totalBytesPerElement += group.size();

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
inline size_t lz4Compression(
    const std::vector<uint8_t>& data,
    ProfilingInfo &pi,
    std::vector<uint8_t>& compressedData
) {
    size_t cSize = compressWithLZ4(data, compressedData, 3);
    pi.type = "LZ4 Full Compression";
    return cSize;
}

inline void lz4Decompression(
    const std::vector<uint8_t>& compressedData,
    std::vector<uint8_t>& decompressedData,
    ProfilingInfo &pi
) {
    decompressWithLZ4(compressedData, decompressedData, globalByteArray.size());
}

//=============================================================================
// Decomposed Compression/Decompression
//=============================================================================
// Fused LZ4 Decomposed Parallel Compression
inline size_t lz4FusedDecomposedParallel(
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

    // Compute the number of elements.
    const size_t numElements = dataSize / totalBytesPerElement;
    compressedComponents.resize(allComponentSizes.size());

    // Timing vectors.
    std::vector<double> splitTimes(allComponentSizes.size(), 0.0);
    std::vector<double> compressTimes(allComponentSizes.size(), 0.0);
    size_t totalCompressedSize = 0;
    omp_set_num_threads(numThreads);
    double overallStart = omp_get_wtime();

    // Parallel over components.
    #pragma omp parallel
    {
        size_t threadCompressedSize = 0;
        #pragma omp for schedule(static)
        for (int compIdx = 0; compIdx < static_cast<int>(allComponentSizes.size()); compIdx++) {
            double localSplitTime = 0.0, localCompTime = 0.0;
            const auto &groupIndices = allComponentSizes[compIdx];
            const size_t groupSize = groupIndices.size();

            // Splitting Phase.
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

            // Compression Phase.
            double t3 = omp_get_wtime();
            std::vector<uint8_t> compData;
            size_t cSize = compressWithLZ4(componentBuffer, compData, 3);
            double t4 = omp_get_wtime();
            localCompTime = t4 - t3;

            compressedComponents[compIdx] = std::move(compData);
            threadCompressedSize += cSize;

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

//////////////////////////////////////////////////
inline void lz4DecomposedParallelDecompression(
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
    for (const auto &group : allComponentSizes)
        totalBytesPerElement += group.size();
    size_t numElements = originalBlockSize / totalBytesPerElement;

    // Compute expected uncompressed size for each component.
    std::vector<size_t> chunkSizes;
    chunkSizes.reserve(allComponentSizes.size());
    for (const auto &group : allComponentSizes)
        chunkSizes.push_back(numElements * group.size());

    omp_set_num_threads(numThreads);
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(compressedComponents.size()); i++) {
        std::vector<uint8_t> temp(chunkSizes[i]);
        decompressWithLZ4(compressedComponents[i], temp, chunkSizes[i]);

        // Directly interleave decompressed data into the final buffer
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
//=============================================================================
inline size_t lz4DecomposedThenChunkedParallelCompression(
    const uint8_t* data,
    size_t dataSize,
    ProfilingInfo &pi,
    std::vector<std::vector<std::vector<uint8_t>>>& compressedBlocks,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads,
    size_t chunkBlockSize
) {
    // 1. Decompose the full data into its components.
    std::vector<uint8_t> inputData(data, data + dataSize);
    std::vector<std::vector<uint8_t>> decomposedComponents;
    splitBytesIntoComponentsNestedlz4(inputData, decomposedComponents, allComponentSizes, numThreads);

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
            size_t cSize = compressWithLZ4(chunkData, compBlock, 3);
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

inline void lz4DecomposedThenChunkedParallelDecompression(
    const std::vector<std::vector<std::vector<uint8_t>>>& compressedBlocks,
    ProfilingInfo &pi,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads,
    size_t originalBlockSize, // original full data size (before decomposition)
    size_t chunkBlockSize,    // must match the size used during compression
    uint8_t* finalReconstructed // pointer to preallocated destination buffer
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
            decompressWithLZ4(chunkCompressed, chunkDecompressed, originalChunkSize);
            std::copy(chunkDecompressed.begin(), chunkDecompressed.end(),
                      compDecompressed.begin() + offset);
            offset += originalChunkSize;
        }
        decompressedComponents[compIdx] = compDecompressed;
    }
    double overallEnd = omp_get_wtime();
    pi.total_time_decompressed = overallEnd - overallStart;

    // Reassemble the full data.
    std::vector<uint8_t> reassembled(originalBlockSize, 0);
    reassembleBytesFromComponentsNestedlz4(decompressedComponents,
                                         reassembled.data(),
                                         originalBlockSize,
                                         allComponentSizes,
                                         numThreads);
    std::memcpy(finalReconstructed, reassembled.data(), originalBlockSize);
}

#endif // LZ4_PARALLEL_H
