#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <zlib.h>             // Use zlib for compression
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
// Basic Zlib Compression/Decompression
//=============================================================================
inline size_t compressWithZlib(
    const std::vector<uint8_t>& data,
    std::vector<uint8_t>& compressedData,
    int compressionLevel = Z_DEFAULT_COMPRESSION
) {
    // Determine worst-case (maximum) compressed size.
    uLongf destLen = compressBound(data.size());
    compressedData.resize(destLen);

    // compress2() returns Z_OK (0) on success.
    int ret = compress2(
        reinterpret_cast<Bytef*>(compressedData.data()),
        &destLen,
        reinterpret_cast<const Bytef*>(data.data()),
        data.size(),
        compressionLevel
    );
    if (ret != Z_OK) {
        std::cerr << "Zlib compression error: " << ret << std::endl;
        return 0;
    }
    compressedData.resize(destLen);
    return destLen;
}

inline size_t decompressWithZlib(
    const std::vector<uint8_t>& compressedData,
    std::vector<uint8_t>& decompressedData,
    size_t originalSize
) {
    decompressedData.resize(originalSize);
    uLongf destLen = originalSize;
    int ret = uncompress(
        reinterpret_cast<Bytef*>(decompressedData.data()),
        &destLen,
        reinterpret_cast<const Bytef*>(compressedData.data()),
        compressedData.size()
    );
    if (ret != Z_OK) {
        std::cerr << "Zlib decompression error: " << ret << std::endl;
        return 0;
    }
    return destLen;
}

//=============================================================================
// Splitting and Reassembly Functions (unchanged)
//=============================================================================
inline void splitBytesIntoComponentsNestedzlib(
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

inline void reassembleBytesFromComponentsNestedzlib(
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

//=============================================================================
// Full (Non-decomposed) Compression/Decompression using Zlib
//=============================================================================
inline size_t zlibCompression(
    const std::vector<uint8_t>& data,
    ProfilingInfo &pi,
    std::vector<uint8_t>& compressedData
) {
    size_t cSize = compressWithZlib(data, compressedData, Z_DEFAULT_COMPRESSION);
    pi.type = "FullCompression";
    return cSize;
}

inline void zlibDecompression(
    const std::vector<uint8_t>& compressedData,
    std::vector<uint8_t>& decompressedData,
    ProfilingInfo &pi
) {
    decompressWithZlib(compressedData, decompressedData, globalByteArray.size());
}

//=============================================================================
// Decomposed Compression/Decompression using Zlib
//
inline size_t zlibFusedDecomposedParallel(
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

    // Vectors to store per-component timing.
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
                    // Adjust from 1-based index to 0-based.
                    const size_t idxInElem = groupIndices[sub] - 1;
                    componentBuffer[writePos + sub] = data[baseIndex + idxInElem];
                }
            }
            double t2 = omp_get_wtime();
            localSplitTime = t2 - t1;

            // --- Compression Phase ---
            double t3 = omp_get_wtime();
            std::vector<uint8_t> compData;
            size_t cSize = compressWithZlib(componentBuffer, compData, Z_DEFAULT_COMPRESSION);
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
    } // End of parallel region

    // Declare and compute the maximum split and compress times.
    double maxSplitTime = 0.0, maxCompressTime = 0.0;
    for (size_t i = 0; i < allComponentSizes.size(); i++) {
        if (splitTimes[i] > maxSplitTime)
            maxSplitTime = splitTimes[i];
        if (compressTimes[i] > maxCompressTime)
            maxCompressTime = compressTimes[i];
    }

    // Update profiling information.
    pi.split_time = maxSplitTime;
    pi.compress_time = maxCompressTime;
    double overallEnd = omp_get_wtime();
    pi.total_time_compressed = overallEnd - overallStart;

    return totalCompressedSize;
}


inline void zlibDecomposedParallelDecompression(
    const std::vector<std::vector<uint8_t>>& compressedComponents,
    ProfilingInfo &pi,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads,
    size_t originalBlockSize,    // Original block size (before compression)
    uint8_t* finalReconstructed  // Preallocated destination buffer
) {
    auto startAll = std::chrono::high_resolution_clock::now();

    // 1) Determine total bytes per element.
    size_t totalBytesPerElement = 0;
    for (const auto &group : allComponentSizes) {
        totalBytesPerElement += group.size();
    }
    size_t numElements = originalBlockSize / totalBytesPerElement;

    // 2) Compute expected uncompressed size for each component.
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
        decompressWithZlib(compressedComponents[i], temp, chunkSizes[i]);
        decompressedSubChunks[i] = temp;
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
// New Functions: Decomposed Then Chunked Parallel Compression/Decompression using Zlib
//=============================================================================
inline size_t zlibDecomposedThenChunkedParallelCompression(
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
    splitBytesIntoComponentsNested2(inputData, decomposedComponents, allComponentSizes, numThreads);

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
            size_t cSize = compressWithZlib(chunkData, compBlock, Z_DEFAULT_COMPRESSION);
            compCompressed[chunk] = compBlock;
            totalCompressedSize += cSize;
        }
        compressedBlocks[compIdx] = compCompressed;
    }
    double overallEnd = omp_get_wtime();
    pi.total_time_compressed = overallEnd - overallStart;
    return totalCompressedSize;
}

inline void zlibDecomposedThenChunkedParallelDecompression(
    const std::vector<std::vector<std::vector<uint8_t>>>& compressedBlocks,
    ProfilingInfo &pi,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads,
    size_t originalBlockSize, // original full data size (before decomposition)
    size_t chunkBlockSize,    // must match the size used during compression
    uint8_t* finalReconstructed // pointer to preallocated destination buffer
) {
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
            decompressWithZlib(chunkCompressed, chunkDecompressed, originalChunkSize);
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
    reassembleBytesFromComponentsNested2(decompressedComponents,
                                         reassembled.data(),
                                         originalBlockSize,
                                         allComponentSizes,
                                         numThreads);
    std::memcpy(finalReconstructed, reassembled.data(), originalBlockSize);
}
