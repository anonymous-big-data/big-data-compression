#ifndef FASTLZ_PARALLEL_H
#define FASTLZ_PARALLEL_H

#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <omp.h>
#include "fastlz.h"
#include "profiling_info.h"

// Global data (defined elsewhere)
extern std::vector<uint8_t> globalByteArray;

//=== Existing functions ======================================================

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

// Splitting function: decomposes full (interleaved) data into its components.
inline void splitBytesIntoComponentsNested(
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
inline void reassembleBytesFromComponentsNested(
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
//////////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////
inline void fastlzDecomposedParallelDecompression(
    const std::vector<std::vector<uint8_t>>& compressedComponents,
    ProfilingInfo &pi,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads,
    size_t originalBlockSize,    // Original block size (before compression)
    uint8_t* finalReconstructed  // Preallocated destination buffer
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
        // Preallocate a temporary vector of the expected uncompressed size.
        std::vector<uint8_t> temp(chunkSizes[i]);
        // Decompress directly into the temporary vector's buffer.
        decompressWithFastLZ1(compressedComponents[i], temp.data(), chunkSizes[i]);
        decompressedSubChunks[i] = temp;
    }

    // 4) Reassemble the full block from the decompressed sub-components.
    // Write directly into the preallocated destination buffer.
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
// Fused Splitting and Compression (New Function)
// This function fuses the splitting (reordering) and compression steps
// into one routine for potential performance benefits.
/////////////////////////////////////////////
inline size_t fastlzFusedDecomposedParallel(
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
        size_t threadCompressedSize = 0;  // Each thread's local accumulation.

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

    // Instead of summing per-component times (which overcounts concurrent work),
    // take the maximum time among components.
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
//////////////////////////////////////////////////////////////////////////////////////////////////////
//=== New Functions: Decompose Then Chunk ====================================

// Compression: First decompose full data into components, then divide each component
// into fixed-size chunks (except possibly the last chunk), and compress each chunk.
inline size_t fastlzDecomposedThenChunkedParallelCompression(
    const uint8_t* data,
    size_t dataSize,
    ProfilingInfo &pi,
    std::vector<std::vector<std::vector<uint8_t>>>& compressedBlocks,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads,
    size_t chunkBlockSize // desired block size for each chunk in a component
) {
    // 1. Decompose full data into its components.
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
        // Compute number of chunks (each of chunkBlockSize, except possibly the last one).
        size_t numChunks = (compDataSize + chunkBlockSize - 1) / chunkBlockSize;
        std::vector<std::vector<uint8_t>> compCompressed(numChunks);

        for (size_t chunk = 0; chunk < numChunks; chunk++) {
            size_t offset = chunk * chunkBlockSize;
            size_t currentBlockSize = std::min(chunkBlockSize, compDataSize - offset);
            std::vector<uint8_t> compBlock;
            size_t cSize = compressWithFastLZ1(&compData[offset], currentBlockSize, compBlock);
            compCompressed[chunk] = std::move(compBlock);
            totalCompressedSize += cSize;
        }
        compressedBlocks[compIdx] = std::move(compCompressed);
    }
    double overallEnd = omp_get_wtime();
    pi.total_time_compressed = overallEnd - overallStart;

    return totalCompressedSize;
}

// Decompression: For each component, decompress each compressed chunk, reassemble
// the full decomposed component, then reassemble the full original data.
inline void fastlzDecomposedThenChunkedParallelDecompression(
    const std::vector<std::vector<std::vector<uint8_t>>>& compressedBlocks,
    ProfilingInfo &pi,
    const std::vector<std::vector<size_t>>& allComponentSizes,
    int numThreads,
    size_t originalBlockSize, // original full data size (before decomposition)
    size_t chunkBlockSize,    // block size used during compression
    uint8_t* finalReconstructed // pointer to preallocated destination buffer
) {
    // First, determine per-component uncompressed sizes.
    size_t totalBytesPerElement = 0;
    for (const auto &group : allComponentSizes)
        totalBytesPerElement += group.size();
    size_t numElements = originalBlockSize / totalBytesPerElement;

    // For each component, the expected size is:
    //   numElements * (component size)
    std::vector<std::vector<uint8_t>> decompressedComponents(compressedBlocks.size());

    double overallStart = omp_get_wtime();
    #pragma omp parallel for num_threads(numThreads)
    for (int compIdx = 0; compIdx < static_cast<int>(compressedBlocks.size()); compIdx++) {
        size_t compElementSize = allComponentSizes[compIdx].size();
        size_t expectedCompSize = numElements * compElementSize;
        std::vector<uint8_t> compDecompressed(expectedCompSize);
        size_t offset = 0;

        // Process each chunk sequentially for this component.
        for (const auto &chunkCompressed : compressedBlocks[compIdx]) {
            // Determine the original size of this chunk.
            size_t remaining = expectedCompSize - offset;
            size_t originalChunkSize = std::min(chunkBlockSize, remaining);
            // Decompress into the appropriate position.
            decompressWithFastLZ1(chunkCompressed, &compDecompressed[offset], originalChunkSize);
            offset += originalChunkSize;
        }
        decompressedComponents[compIdx] = compDecompressed;
    }
    double overallEnd = omp_get_wtime();
    pi.total_time_decompressed = overallEnd - overallStart;

    // Reassemble the full interleaved data from the decomposed components.
    reassembleBytesFromComponentsNested(decompressedComponents, finalReconstructed, originalBlockSize, allComponentSizes, numThreads);
}

// Utility: Compute Overall Compression Ratio.
inline double calculateCompressionRatio(size_t originalSize, size_t compressedSize) {
    return (compressedSize == 0)
        ? 0.0
        : static_cast<double>(originalSize) / static_cast<double>(compressedSize);
}

#endif // FASTLZ_PARALLEL_H