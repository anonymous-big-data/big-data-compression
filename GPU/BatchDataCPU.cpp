//
// Created by jamalids on 15/04/25.
//
#include "BatchDataCPU.h"
#include "BatchData.h"  // This must provide the complete definition of BatchData.
#include <numeric>
#include <cstring>
#include <stdexcept>
#include <cuda_runtime.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      throw std::runtime_error("CUDA API call failure: " +                           \
          std::string(cudaGetErrorString(rt)));                              \
    }                                                                          \
  } while (0)
#endif

// Example implementations for compute_batch_size, compute_chunk_sizes, get_input_ptrs
// These must be defined somewhere; here we provide stubs for a complete example.
size_t compute_batch_size(const std::vector<std::vector<char>>& host_data, size_t chunk_size)
{
  // Stub: For example, assume each file is partitioned into chunks of size chunk_size.
  // Here we simply return the total number of chunks across all files.
  size_t totalChunks = 0;
  for (const auto& file : host_data)
  {
    totalChunks += (file.size() + chunk_size - 1) / chunk_size;
  }
  return totalChunks;
}

std::vector<size_t> compute_chunk_sizes(const std::vector<std::vector<char>>& host_data, size_t batch_size, size_t chunk_size)
{
  // Stub: Returns a vector with each chunk size equal to chunk_size,
  // except possibly for the last chunk. For simplicity, we assume all chunks are full.
  return std::vector<size_t>(batch_size, chunk_size);
}

std::vector<void*> get_input_ptrs(const std::vector<std::vector<char>>& host_data, size_t batch_size, size_t chunk_size)
{
  // Stub: Allocate pointers into host_data (this is very simplified)
  std::vector<void*> ptrs(batch_size, nullptr);
  // In a real implementation, you would compute the start of each chunk in each file.
  return ptrs;
}

// Implement the constructor that initializes a BatchDataCPU from GPU BatchData.
BatchDataCPU::BatchDataCPU(const BatchData& batch_data, bool copy_data)
  : m_size(batch_data.size())
{
  // Retrieve chunk sizes from the GPU batch (assumes batch_data.sizes() returns a device pointer)
  m_sizes.resize(m_size);
  CUDA_CHECK(cudaMemcpy(m_sizes.data(), batch_data.sizes(), m_size * sizeof(size_t), cudaMemcpyDeviceToHost));

  // Compute total size and allocate m_data accordingly.
  size_t total_data_size = std::accumulate(m_sizes.begin(), m_sizes.end(), static_cast<size_t>(0));
  m_data.resize(total_data_size);

  // Build m_ptrs to point into m_data.
  m_ptrs.resize(m_size);
  size_t offset = 0;
  for (size_t i = 0; i < m_size; i++) {
    m_ptrs[i] = m_data.data() + offset;
    offset += m_sizes[i];
  }

  // Optionally, copy the actual data from the GPU.
  if (copy_data) {
    std::vector<void*> hostPtrArray(m_size);
    CUDA_CHECK(cudaMemcpy(hostPtrArray.data(), batch_data.ptrs(), m_size * sizeof(void*), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < m_size; i++) {
      CUDA_CHECK(cudaMemcpy(m_ptrs[i], hostPtrArray[i], m_sizes[i], cudaMemcpyDeviceToHost));
    }
  }
}
