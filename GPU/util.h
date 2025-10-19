/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
*/

#pragma once

#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string.h>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      std::cout << "API call failure \"" #func "\" with " << rt << " at "      \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      throw;                                                                   \
    }                                                                          \
  } while (0)

size_t compute_batch_size(
    const std::vector<std::vector<char>>& data, const size_t chunk_size)
{
  size_t batch_size = 0;
  for (size_t i = 0; i < data.size(); ++i) {
    const size_t num_chunks = (data[i].size() + chunk_size - 1) / chunk_size;
    batch_size += num_chunks;
  }

  return batch_size;
}

std::vector<size_t> compute_chunk_sizes(
    const std::vector<std::vector<char>>& data,
    const size_t batch_size,
    const size_t chunk_size)
{
  std::vector<size_t> sizes(batch_size, chunk_size);

  size_t offset = 0;
  for (size_t i = 0; i < data.size(); ++i) {
    const size_t num_chunks = (data[i].size() + chunk_size - 1) / chunk_size;
    offset += num_chunks;
    if (data[i].size() % chunk_size != 0) {
      sizes[offset - 1] = data[i].size() % chunk_size;
    }
  }
  return sizes;
}

std::vector<void*> get_input_ptrs(
    const std::vector<std::vector<char>>& data,
    const size_t batch_size,
    const size_t chunk_size)
{
  std::vector<void*> input_ptrs(batch_size);
  size_t chunk = 0;
  for (size_t i = 0; i < data.size(); ++i) {
    const size_t num_chunks = (data[i].size() + chunk_size - 1) / chunk_size;
    for (size_t j = 0; j < num_chunks; ++j)
      input_ptrs[chunk++] = const_cast<void*>(
          static_cast<const void*>(data[i].data() + j * chunk_size));
  }
  return input_ptrs;
}


//
//// TODO: add clustering here in a generic term
//std::vector<std::vector<char>>
//read_and_cluster_chunk(const std::string& filename, const std::vector<int>& cluster_sizes){
//  // TODO read the file
//
//  // Hack, generate random data of floats instead of reading from file, fix this
//  int data_size = 400000;
//  std::vector<float> data = aligned_vector<float>(data_size);;
//  for (int i = 0; i < data_size; i++){
//    float f = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
//    data[i] = f;
//  }
//  // TODO: Cluster the data, please use a generic approach.
//  std::vector<std::vector<char>> clusters;
//  // see data as an array of chars
//  auto data_char = reinterpret_cast<char*>(data.data());
//  bool clustering_enabled = true; // This is a switch to test clustering or not
//  if(clustering_enabled){
//    // for each 4 bytes, bytes 0 and 1 one cluster, byte 2 is another cluster, byte 3 is another cluster
//    // This is a size smaller than 65536, which is the maximum size for output buffer in libdeflate
//    int chunk = 60000;
//    std::vector<char> cluster1 = aligned_vector<char>(data_size*2); //cluster1.reserve(data.size() * 2);
//    std::vector<char> cluster2 = aligned_vector<char>(data_size); //cluster2.reserve(data.size());
//    std::vector<char> cluster3 = aligned_vector<char>(data_size); //cluster3.reserve(data.size());
//    int cnt1 = 0, cnt2 = 0, cnt3 = 0;
//    for(int i = 0; i < 4*data.size(); i+=4){
//      cluster1[cnt1++] = data_char[i];
//      cluster1[cnt1++] = data_char[i+1];
//      cluster2[cnt2++] = data_char[i+2];
//      cluster3[cnt3++] = data_char[i+3];
//    }
//    int num_chunk1 = (cluster1.size()) / chunk;
//    std::vector<char> buffer1 = aligned_vector<char>(chunk);
//    for (int j = 0; j < num_chunk1; j++){
//      std::copy(cluster1.begin() + j*chunk, cluster1.begin() + (j+1)*chunk, buffer1.begin());
//      clusters.push_back(buffer1);
//    }
//    // remaining
//    buffer1.clear();
//    buffer1 = aligned_vector<char>(cluster1.size() - num_chunk1*chunk);
//    std::copy(cluster1.begin() + num_chunk1*chunk, cluster1.end(), buffer1.begin());
//    clusters.push_back(buffer1);
//
//    int num_chunk2 = (cluster2.size()) / chunk;
//    std::vector<char> buffer2 = aligned_vector<char>(chunk);
//    for (int j = 0; j < num_chunk2; j++){
//      std::copy(cluster2.begin() + j*chunk, cluster2.begin() + (j+1)*chunk, buffer2.begin());
//      clusters.push_back(buffer2);
//    }
//    // remaining
//    buffer2.clear();
//    buffer2 = aligned_vector<char>(cluster2.size() - num_chunk2*chunk);
//    std::copy(cluster2.begin() + num_chunk2*chunk, cluster2.end(), buffer2.begin());
//    clusters.push_back(buffer2);
//
//    int num_chunk3 = (cluster3.size()) / chunk;
//    std::vector<char> buffer3 = aligned_vector<char>(chunk);
//    for (int j = 0; j < num_chunk3; j++){
//      std::copy(cluster3.begin() + j*chunk, cluster3.begin() + (j+1)*chunk, buffer3.begin());
//      clusters.push_back(buffer3);
//    }
//    // remaining
//    buffer3.clear();
//    buffer3 = aligned_vector<char>(cluster3.size() - num_chunk3*chunk);
//    std::copy(cluster3.begin() + num_chunk3*chunk, cluster3.end(), buffer3.begin());
//    clusters.push_back(buffer3);
//
//  } else {
//    clusters.push_back(std::vector<char>(data_char, data_char + (4*data.size()) ));
//  }
//  return clusters;
//}