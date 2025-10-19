
#pragma once
#include "util.h"
#include "BatchData.h"  

class BatchData;

class BatchDataCPU
{
public:
  // 1) Constructor from host data (chunk-based)
  BatchDataCPU(const std::vector<std::vector<char>>& host_data, size_t chunk_size);

  // 2) Constructor for fixed-size allocation
  BatchDataCPU(size_t max_output_size, size_t batch_size);

  // 3) Construct from device pointers (copy data from GPU if requested)
  BatchDataCPU(const void* const* in_ptrs, const size_t* in_sizes, const uint8_t* in_data, size_t in_size, bool copy_data = false);

  // 4) Move Constructor
  BatchDataCPU(BatchDataCPU&& other) = default;

  // 5) Construct from GPU BatchData; declare only here.
  BatchDataCPU(const BatchData& batch_data, bool copy_data = false);

  // 6) Custom Copy Constructor and Assignment Operator
  BatchDataCPU(const BatchDataCPU& other);
  BatchDataCPU& operator=(const BatchDataCPU& other);

  // Accessors
  uint8_t* data();
  const uint8_t* data() const;
  void** ptrs();
  const void* const* ptrs() const;
  size_t* sizes();
  const size_t* sizes() const;
  size_t size() const;

private:
  std::vector<void*>   m_ptrs;
  std::vector<size_t>  m_sizes;
  std::vector<uint8_t> m_data;
  size_t               m_size;
};

inline bool operator==(const BatchDataCPU& lhs, const BatchDataCPU& rhs)
{
  if (lhs.size() != rhs.size())
    return false;
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs.sizes()[i] != rhs.sizes()[i])
      return false;
    const uint8_t* lhs_ptr = reinterpret_cast<const uint8_t*>(lhs.ptrs()[i]);
    const uint8_t* rhs_ptr = reinterpret_cast<const uint8_t*>(rhs.ptrs()[i]);
    for (size_t j = 0; j < lhs.sizes()[i]; ++j)
      if (lhs_ptr[j] != rhs_ptr[j])
        return false;
  }
  return true;
}

