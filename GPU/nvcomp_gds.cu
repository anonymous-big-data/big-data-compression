
#include <fcntl.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <iomanip>
#include <cstdlib>
#include <cstdint>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <map>
#include <functional>
#include <algorithm>
#include <memory>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "nvcomp.hpp"
#include "nvcomp/lz4.hpp"

#if __has_include("nvcomp/gdeflate.hpp")
  #include "nvcomp/gdeflate.hpp"
  #define HAVE_NVCOMP_GDEFLATE 1
#else
  #define HAVE_NVCOMP_GDEFLATE 0
#endif
#if __has_include("nvcomp/zstd.hpp")
  #include "nvcomp/zstd.hpp"
  #define HAVE_NVCOMP_ZSTD 1
#else
  #define HAVE_NVCOMP_ZSTD 0
#endif

using namespace nvcomp;
//---------------------------------
static inline int env_int(const char* name, int defv) {
  if (const char* s = std::getenv(name)) {
    try { return std::max(32, std::stoi(s)); } catch(...) {}
  }
  return defv;
}
static int TPB() { return env_int("TPB", 256); }  // Threads Per Block

//---------------------------------------------------------
// CUDA error checking
//---------------------------------------------------------
#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      std::cerr << "CUDA API call failure '" #func "' with error: "            \
                << cudaGetErrorString(rt) << " at " << __FILE__                \
                << ":" << __LINE__ << std::endl;                               \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

//---------------------------------------------------------
// Codec selector (env: NVCOMP_CODEC = lz4|gdeflate|zstd)
//---------------------------------------------------------
enum class Codec { LZ4, GDEFLATE, ZSTD };
static inline Codec parseCodec(const char* s)
{
  if (!s) return Codec::LZ4;
  std::string t = s;
  std::transform(t.begin(), t.end(), t.begin(), ::tolower);
  if (t == "gdeflate") return Codec::GDEFLATE;
  if (t == "zstd")     return Codec::ZSTD;
  return Codec::LZ4;
}
//---------------------------------------------------------
// --- helpers ---
static inline void trim_inplace(std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  size_t b = s.find_last_not_of(" \t\r\n");
  if (a == std::string::npos) { s.clear(); return; }
  s = s.substr(a, b - a + 1);
}

static inline bool parse_double_loose(const std::string& s, double& out) {
  errno = 0;
  char* end = nullptr;
  out = std::strtod(s.c_str(), &end);
  if (errno == ERANGE) return false;
  // skip trailing spaces
  while (end && *end==' ') ++end;
  return end && *end=='\0';
}


template <typename T>
std::pair<std::vector<T>, size_t> loadTSVDataset(const std::string &fp) {
  std::vector<T> arr;
  std::ifstream f(fp);
  std::string line;
  size_t rows = 0;
  if (!f.is_open()) throw std::runtime_error("Open failed: " + fp);

  while (std::getline(f, line)) {
    std::stringstream ss(line);
    std::string val;

    std::getline(ss, val, '\t');

    while (std::getline(ss, val, '\t')) {
      arr.push_back(static_cast<T>(std::stod(val)));
    }
    ++rows;
  }
  return {arr, rows};
}


////
template <typename T>
std::pair<std::vector<T>, size_t> loadTSVDataset11(const std::string &fp) {
  std::vector<T> arr;
  std::ifstream f(fp);
  if (!f.is_open()) throw std::runtime_error("Open failed: " + fp);

  std::string line;
  size_t rows = 0;
  bool header_checked = false;

  while (std::getline(f, line)) {
    // handle possible CRLF
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) { ++rows; continue; }


    std::replace(line.begin(), line.end(), ',', '\t');

    if (rows == 0 && line.size() >= 3 &&
        static_cast<unsigned char>(line[0]) == 0xEF &&
        static_cast<unsigned char>(line[1]) == 0xBB &&
        static_cast<unsigned char>(line[2]) == 0xBF) {
      line.erase(0,3);
    }

    std::stringstream ss(line);
    std::string val;

    std::getline(ss, val, '\t');

    if (!header_checked) {
      std::string peek = ss.str();
      std::streampos pos = ss.tellg();
      if (pos != std::streampos(-1)) peek = peek.substr(static_cast<size_t>(pos));
      if (peek.find_first_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") != std::string::npos) {
        ++rows; header_checked = true; continue;
      }
      header_checked = true;
    }

    while (std::getline(ss, val, '\t')) {
      trim_inplace(val);
      if (val.empty()) continue;

      if (val == "nan" || val == "NaN") {
        arr.push_back(static_cast<T>(std::numeric_limits<double>::quiet_NaN()));
        continue;
      }
      if (val == "inf" || val == "Inf" || val == "+inf" || val == "+Inf") {
        arr.push_back(static_cast<T>(std::numeric_limits<double>::infinity()));
        continue;
      }
      if (val == "-inf" || val == "-Inf") {
        arr.push_back(static_cast<T>(-std::numeric_limits<double>::infinity()));
        continue;
      }

      double d;
      if (parse_double_loose(val, d)) {
        arr.push_back(static_cast<T>(d));
      } else {
        continue;
      }
    }
    ++rows;
  }
  return {arr, rows};
}


//---------------------------------------------------------
// Helpers: TSV load, conversion
//---------------------------------------------------------
template <typename T>
std::pair<std::vector<T>, size_t> loadTSVDataset1(const std::string &fp) {
  std::vector<T> arr;
  std::ifstream f(fp);
  std::string line;
  size_t rows = 0;
  if (!f.is_open()) throw std::runtime_error("Open 1failed: " + fp);
  while (std::getline(f, line)) {
    std::stringstream ss(line);
    std::string val;
    std::getline(ss, val, '\t'); // skip first column
    while (std::getline(ss, val, '\t')) {
      arr.push_back(static_cast<T>(std::stod(val)));
    }
    ++rows;
  }
  return {arr, rows};
}

std::vector<uint8_t> toBytes(const std::vector<float> &v) {
  std::vector<uint8_t> b(v.size() * sizeof(float));
  std::memcpy(b.data(), v.data(), b.size());
  return b;
}
std::vector<uint8_t> toBytes(const std::vector<double> &v) {
  std::vector<uint8_t> b(v.size() * sizeof(double));
  std::memcpy(b.data(), v.data(), b.size());
  return b;
}

//---------------------------------------------------------
// Convert config to string
//---------------------------------------------------------
std::string configToString(const std::vector<std::vector<size_t>> &cfg) {
  std::ostringstream ss;
  ss << '"';
  for (size_t i = 0; i < cfg.size(); ++i) {
    ss << '[';
    for (size_t j = 0; j < cfg[i].size(); ++j) {
      ss << cfg[i][j];
      if (j + 1 < cfg[i].size()) ss << ',';
    }
    ss << ']';
    if (i + 1 < cfg.size()) ss << ',';
  }
  ss << '"';
  return ss.str();
}

//---------------------------------------------------------
// Split bytes into components
//---------------------------------------------------------
void splitBytesIntoComponents(
  const std::vector<uint8_t> &in,
  std::vector<std::vector<uint8_t>> &out,
  const std::vector<std::vector<size_t>> &cfg,
  int numThreads)
{
  size_t elemBytes = 0;
  for (auto &g : cfg) elemBytes += g.size();
  size_t elems = in.size() / elemBytes;

  out.assign(cfg.size(), {});
  for (size_t i = 0; i < cfg.size(); ++i)
    out[i].resize(elems * cfg[i].size());

#ifdef _OPENMP
  #pragma omp parallel for num_threads(numThreads)
#endif
  for (size_t e = 0; e < elems; ++e) {
    for (size_t c = 0; c < cfg.size(); ++c) {
      for (size_t j = 0; j < cfg[c].size(); ++j) {
        size_t idx = e * elemBytes + (cfg[c][j] - 1);
        out[c][e * cfg[c].size() + j] = in[idx];
      }
    }
  }
}

//---------------------------------------------------------
// Timing helper
//---------------------------------------------------------
float measureCudaTime(std::function<void(cudaStream_t)> fn, cudaStream_t s) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, s));
  fn(s);
  CUDA_CHECK(cudaEventRecord(stop, s));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms;
}

//---------------------------------------------------------
// Validation kernel
//---------------------------------------------------------
__global__
void compareBuffers(const uint8_t *a, const uint8_t *b, int *invalid, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (; idx < n; idx += stride) {
    if (a[idx] != b[idx]) *invalid = 1;
  }
}

//---------------------------------------------------------
// WHOLE-dataset helpers per codec
//---------------------------------------------------------
struct WholeMetrics {
  float tCompMs{}, tDecompMs{};
  size_t outBytes{};
  bool verified{};
};

static WholeMetrics runWholeLZ4(uint8_t* d_in, size_t totalBytes)
{
  WholeMetrics M{};
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  uint8_t *d_out = nullptr, *d_whole = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, totalBytes));

  nvcompBatchedLZ4Opts_t opts{NVCOMP_TYPE_CHAR};
  LZ4Manager mgr(1 << 16, opts, stream);
  auto cfg = mgr.configure_compression(totalBytes);
  const size_t maxWhole = ((cfg.max_compressed_buffer_size - 1) / 4096 + 1) * 4096;
  CUDA_CHECK(cudaMalloc(&d_whole, maxWhole));

  M.tCompMs = measureCudaTime([&](cudaStream_t s){ mgr.compress(d_in, d_whole, cfg); }, stream);
  M.outBytes = mgr.get_compressed_output_size(d_whole);

  auto dcfg = mgr.configure_decompression(cfg);
  M.tDecompMs = measureCudaTime([&](cudaStream_t s){ mgr.decompress(d_out, d_whole, dcfg); }, stream);

  int *h_invalid = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_invalid, sizeof(int)));
  *h_invalid = 0;
  compareBuffers<<<64,256,0,stream>>>(d_in, d_out, h_invalid, totalBytes);
 //-----------


//-------------
  CUDA_CHECK(cudaStreamSynchronize(stream));
  M.verified = (*h_invalid == 0);

  mgr.deallocate_gpu_mem();
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_whole));
  CUDA_CHECK(cudaFreeHost(h_invalid));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return M;
}

#if HAVE_NVCOMP_GDEFLATE
static WholeMetrics runWholeGdeflate(uint8_t* d_in, size_t totalBytes)
{
  WholeMetrics M{};
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  uint8_t *d_out = nullptr, *d_whole = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, totalBytes));

  nvcompBatchedGdeflateOpts_t opts{NVCOMP_TYPE_CHAR};
  GdeflateManager mgr(1 << 16, opts, stream);
  auto cfg = mgr.configure_compression(totalBytes);
  const size_t maxWhole = ((cfg.max_compressed_buffer_size - 1) / 4096 + 1) * 4096;
  CUDA_CHECK(cudaMalloc(&d_whole, maxWhole));

  M.tCompMs = measureCudaTime([&](cudaStream_t s){ mgr.compress(d_in, d_whole, cfg); }, stream);
  M.outBytes = mgr.get_compressed_output_size(d_whole);

  auto dcfg = mgr.configure_decompression(cfg);
  M.tDecompMs = measureCudaTime([&](cudaStream_t s){ mgr.decompress(d_out, d_whole, dcfg); }, stream);

  int *h_invalid = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_invalid, sizeof(int)));
  *h_invalid = 0;
   compareBuffers<<<64,256,0,stream>>>(d_in, d_out, h_invalid, totalBytes);
//-------------
//const int tpb = TPB();
// ...
//for (size_t i=0;i<N;++i) {
 // const size_t sz = components[i].size();
 // const int blocks = (int)((sz + tpb - 1) / tpb);
  //compareBuffers<<<blocks, tpb, 0, streams[i]>>>(d_in[i], d_out[i], d_invalid, sz);
//}

//--------------------
  CUDA_CHECK(cudaStreamSynchronize(stream));
  M.verified = (*h_invalid == 0);

  mgr.deallocate_gpu_mem();
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_whole));
  CUDA_CHECK(cudaFreeHost(h_invalid));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return M;
}
#endif

#if HAVE_NVCOMP_ZSTD
static WholeMetrics runWholeZstd(uint8_t* d_in, size_t totalBytes)
{
  WholeMetrics M{};
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  uint8_t *d_out = nullptr, *d_whole = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, totalBytes));

  nvcompBatchedZstdOpts_t opts{NVCOMP_TYPE_CHAR};
  ZstdManager mgr(1 << 16, opts, stream);
  auto cfg = mgr.configure_compression(totalBytes);
  const size_t maxWhole = ((cfg.max_compressed_buffer_size - 1) / 4096 + 1) * 4096;
  CUDA_CHECK(cudaMalloc(&d_whole, maxWhole));

  M.tCompMs = measureCudaTime([&](cudaStream_t s){ mgr.compress(d_in, d_whole, cfg); }, stream);
  M.outBytes = mgr.get_compressed_output_size(d_whole);

  auto dcfg = mgr.configure_decompression(cfg);
  M.tDecompMs = measureCudaTime([&](cudaStream_t s){ mgr.decompress(d_out, d_whole, dcfg); }, stream);

  int *h_invalid = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_invalid, sizeof(int)));
  *h_invalid = 0;
  compareBuffers<<<64,256,0,stream>>>(d_in, d_out, h_invalid, totalBytes);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  M.verified = (*h_invalid == 0);

  mgr.deallocate_gpu_mem();
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_whole));
  CUDA_CHECK(cudaFreeHost(h_invalid));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return M;
}
#endif

//---------------------------------------------------------
// Generic COMPONENT runner (works for LZ4/GDeflate/Zstd)
//---------------------------------------------------------
template <class ManagerT, class OptsT>
void run_components_with_manager(
    const std::vector<uint8_t>& globalBytes,
    const std::vector<std::vector<std::vector<size_t>>>& cfgList,
    const std::string& datasetName,
    int runId,
    const char* modeLabel,  // e.g., "Component(LZ4)"
    std::vector<std::string>& compRows)
{
  const size_t totalBytes = globalBytes.size();

  for (const auto& cfg : cfgList) {
    std::cout << "Selected config: " << configToString(cfg) << std::endl;

    // 1) Split host-side components
    std::vector<std::vector<uint8_t>> components;
    splitBytesIntoComponents(globalBytes, components, cfg, /*threads=*/16);
    const size_t N = components.size();

    // 2) Per-component stream + Manager
    std::vector<cudaStream_t> streams(N);
    std::vector<std::unique_ptr<ManagerT>> mgrs;
    mgrs.reserve(N);
    OptsT opts{NVCOMP_TYPE_CHAR};
    for (size_t i = 0; i < N; ++i) {
      CUDA_CHECK(cudaStreamCreate(&streams[i]));
      mgrs.emplace_back(new ManagerT(1<<16, opts, streams[i]));
    }

    // 3) Configure each
    using CompCfgT = decltype(std::declval<ManagerT>().configure_compression(size_t{}));
    std::vector<CompCfgT> cCfgs; cCfgs.reserve(N);
    for (size_t i = 0; i < N; ++i) {
      cCfgs.push_back(mgrs[i]->configure_compression(components[i].size()));
    }

    // 4) Allocate & copy
    std::vector<uint8_t*> d_in(N), d_buf(N), d_out(N);
    for (size_t i = 0; i < N; ++i) {
      const size_t sz     = components[i].size();
      const size_t maxBuf = ((cCfgs[i].max_compressed_buffer_size - 1)/4096 + 1)*4096;
      CUDA_CHECK(cudaMalloc(&d_in[i],  sz));
      CUDA_CHECK(cudaMalloc(&d_buf[i], maxBuf));
      CUDA_CHECK(cudaMalloc(&d_out[i], sz));
      CUDA_CHECK(cudaMemcpy(d_in[i], components[i].data(), sz, cudaMemcpyHostToDevice));
    }

    // 5) Compress all (timed)
    cudaEvent_t cStart, cEnd;
    CUDA_CHECK(cudaEventCreate(&cStart));
    CUDA_CHECK(cudaEventCreate(&cEnd));
    CUDA_CHECK(cudaEventRecord(cStart,0));
    for (size_t i=0;i<N;++i) {
      mgrs[i]->compress(d_in[i], d_buf[i], cCfgs[i]);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(cEnd,0));
    CUDA_CHECK(cudaEventSynchronize(cEnd));
    float tC_ms=0;
    CUDA_CHECK(cudaEventElapsedTime(&tC_ms, cStart, cEnd));
    CUDA_CHECK(cudaEventDestroy(cStart));
    CUDA_CHECK(cudaEventDestroy(cEnd));

    // 6) Sum compressed sizes
    size_t sumCompressed=0;
    for (size_t i=0;i<N;++i) sumCompressed += mgrs[i]->get_compressed_output_size(d_buf[i]);

    // 7) Decompress all (timed)
    cudaEvent_t dStart, dEnd;
    CUDA_CHECK(cudaEventCreate(&dStart));
    CUDA_CHECK(cudaEventCreate(&dEnd));
    CUDA_CHECK(cudaEventRecord(dStart,0));
    for (size_t i=0;i<N;++i) {
      auto dCfg = mgrs[i]->configure_decompression(cCfgs[i]);
      mgrs[i]->decompress(d_out[i], d_buf[i], dCfg);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(dEnd,0));
    CUDA_CHECK(cudaEventSynchronize(dEnd));
    float tD_ms=0;
    CUDA_CHECK(cudaEventElapsedTime(&tD_ms, dStart, dEnd));
    CUDA_CHECK(cudaEventDestroy(dStart));
    CUDA_CHECK(cudaEventDestroy(dEnd));

    // 8) Verify
    int *d_invalid=nullptr, h_invalid=0;
    CUDA_CHECK(cudaMalloc(&d_invalid,sizeof(int)));
    CUDA_CHECK(cudaMemset(d_invalid,0,sizeof(int)));
    for (size_t i=0;i<N;++i) {
      const size_t sz = components[i].size();
      const int blocks = (sz+255)/256;
      compareBuffers<<<blocks,256,0,streams[i]>>>(d_in[i], d_out[i], d_invalid, sz);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_invalid, d_invalid, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_invalid));
    if (h_invalid) std::cerr<<"ERROR: Component-based mismatch!\n";
    else           std::cout<<"Component-based decompression verified.\n";

    // 9) Metrics row
    const double thrC = (totalBytes/1e6)/tC_ms;
    const double thrD = (totalBytes/1e6)/tD_ms;
    const double ratio = double(totalBytes)/std::max<size_t>(sumCompressed,1);

    std::ostringstream row;
    row << runId              << ","
        << datasetName        << ","
        << modeLabel          << ","    // e.g., Component(LZ4)
        << totalBytes         << ","
        << sumCompressed      << ","
        << std::fixed << std::setprecision(2)
        << ratio              << ","
        << tC_ms              << ","
        << tD_ms              << ","
        << thrC               << ","
        << thrD               << ","
        << configToString(cfg);

    compRows.push_back(row.str());

    // 10) Cleanup
    for (size_t i=0;i<N;++i) {
      mgrs[i]->deallocate_gpu_mem();
      CUDA_CHECK(cudaFree(d_in[i]));
      CUDA_CHECK(cudaFree(d_buf[i]));
      CUDA_CHECK(cudaFree(d_out[i]));
      CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
  }
}

//---------------------------------------------------------
// Process one dataset
//---------------------------------------------------------
using ComponentConfig = std::vector<std::vector<std::vector<size_t>>>;

int runSingleDataset(const std::string &path, int precisionBits, int runId){

  // Load dataset and convert to bytes
  std::vector<uint8_t> globalBytes;
  if (precisionBits == 64) {
    auto tmp = loadTSVDataset<double>(path);
    globalBytes = toBytes(tmp.first);
  } else {
    auto tmp = loadTSVDataset<float>(path);
    globalBytes = toBytes(tmp.first);
  }
  const size_t totalBytes = globalBytes.size();

  // Derive dataset name & CSV filename
  std::string datasetName = path;
  if (auto p = datasetName.find_last_of("/\\"); p != std::string::npos)
    datasetName = datasetName.substr(p + 1);
  if (auto d = datasetName.find_last_of('.'); d != std::string::npos)
    datasetName = datasetName.substr(0, d);
  std::string csvFilename = "/home/jamalids/Documents/" + datasetName + ".csv";

  // --- WHOLE: choose codec via env (default LZ4) ---
  cudaStream_t copyStream;
  CUDA_CHECK(cudaStreamCreate(&copyStream));
  uint8_t *d_in = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, totalBytes));
  CUDA_CHECK(cudaMemcpyAsync(d_in, globalBytes.data(), totalBytes, cudaMemcpyHostToDevice, copyStream));
  CUDA_CHECK(cudaStreamSynchronize(copyStream));
  CUDA_CHECK(cudaStreamDestroy(copyStream));

  Codec codec = parseCodec(std::getenv("NVCOMP_CODEC"));
  WholeMetrics WM{};

  if (codec == Codec::LZ4) {
    WM = runWholeLZ4(d_in, totalBytes);
    std::cout << "Whole(LZ4) " << (WM.verified ? "PASS\n" : "FAIL\n");
  }
#if HAVE_NVCOMP_GDEFLATE
  else if (codec == Codec::GDEFLATE) {
    WM = runWholeGdeflate(d_in, totalBytes);
    std::cout << "Whole(GDeflate) " << (WM.verified ? "PASS\n" : "FAIL\n");
  }
#endif
#if HAVE_NVCOMP_ZSTD
  else if (codec == Codec::ZSTD) {
    WM = runWholeZstd(d_in, totalBytes);
    std::cout << "Whole(Zstd) " << (WM.verified ? "PASS\n" : "FAIL\n");
  }
#endif
  else {
    WM = runWholeLZ4(d_in, totalBytes);
    std::cout << "Whole(LZ4) " << (WM.verified ? "PASS\n" : "FAIL\n");
  }

  const double thrC1 = (totalBytes/1e6)/WM.tCompMs;
  const double thrD1 = (totalBytes/1e6)/WM.tDecompMs;
  const double rat1  = double(totalBytes)/std::max<size_t>(WM.outBytes,1);

  CUDA_CHECK(cudaFree(d_in));

  // ---------- Component configs ----------
  ComponentConfig acs_wht_f32   = { { {1,2}, {3}, {4} } };
  ComponentConfig citytemp_f32  = { { {1,2}, {3}, {4} } };
  ComponentConfig hdr_night_f32 = { { {1,2}, {3}, {4} } };
  ComponentConfig hdr_palermo_f32 = { { {1,2}, {3}, {4} } };
  ComponentConfig hst_wfc3_ir_f32 = { { {1,2}, {3}, {4} } };
  ComponentConfig hst_wfc3_uvis_f32 = { { {1,2,3}, {4} } };
  ComponentConfig jw_mirimage_f32 = { { {1,2,3}, {4} } };
  ComponentConfig rsim_f32 = { { {1,2,3}, {4} } };
  ComponentConfig solar_wind_f32 = { { {1,2}, {3}, {4} } };
  ComponentConfig spitzer_irac_f32 = { { {1,2,3}, {4} } };
  ComponentConfig tpcds_catalog_f32 = { { {1}, {2}, {3}, {4} } };
  ComponentConfig tpcds_store_f32 = { { {1,2}, {3}, {4} } };
  ComponentConfig tpcds_web_f32 = { { {1}, {2}, {3}, {4} } };
  ComponentConfig tpch_lineitem_f32 = { { {1,2,3}, {4} } };
  ComponentConfig wave_f32 = { { {1,2,3}, {4} } };
  ComponentConfig def_f32 = { { {1}, {2}, {3}, {4} } };

  // 64-bit configs
  ComponentConfig astro_mhd_f64 = { { {1,2,3,4,5,6}, {7,8} } };
  ComponentConfig tpch_order_f64 = { { {5,6}, {1,2,3,4}, {7}, {8} } };
  ComponentConfig astro_pt_f64 = { { {4,5}, {1,2,3}, {6}, {7}, {8} } };
  ComponentConfig wesad_chest_f64 = { { {1,2,3,4,8}, {5,6,7} } };
  ComponentConfig phone_gyro_f64 = { { {1,2,3,4,5,6,7}, {8} } };
  ComponentConfig tpcxbb_store_f64 = { { {6,7}, {1,2,3,4,5}, {8} } };
  ComponentConfig num_brain_f64 = { { {2,5}, {1,3,4}, {6}, {7}, {8} },
                                    { {1,2}, {3}, {6}, {4}, {5}, {7}, {8} } };
  ComponentConfig num_control_f64 ={{{1,2}, {3,6}, {4}, {5}, {7}, {8}}};
  ComponentConfig msg_bt_f64 = { { {2,3}, {1}, {4}, {5}, {6}, {7}, {8} } };
  ComponentConfig tpcxbb_web_f64 = { { {6,7}, {1,2,3,4,5}, {8} } };
  ComponentConfig nyc_taxi2015_f64 = { { {1,2,3,8}, {4,5,6,7} } };
  ComponentConfig cms1_tw = { { {1,2,3,4,5,6,7}, {8} } };
  ComponentConfig cms25_tw = { { {1,2,3,4,5,6,7}, {8} } };
  ComponentConfig cms9_tw = { { {1,2,3,4,5,6,8}, {7} } };
  ComponentConfig gov30_tw= { { {6,7}, {1,2,3,4,5,8} } };
  ComponentConfig gov40_tw= { { {5,6,7}, {1,2,3,4,8} } };
  ComponentConfig poi_lat_tw = { { {1,2,3,4,5,6}, {7},{8} } };
  ComponentConfig poi_lon_tw = { { {1,2,3,4,5,6,7}, {8} } };

  std::vector<std::pair<std::string, ComponentConfig>> candidateConfigs = {
    {"acs_wht_f32", acs_wht_f32},
    {"citytemp_f32", citytemp_f32},
    {"hdr_night_f32", hdr_night_f32},
    {"hdr_palermo_f32", hdr_palermo_f32},
    {"hst_wfc3_ir_f32", hst_wfc3_ir_f32},
    {"hst_wfc3_uvis_f32", hst_wfc3_uvis_f32},
    {"jw_mirimage_f32", jw_mirimage_f32},
    {"rsim_f32", rsim_f32},
    {"solar_wind_f32", solar_wind_f32},
    {"spitzer_irac_f32", spitzer_irac_f32},
    {"tpcds_catalog_f32", tpcds_catalog_f32},
    {"tpcds_store_f32", tpcds_store_f32},
    {"tpcds_web_f32", tpcds_web_f32},
    {"tpch_lineitem_f32", tpch_lineitem_f32},
    {"wave_f32", wave_f32},
    {"default", def_f32},
    {"astro_mhd_f64", astro_mhd_f64},
    {"tpch_order_f64", tpch_order_f64},
    {"astro_pt_f64", astro_pt_f64},
    {"wesad_chest_f64", wesad_chest_f64},
    {"phone_gyro_f64", phone_gyro_f64},
    {"tpcxbb_store_f64", tpcxbb_store_f64},
    {"num_brain_f64", num_brain_f64},
    {"num_control_f64",num_control_f64},
    {"msg_bt_f64", msg_bt_f64},
    {"tpcxbb_web_f64", tpcxbb_web_f64},
    {"nyc_taxi2015_f64", nyc_taxi2015_f64},
{"cms1_tw",  cms1_tw},
{"cms9_tw",  cms9_tw},
{"cms25_tw", cms25_tw},
{"gov30_tw", gov30_tw},
{"gov40_tw", gov40_tw},
{"poi_lat_tw", poi_lat_tw},
{"poi_lon_tw", poi_lon_tw},
  };

  // choose config list for datasetName
  const ComponentConfig* cfgListPtr = nullptr;
  for (auto &e : candidateConfigs) {
    if (e.first == datasetName) { cfgListPtr = &e.second; break; }
  }
  if (!cfgListPtr) {
    for (auto &e : candidateConfigs)
      if (e.first == "default") { cfgListPtr = &e.second; break; }
  }
  const auto &cfgList = *cfgListPtr;

  // ---------- COMPONENT: follow NVCOMP_CODEC ----------
  std::vector<std::string> compRows, blockRows;
  if (codec == Codec::LZ4) {
    run_components_with_manager<LZ4Manager, nvcompBatchedLZ4Opts_t>(
        globalBytes, cfgList, datasetName, runId, "Component(LZ4)", compRows);
  }
#if HAVE_NVCOMP_GDEFLATE
  else if (codec == Codec::GDEFLATE) {
    run_components_with_manager<GdeflateManager, nvcompBatchedGdeflateOpts_t>(
        globalBytes, cfgList, datasetName, runId, "Component(GDeflate)", compRows);
  }
#endif
#if HAVE_NVCOMP_ZSTD
  else if (codec == Codec::ZSTD) {
    run_components_with_manager<ZstdManager, nvcompBatchedZstdOpts_t>(
        globalBytes, cfgList, datasetName, runId, "Component(Zstd)", compRows);
  }
#endif
  else {
    run_components_with_manager<LZ4Manager, nvcompBatchedLZ4Opts_t>(
        globalBytes, cfgList, datasetName, runId, "Component(LZ4)", compRows);
  }

  // --- Write all results to CSV (Whole row + Component rows) ---
  std::ofstream csv(csvFilename, std::ios::app);
  const char* wholeLabel =
    (codec==Codec::LZ4) ? "Whole(LZ4)" :
    (codec==Codec::GDEFLATE) ? "Whole(GDeflate)" :
    (codec==Codec::ZSTD) ? "Whole(Zstd)" : "Whole";
  csv << runId << "," << datasetName << "," << wholeLabel << "," << totalBytes << "," << WM.outBytes << ","
      << std::fixed << std::setprecision(2) << rat1 << ","
      << WM.tCompMs << "," << WM.tDecompMs << ","
      << thrC1 << "," << thrD1 << ",\"whole\"\n";
  for (auto &r : compRows)  csv << r << "\n";
  for (auto &r : blockRows) csv << r << "\n";
  csv.close();

  return EXIT_SUCCESS;
}

//---------------------------------------------------------
// Main
//---------------------------------------------------------
bool isDirectory(const std::string &p) {
  struct stat sb;
  return stat(p.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode);
}

std::vector<std::string> getAllTsvFiles(const std::string &folder) {
  std::vector<std::string> paths;
  DIR *dp = opendir(folder.c_str());
  if (!dp) return paths;
  struct dirent *de;
  while ((de = readdir(dp))) {
    std::string f(de->d_name);
    if (f.size() > 4 && f.substr(f.size() - 4) == ".tsv") {
      std::string full = folder + "/" + f;
      struct stat sb;
      if (stat(full.c_str(), &sb) == 0 && S_ISREG(sb.st_mode))
        paths.push_back(full);
    }
  }
  closedir(dp);
  return paths;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <file|folder> <32|64>\n";
    return EXIT_FAILURE;
  }
  std::string path = argv[1];
  int prec = std::stoi(argv[2]);

  const int runs = 5;

  if (isDirectory(path)) {
    for (auto &f : getAllTsvFiles(path)) {
      std::string datasetName = f;
      if (auto p = datasetName.find_last_of("/\\"); p != std::string::npos)
        datasetName = datasetName.substr(p + 1);
      if (auto d = datasetName.find_last_of('.'); d != std::string::npos)
        datasetName = datasetName.substr(0, d);
      std::string csvFilename = "/home/jamalids/Documents/" + datasetName + ".csv";

      std::ofstream csv(csvFilename);
      csv << "RUN,Dataset,Mode,TotalBytes,CompressedBytes,Ratio,CompTime,DecompTime,"
             "CompThroughput,DecompThroughput,Config\n";
      csv.close();

      for (int r = 1; r <= runs; ++r) {
        std::cout << "=== Run " << r << " for " << f << " ===\n";
        runSingleDataset(f, prec, r);
      }
    }
  } else {
    std::string datasetName = path;
    if (auto p = datasetName.find_last_of("/\\"); p != std::string::npos)
      datasetName = datasetName.substr(p + 1);
    if (auto d = datasetName.find_last_of('.'); d != std::string::npos)
      datasetName = datasetName.substr(0, d);
    std::string csvFilename = "/home/jamalids/Documents/" + datasetName + ".csv";

    std::ofstream csv(csvFilename);
    csv << "RUN,Dataset,Mode,TotalBytes,CompressedBytes,Ratio,CompTime,DecompTime,"
           "CompThroughput,DecompThroughput,Config\n";
    csv.close();

    for (int r = 1; r <= runs; ++r) {
      std::cout << "=== Run " << r << " for " << path << " ===\n";
      runSingleDataset(path, prec, r);
    }
  }
  return 0;
}


