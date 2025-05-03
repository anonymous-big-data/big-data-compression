#ifndef PROFILING_INFO_H
#define PROFILING_INFO_H

#include <string>
#include <fstream>

#define PROFILING_INFO_H


struct ProfilingInfo {
  double com_ratio = 0.0;
  double total_time_compressed = 0.0;
  double split_time = 0.0;
  double compress_time=0.0;
  double total_time_decompressed = 0.0;
  std::string type;
  std::vector<double> component_times; // Store times for each component
  double compression_throughput = 0.0;
  double decompression_throughput = 0.0;
  double total_values = 0.0;
  int thread_count = 0; // Add thread count
  std::string config_string;

  // Constructor with default initialization
  ProfilingInfo(size_t num_components = 0)
      : component_times(num_components, 0.0) {}

  // Add const to this method
  void printCSV(std::ofstream &file, int iteration) const {
    file << iteration << ","
         << thread_count << "," // Add thread count to the CSV
         << type << ","
         << com_ratio << ","
         << total_time_compressed << ","
         << total_time_decompressed << ","
         << split_time << ","
         << compress_time << ",";

    // Append component times dynamically
    for (size_t i = 0; i < component_times.size(); ++i) {
      file << component_times[i];
      if (i < component_times.size() - 1) {
        file << ",";
      }
    }

    file << "," << compression_throughput << ","
         << decompression_throughput << ","
         << total_values << "\n";
  }
};


#endif // PROFILING_INFO_H
