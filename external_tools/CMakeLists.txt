set(CMAKE_CXX_STANDARD 17)
project(zstd-test-project CXX C)

# Set the C++ standard globally
set(CMAKE_CXX_STANDARD 17)
set(ZLIB_ROOT "/usr")
find_package(ZLIB REQUIRED)
FetchContent_Declare(
        cxxopts
        GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
        GIT_TAG v2.2.1
)
FetchContent_GetProperties(cxxopts)
if(NOT cxxopts_POPULATED)
    FetchContent_Populate(cxxopts)
    set(CXXOPTS_BUILD_TESTS OFF CACHE BOOL "Enable building of tests." FORCE)  # Disable test building for cxxopts
    add_subdirectory(${cxxopts_SOURCE_DIR} ${cxxopts_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Fetch LZ4 for compression
FetchContent_Declare(
        lz4
        GIT_REPOSITORY https://github.com/lz4/lz4.git
        GIT_TAG dev #  branch
)
FetchContent_GetProperties(lz4)
if(NOT lz4_POPULATED)
    FetchContent_Populate(lz4)
    add_subdirectory(${lz4_SOURCE_DIR}/build/cmake ${lz4_BINARY_DIR})
endif()


# Fetch Snappy for compression
FetchContent_Declare(
        snappy
        GIT_REPOSITORY https://github.com/google/snappy.git
        GIT_TAG main
)
FetchContent_GetProperties(snappy)
if(NOT snappy_POPULATED)
    FetchContent_Populate(snappy)
    set(SNAPPY_BUILD_BENCHMARKS OFF CACHE BOOL "Disable Snappy benchmarks" FORCE)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable Google Benchmark tests" FORCE)
    set(SNAPPY_BUILD_TESTS OFF CACHE BOOL "Disable Snappy tests" FORCE)
    add_subdirectory(${snappy_SOURCE_DIR} ${snappy_BINARY_DIR})
endif()

# Fetch Google Benchmark (ensure tests are disabled)
FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG main
)
FetchContent_GetProperties(benchmark)
if(NOT benchmark_POPULATED)
    FetchContent_Populate(benchmark)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable Google Benchmark tests" FORCE)
    add_subdirectory(${benchmark_SOURCE_DIR} ${benchmark_BINARY_DIR})
endif()

FetchContent_Declare(
        bzip2
        GIT_REPOSITORY https://gitlab.com/bzip2/bzip2.git
        GIT_TAG master
)

FetchContent_GetProperties(bzip2)
if(NOT bzip2_POPULATED)
    FetchContent_Populate(bzip2)

    # Generate bz_version.h
    set(BZIP2_VERSION "1.0.8") # Set the BZip2 version
    configure_file(
            ${bzip2_SOURCE_DIR}/bz_version.h.in
            ${bzip2_SOURCE_DIR}/bz_version.h
            @ONLY
    )

    # Create a static library for BZip2
    add_library(bzip2 STATIC
            ${bzip2_SOURCE_DIR}/blocksort.c
            ${bzip2_SOURCE_DIR}/huffman.c
            ${bzip2_SOURCE_DIR}/crctable.c
            ${bzip2_SOURCE_DIR}/randtable.c
            ${bzip2_SOURCE_DIR}/compress.c
            ${bzip2_SOURCE_DIR}/decompress.c
            ${bzip2_SOURCE_DIR}/bzlib.c
            zstd_parallel.h
    )
    target_include_directories(bzip2 PRIVATE ${bzip2_SOURCE_DIR})
endif()
add_executable(parallel-test
        #parralel-zstd.cpp
        zstd_parallel.h
        profiling_info.h
        #        FASTLZ_PARALLEL.h
        #        FASTLZ_PARALLEL.cpp
        decompose.h
        decompose-new1.cpp
        lz4_parallel.h
        # parallel-lz4.cpp
        snappy_parallel.h
        #  parallel-snappy.cpp
        zlib-parallel.h
        bzib_parallel.h
        #  zlib-parallel.cpp
        # bz2_parallel.h
        # parallel-bzip2.cpp
)
# Let the compiler search external_tools/profiles/ for #include "profiles/…"
target_include_directories(parallel-test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/profiles
)
add_library(fastlz STATIC fastlz.c
        bzib_parallel.h
        dataset_loaders.h
)
target_link_libraries(parallel-test PRIVATE fastlz)
target_link_libraries(parallel-test
        PRIVATE
        libzstd_static
        benchmark::benchmark
        OpenMP::OpenMP_CXX
        lz4
        snappy
        ZLIB::ZLIB
        bzip2
        # argparse can be replaced or used alongside cxxopts if required
)

target_include_directories(parallel-test
        PRIVATE
        ${zstd_SOURCE_DIR}/lib
        # Include cxxopts only for this target if desired
        ${cxxopts_SOURCE_DIR}/include
        ${lz4_SOURCE_DIR}/lib
        ${bzip2_SOURCE_DIR}
)
############################
set(HALF_HPP_DIR ${CMAKE_SOURCE_DIR}/external/half)
set(HALF_HPP_FILE ${HALF_HPP_DIR}/half.hpp)

file(MAKE_DIRECTORY ${HALF_HPP_DIR})

if(NOT EXISTS ${HALF_HPP_FILE})
    message(STATUS "Downloading half.hpp from GitHub...")
    file(DOWNLOAD
            https://raw.githubusercontent.com/halffloat/half/master/half.hpp
            ${HALF_HPP_FILE}
            SHOW_PROGRESS
            STATUS status
            LOG log
    )

    list(GET status 0 status_code)
    list(GET status 1 status_string)
    if(NOT status_code EQUAL 0)
        message(FATAL_ERROR "Failed to download half.hpp: ${status_string}\nLog:\n${log}")
    endif()
endif()

include_directories(${CMAKE_SOURCE_DIR}/external/half)


