cmake_minimum_required( VERSION 3.20)


project(big-data CXX C )


# ZSTD cloning
include(FetchContent)

set(ZSTD_BUILD_STATIC ON)
set(ZSTD_BUILD_SHARED OFF)
#set multithread on
set(ZSTD_MULTITHREAD_SUPPORT ON)

FetchContent_Declare(
        zstd
        URL "https://github.com/facebook/zstd/releases/download/v1.5.6/zstd-1.5.6.tar.gz"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_SUBDIR build/cmake
)

FetchContent_MakeAvailable(zstd)

#target_link_libraries(
#        ${PROJECT_NAME}
#        PRIVATE
#        libzstd_static
#)
#
## On windows and macos this is needed
#target_include_directories(
#        ${PROJECT_NAME}
#        PRIVATE
#        ${zstd_SOURCE_DIR}/lib
#)



option(PROFILING_ENABLED "Enables Modules for Profiling, needs LIBFPM4" OFF)

#FetchContent_Declare(
# googlebenchmark
#  GIT_REPOSITORY https://github.com/google/benchmark.git
#    GIT_TAG v1.9.0
#)
# enable BENCHMARK_DOWNLOAD_DEPENDENCIES
#set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)
#if (PROFILING_ENABLED)
    # enable PFM
# set(BENCHMARK_ENABLE_LIBPFM OFF)
#endif()
# disable tests
#set(BENCHMARK_ENABLE_TESTING OFF)

#FetchContent_MakeAvailable(googlebenchmark)

## Argparse
include(FetchContent)
FetchContent_Declare(
        argparse
        GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)


# find openmp
find_package(OpenMP REQUIRED)


add_subdirectory(external_tools)