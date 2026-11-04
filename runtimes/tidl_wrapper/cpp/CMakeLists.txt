cmake_minimum_required(VERSION 3.10.0)

project(tidl_wrapper)

option(BUILD_TVMRT "Build the TVM runtime wrapper" ON)
option(ORT_TI_BENCHMARK_API "Target ONNX Runtime provides OrtSessionGetTIBenchmarkData_Tidl" ON)

if(NOT ORT_TI_BENCHMARK_API)
  add_compile_definitions(ORT_NO_TI_BENCHMARK_API)
  message(STATUS "ORT_TI_BENCHMARK_API=OFF: ONNX performance counters disabled")
endif()

include(${CMAKE_SOURCE_DIR}/../../cmake/common.cmake)

include_directories(${CMAKE_SOURCE_DIR}
                    ${CMAKE_SOURCE_DIR}/common
                    ${CMAKE_SOURCE_DIR}/onnxrt
                    ${CMAKE_SOURCE_DIR}/tflitert
                    ${CMAKE_SOURCE_DIR}/tidlrt)

if(BUILD_TVMRT)
  include_directories(${CMAKE_SOURCE_DIR}/tvmrt)
endif()

set(WRAPPER_SRCS
    ${WRAPPER_SRCS}
    onnxrt/onnxrt_wrapper.cpp
    tflitert/tflitert_wrapper.cpp
    tidlrt/tidlrt_wrapper.cpp)

if(BUILD_TVMRT)
  list(APPEND WRAPPER_SRCS tvmrt/tvmrt_wrapper.cpp)
else()
  message(STATUS "BUILD_TVMRT=OFF: skipping TVM runtime wrapper")
endif()

build_lib(${PROJECT_NAME}   # Named argument: library name
          STATIC            # Named argument: library type
          1.0.0             # Named argument: library version
          ${WRAPPER_SRCS})