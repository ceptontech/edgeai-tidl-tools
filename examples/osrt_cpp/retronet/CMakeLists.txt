cmake_minimum_required(VERSION 3.0.2)

set(PROJ_NAME retronet_main)

project(${PROJ_NAME})

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/common.cmake)

# sTIDL_IOBufDesc_t (itidl_ti.h) is sized per SOC; the artifact check in
# retronet_main.cpp relies on it. TIDL_IMPORT_TOOL skips the ivision.h include,
# only the structure definitions are needed here.
add_compile_definitions(SOC_J722S TIDL_IMPORT_TOOL)
include_directories(${TARGET_FS_PATH}/usr/include/processor_sdk/tidl_j7/arm-tidl/rt/inc
                    /usr/include/processor_sdk/tidl_j7/arm-tidl/rt/inc)

# No image pre/post-processing, so build_app() and its post_process/utils
# libraries (which need the TFLite headers) are not used.
add_executable(${PROJ_NAME} retronet_main.cpp)
# The targetfs libc/libm .so files are linker scripts with absolute paths
# (/usr/lib/libm.so.6), so a cross link has to treat the targetfs as sysroot.
if((${HOST_CPU} STREQUAL "x86") AND (${TARGET_CPU} STREQUAL "arm"))
  target_link_options(${PROJ_NAME} PRIVATE --sysroot=${TARGET_FS_PATH})
endif()
target_link_libraries(${PROJ_NAME}
                      -Wl,--unresolved-symbols=ignore-in-shared-libs
                      onnxruntime
                      vx_tidl_rt
                      pthread
                      dl)
