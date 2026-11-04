cmake_minimum_required(VERSION 3.10.0)

option(BUILD_BASIC_EXAMPLE "Build Basic Example" ON)
option(BUILD_PREEMPTION_EXAMPLE "Build Preemption Example" ON)
option(BUILD_RETRONET_EXAMPLE "Build Retronet Example" ON)
option(BUILD_TVMRT "Build the TVM runtime wrapper" ON)

if(BUILD_TVMRT)
    add_compile_definitions(BUILD_TVMRT)
else()
    message(STATUS "BUILD_TVMRT=OFF: TVM runtime excluded from examples")
endif()

project(examples)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/common.cmake)


if((${HOST_CPU} STREQUAL  "x86") AND (${TARGET_CPU}  STREQUAL  "x86"))
    message(STATUS "Directory of CNPY: ${CMAKE_CURRENT_LIST_DIR}/../../../tools/cnpy/")
    if (EXISTS ${CMAKE_CURRENT_LIST_DIR}/../../../tools/cnpy/)
        set(CNPY_INSTALL_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../tools/cnpy/)
    else()
        message (FATAL_ERROR  "CNPY not found")
    endif()

    include_directories(${CMAKE_SOURCE_DIR}
                        ${CMAKE_SOURCE_DIR}/utils
                        ${CMAKE_SOURCE_DIR}/../../tidl_wrapper/cpp/common
                        ${CMAKE_SOURCE_DIR}/../../tidl_wrapper/cpp/onnxrt
                        ${CMAKE_SOURCE_DIR}/../../tidl_wrapper/cpp/tflitert
                        ${CMAKE_SOURCE_DIR}/../../tidl_wrapper/cpp/tvmrt
                        ${CMAKE_SOURCE_DIR}/../../tidl_wrapper/cpp/tidlrt
                        ${CNPY_INSTALL_DIR}/include
                        )

    # Update link directories to include both Release and Debug builds
    link_directories(${CMAKE_SOURCE_DIR}/../../tidl_wrapper/cpp/lib/${CMAKE_BUILD_TYPE}
                    ${CNPY_INSTALL_DIR}/lib
                    )

    # Define the libraries to link against
    set(COMMON_LINK_LIBS
        ${COMMON_LINK_LIBS}
        tidl_wrapper
        cnpy
        )

    set(SYSTEM_LINK_LIBS
        ${SYSTEM_LINK_LIBS}
        yaml-cpp
        vx_tidl_rt
        png16
        dl
        )
elseif((${HOST_CPU} STREQUAL  "aarch64") AND (${TARGET_CPU}  STREQUAL  "aarch64"))
    include_directories(${CMAKE_SOURCE_DIR}
                        ${CMAKE_SOURCE_DIR}/utils
                        ${CMAKE_SOURCE_DIR}/../../tidl_wrapper/cpp/common
                        ${CMAKE_SOURCE_DIR}/../../tidl_wrapper/cpp/onnxrt
                        ${CMAKE_SOURCE_DIR}/../../tidl_wrapper/cpp/tflitert
                        ${CMAKE_SOURCE_DIR}/../../tidl_wrapper/cpp/tvmrt
                        ${CMAKE_SOURCE_DIR}/../../tidl_wrapper/cpp/tidlrt
                        )

    # Update link directories to include both Release and Debug builds
    link_directories(${CMAKE_SOURCE_DIR}/../../tidl_wrapper/cpp/lib/${CMAKE_BUILD_TYPE})

    # Define the libraries to link against
    set(COMMON_LINK_LIBS
        ${COMMON_LINK_LIBS}
        tidl_wrapper
        )

    set(SYSTEM_LINK_LIBS
        ${SYSTEM_LINK_LIBS}
        yaml-cpp
        vx_tidl_rt
        png16
        dl
        cnpy
        )
elseif((${HOST_CPU} STREQUAL  "x86") AND (${TARGET_CPU}  STREQUAL  "aarch64"))
        include_directories(${CMAKE_SOURCE_DIR}
                            ${CMAKE_SOURCE_DIR}/utils
                            ${CMAKE_SOURCE_DIR}/../../tidl_wrapper/cpp/common
                            ${CMAKE_SOURCE_DIR}/../../tidl_wrapper/cpp/onnxrt
                            ${CMAKE_SOURCE_DIR}/../../tidl_wrapper/cpp/tflitert
                            ${CMAKE_SOURCE_DIR}/../../tidl_wrapper/cpp/tvmrt
                            ${CMAKE_SOURCE_DIR}/../../tidl_wrapper/cpp/tidlrt
                            ${CMAKE_CURRENT_LIST_DIR}/../../../tools/cnpy_aarch64/include
                            )
    
        # Update link directories to include both Release and Debug builds
        link_directories(${CMAKE_SOURCE_DIR}/../../tidl_wrapper/cpp/lib/${CMAKE_BUILD_TYPE}
                        ${CMAKE_CURRENT_LIST_DIR}/../../../tools/cnpy_aarch64/lib)
    
        # Define the libraries to link against
        set(COMMON_LINK_LIBS
            ${COMMON_LINK_LIBS}
            tidl_wrapper
            )
    
        set(SYSTEM_LINK_LIBS
            ${SYSTEM_LINK_LIBS}
            yaml-cpp
            vx_tidl_rt
            png16
            dl
            cnpy
            z
            )
else()
    message(FATAL_ERROR "Compilation on HOST_CPU=${HOST_CPU} for TARGET_CPU=${TARGET_CPU} is not supported.")
endif()

if(BUILD_BASIC_EXAMPLE)
    set(BASIC_EXAMPLE_SRC basic_example/basic_example.cpp)
    build_app(basic_example ${BASIC_EXAMPLE_SRC})
endif()

if(BUILD_PREEMPTION_EXAMPLE)
    set(PREEMPTION_EXAMPLE_SRC preemption_example/preemption_example.cpp)
    build_app(preemption_example ${PREEMPTION_EXAMPLE_SRC})
endif()

if(BUILD_RETRONET_EXAMPLE)
    set(RETRONET_EXAMPLE_SRC retronet_example/retronet_example.cpp)
    build_app(retronet_example ${RETRONET_EXAMPLE_SRC})
endif()
