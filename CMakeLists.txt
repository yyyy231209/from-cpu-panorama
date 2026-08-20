cmake_minimum_required(VERSION 3.10)
project(rga_gpu_cpu_stitch LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optional local 3rdparty dir; if empty, system libs (/usr/lib, /usr/local/lib) are used
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
# Vendored yolo helper SDK (rknn_api, yolov8 wrapper, postprocess, image utils)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_compile_options(-Wall -Wextra -Wno-unused-parameter -fopenmp)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

find_library(RGA_LIB
    NAMES rga
    PATHS
        ${THIRD_PARTY_DIR}/librga/Linux/aarch64
        /usr/lib /usr/lib/aarch64-linux-gnu /usr/local/lib
)

find_library(RKNNRT_LIB
    NAMES rknnrt
    PATHS
        /usr/lib /usr/lib/aarch64-linux-gnu /usr/local/lib
)

if(NOT RGA_LIB)
    message(FATAL_ERROR "librga not found")
endif()
if(NOT RKNNRT_LIB)
    message(FATAL_ERROR "librknnrt not found")
endif()

find_library(TURBOJPEG_LIB
    NAMES turbojpeg
    PATHS
        /usr/lib /usr/lib/aarch64-linux-gnu /usr/local/lib
)

if(NOT TURBOJPEG_LIB)
    message(FATAL_ERROR "libturbojpeg not found (install libturbojpeg-dev)")
endif()

add_executable(rga_gpu_cpu_stitch
    main.cpp
    cpu_H_fuc.cpp
    cpu_queue.cpp
    cpu_warp.cpp
    npu_queue.cpp
    worker_cpu.cpp
    worker_npu.cpp
    postprocess_crack.cpp
    mjpeg_streamer.cpp
    ${INCLUDE_DIR}/include_yolo/dma_alloc.cpp
    ${INCLUDE_DIR}/yolo/yolov8.cc
    ${INCLUDE_DIR}/utils/preprocess.cc
    ${INCLUDE_DIR}/include_yolo/postprocess.cc
    ${INCLUDE_DIR}/utils/file_utils.c
    ${INCLUDE_DIR}/utils/image_utils.c
    ${INCLUDE_DIR}/utils/image_drawing.c
)

target_include_directories(rga_gpu_cpu_stitch PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${INCLUDE_DIR}
    ${INCLUDE_DIR}/yolo
    ${INCLUDE_DIR}/include_yolo
    ${INCLUDE_DIR}/utils
    ${OpenCV_INCLUDE_DIRS}
    ${THIRD_PARTY_DIR}/librga/include
)

target_link_libraries(rga_gpu_cpu_stitch PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
    OpenMP::OpenMP_CXX
    ${RGA_LIB}
    ${RKNNRT_LIB}
    ${TURBOJPEG_LIB}
    dl
    m
)
