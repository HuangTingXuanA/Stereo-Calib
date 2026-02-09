cmake_minimum_required(VERSION 3.16)
project(stereo_calibrator LANGUAGES CXX)

# C++17标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 查找ONNX Runtime（必需）
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_ROOT}/include
    PATH_SUFFIXES onnxruntime
)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
    HINTS ${ONNXRUNTIME_ROOT}/lib
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime not found. Please install ONNX Runtime.")
endif()

message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIBRARY}")

# 头文件目录
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${ONNXRUNTIME_INCLUDE_DIR})

# 源文件
set(SOURCES
    src/main.cpp
    src/StereoCalibrator.cpp
    src/BoardConfig.cpp
    src/dsf.cpp
    src/ellipse_detector.cpp
    src/YoloSegmentor.cpp
)

# 头文件
set(HEADERS
    include/CalibrationTypes.h
    include/StereoCalibrator.h
    include/BoardConfig.h
    include/dsf.hpp
    include/ellipse_detector.hpp
    include/YoloSegmentor.h
)

# 可执行目标
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# 链接库
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARY})

# 编译选项
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -g -O0)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -O2)
endif()

# 打印配置信息
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_LIBRARY}")

# 注释掉visualize_all_matches（使用了旧代码）
# add_executable(visualize_all_matches
#     visualize_all_matches.cpp
#     src/BoardConfig.cpp
#     src/dsf.cpp
#     src/ellipse_detector.cpp
# )
# target_link_libraries(visualize_all_matches ${OpenCV_LIBS})

