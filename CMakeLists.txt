cmake_minimum_required(VERSION 3.5)

project(FaceDetector)

add_definitions(-DNOMINMAX)

# 设置 C++ 标准为 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(NCNN_DIR  ${CMAKE_SOURCE_DIR}/third_party/ncnn)
include_directories(${NCNN_DIR}/include)
include_directories(${NCNN_DIR}/include/ncnn)
link_directories("${NCNN_DIR}/lib")
set(NCNN_LIBS ncnn)

set(OPENCV_DIR  ${CMAKE_SOURCE_DIR}/third_party/opencv4)
include_directories(${OPENCV_DIR}/include)
link_directories("${OPENCV_DIR}/lib")
set(OpenCV_LIBS opencv_world4100)

set(ONNXRUNTIME_DIR  ${CMAKE_SOURCE_DIR}/third_party/onnxruntime-win-x64-1.14.1)
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories("${ONNXRUNTIME_DIR}/lib")
set(ONNXRUNTIME_LIBS onnxruntime onnxruntime_providers_shared)

# compile own file
file(GLOB SRC_LIST 
    ${CMAKE_SOURCE_DIR}/src/*.cpp
    ${CMAKE_SOURCE_DIR}/include/*.h

)

add_executable(demo ${SRC_LIST})

target_link_libraries(demo ${NCNN_LIBS} ${OpenCV_LIBS} ${ONNXRUNTIME_LIBS})

