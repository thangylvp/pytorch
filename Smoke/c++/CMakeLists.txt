cmake_minimum_required(VERSION 3.1)
project(smoke_libtorch)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set (CMAKE_PREFIX_PATH "/home/thangnv/Github/libtorch")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(example src/example.cpp)
target_link_libraries(example "${TORCH_LIBRARIES}" ${OpenCV_LIBS} )

add_executable(TestOpenCV src/testOpenCV.cpp)
target_link_libraries(TestOpenCV ${OpenCV_LIBS} )

