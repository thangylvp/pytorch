cmake_minimum_required(VERSION 3.0)
project(smoke_libtorch)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set (CMAKE_PREFIX_PATH "/home/thangnv/Github/libtorch")

find_package(Torch REQUIRED)
find_package(OpenCV 4.0 REQUIRED)

add_executable(example-app src/example.cpp)
target_link_libraries(example-app ${TORCH_LIBRARIES} ${OpenCV_LIBS})

