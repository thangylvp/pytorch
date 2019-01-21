cmake_minimum_required(VERSION 3.0)
project(example-app)

set (CMAKE_PREFIX_PATH "/home/thangnv/Github/libtorch")
find_package(Torch REQUIRED)

add_executable(example-app src/example.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 11)
