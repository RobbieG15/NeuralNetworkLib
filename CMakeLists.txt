cmake_minimum_required(VERSION 3.10)

project(VectorProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Library: use source file
add_library(VectorLib src/vector.cpp)

# Tell CMake where to find headers
target_include_directories(VectorLib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Add tests
add_subdirectory(tests)