cmake_minimum_required(VERSION 3.10)
project(VectorProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Library
add_library(VectorLib src/vector.cpp)
add_library(PerceptronLib src/perceptron.cpp)
target_include_directories(VectorLib PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(PerceptronLib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Add examples subdirectory
add_subdirectory(examples)

# Add tests
add_subdirectory(tests)
