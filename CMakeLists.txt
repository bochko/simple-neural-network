cmake_minimum_required(VERSION 3.14)
project(SimpleNeuralNetwork)
set(CMAKE_CXX_STANDARD 14)

### TESTING
# GoogleTest specific configuration
ADD_SUBDIRECTORY(./test/googletest)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
enable_testing()

# Project-specific unit tests
ADD_SUBDIRECTORY(./test/units)
# Project-specific SimpleNeuralNetwork binary
ADD_SUBDIRECTORY(./src)
