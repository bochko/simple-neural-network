cmake_minimum_required(VERSION 3.14)
project(SimpleNeuralNetwork_C__)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(SimpleNeuralNetwork
        SNN.cpp neuron.cpp neuron.h snn_config.h)
