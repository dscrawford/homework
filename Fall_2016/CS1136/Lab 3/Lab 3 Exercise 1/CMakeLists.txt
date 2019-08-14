cmake_minimum_required(VERSION 3.6)
project(Lab_3_Exercise_1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES Lab3_Exercise1.cpp)
add_executable(Lab_3_Exercise_1 ${SOURCE_FILES})