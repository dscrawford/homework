cmake_minimum_required(VERSION 3.6)
project(Lab_5b_Exercise_3_Project)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES Lab5b_Exercise3.cpp)
add_executable(Lab_5b_Exercise_3_Project ${SOURCE_FILES})