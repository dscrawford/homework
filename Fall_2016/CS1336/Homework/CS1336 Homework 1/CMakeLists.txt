cmake_minimum_required(VERSION 3.6)
project(CS1336_Homework_1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES CS1336_Homework1.cpp)
add_executable(CS1336_Homework_1 ${SOURCE_FILES})