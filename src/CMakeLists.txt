cmake_minimum_required(VERSION 2.8)

project(ParaHausdorffLibrary)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_library(ParaHausdorffLibrary improc.h improc.cpp)

set_target_properties(ParaHausdorffLibrary PROPERTIES LINKER_LANGUAGE CXX)