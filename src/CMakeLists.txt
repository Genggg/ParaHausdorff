cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(ParaHausdorffLibrary)



# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -G -g ")

cuda_add_library(ParaHausdorffLibrary improc.h improc.cu gpukernels.cu gpukernels.h)

set_target_properties(ParaHausdorffLibrary PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(ParaHausdorffLibrary PROPERTIES CUDA_SEPARABLE_COMPILATION ON)