cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(ParaHausdorff) 

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" )

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -G -g
    )


find_package(OpenMP REQUIRED)

find_package(CUDA REQUIRED)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(src)
add_subdirectory(src)

cuda_add_executable( main main.cu )
target_link_libraries(main ParaHausdorffLibrary ${OpenCV_LIBS})
