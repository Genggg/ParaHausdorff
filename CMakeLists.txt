cmake_minimum_required(VERSION 2.8)
project(ParaHausdorff)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )


include_directories(src)
add_subdirectory(src)

add_executable( main main.cpp )
target_link_libraries(main ParaHausdorffLibrary ${OpenCV_LIBS} )

