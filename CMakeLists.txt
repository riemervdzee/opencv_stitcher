cmake_minimum_required(VERSION 2.8)
project(opencv_stitcher)
find_package( OpenCV REQUIRED )

SET(CMAKE_CXX_FLAGS "-std=c++11 -o2 -Wall")
SET(CMAKE_CXX_LINK_FLAGS "-std=c++11 -o2")

aux_source_directory(. SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} )
