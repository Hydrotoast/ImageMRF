cmake_minimum_required(VERSION 2.6)
project(MRF_IMAGE)

find_package(OpenCV REQUIRED)

set(SOURCE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/utility.cpp)

include_directories("include")
add_definitions(-std=c++11)

add_executable(mrf ${SOURCE})
target_link_libraries(mrf ${OpenCV_LIBS})
