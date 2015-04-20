cmake_minimum_required (VERSION 2.6)
project(dqn)

option(CAFFE_CPU_ONLY "Use CPU only for Caffe" OFF)

add_executable(dqn dqn_main.cpp dqn.cpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -march=native -Wno-deprecated-declarations")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

find_package(Boost 1.40 COMPONENTS filesystem system REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(dqn ${Boost_LIBRARIES})

find_package(GFLAGS REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIR})
target_link_libraries(dqn ${GFLAGS_LIBRARY})

find_package(GLOG REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})
target_link_libraries(dqn ${GLOG_LIBRARIES})

include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})
target_link_libraries(dqn ${PROTOBUF_LIBRARIES})

find_package(CAFFE REQUIRED)
include_directories(${CAFFE_INCLUDE_DIRS})
target_link_libraries(dqn ${CAFFE_LIBRARIES})

find_package(HFO REQUIRED)
include_directories(${HFO_INCLUDE_DIRS})
target_link_libraries(dqn ${HFO_LIBRARIES})

if(CPU_ONLY)
  add_definitions(-DCPU_ONLY)
endif()