cmake_minimum_required (VERSION 2.6)
project(dqn)

option(CAFFE_CPU_ONLY "Use CPU only for Caffe" OFF)
option(USE_SDL "Use SDL for ALE" ON)

# Edit these paths to point to your caffe & ALE installations
set(ALE_ROOT_DIR "~/projects/Arcade-Learning-Environment")
set(CAFFE_ROOT_DIR "~/projects/caffe")
list(APPEND CMAKE_PREFIX_PATH ${ALE_ROOT_DIR} ${CAFFE_ROOT_DIR})

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

find_package(ALE REQUIRED)
include_directories(${ALE_INCLUDE_DIRS})
target_link_libraries(dqn ${ALE_LIBRARIES})

if(USE_SDL)
  find_package(SDL REQUIRED)
  add_definitions(-D__USE_SDL)
  include_directories(${SDL_INCLUDE_DIR})
  target_link_libraries(dqn ${SDL_LIBRARY} ${SDL_MAIN_LIBRARY})
endif()

if(CPU_ONLY)
  add_definitions(-DCPU_ONLY)
endif()