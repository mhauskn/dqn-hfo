cmake_minimum_required (VERSION 2.8.12)
project(dqn-hfo)

add_definitions(${Caffe_DEFINITIONS})
option(CPU_ONLY "Use CPU only for Caffe" ON)
if(CPU_ONLY)
  add_definitions(-DCPU_ONLY)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -Wno-deprecated-declarations")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

find_package(Boost 1.40 COMPONENTS filesystem system iostreams regex REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(GFLAGS REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIR})

find_package(GLOG REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})

find_package(CAFFE REQUIRED)
include_directories(${CAFFE_INCLUDE_DIRS})

find_package(HFO REQUIRED)
include_directories(${HFO_INCLUDE_DIRS})

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_executable(dqn ${SOURCES})
target_link_libraries(dqn ${Boost_LIBRARIES})
target_link_libraries(dqn ${GFLAGS_LIBRARY})
target_link_libraries(dqn ${GLOG_LIBRARIES})
target_link_libraries(dqn ${PROTOBUF_LIBRARIES})
target_link_libraries(dqn ${CAFFE_LIBRARIES})
target_link_libraries(dqn ${HFO_LIBRARIES})
set_target_properties(dqn PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

add_executable(dummy_teammate ${CMAKE_CURRENT_SOURCE_DIR}/src/hfo_policies/dummy_teammate.cpp)
set_target_properties(dummy_teammate PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
target_link_libraries(dummy_teammate ${HFO_LIBRARIES})

add_executable(dummy_goalie ${CMAKE_CURRENT_SOURCE_DIR}/src/hfo_policies/dummy_goalie.cpp)
set_target_properties(dummy_goalie PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
target_link_libraries(dummy_goalie ${HFO_LIBRARIES})

add_executable(chaser ${CMAKE_CURRENT_SOURCE_DIR}/src/hfo_policies/chaser.cpp)
set_target_properties(chaser PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
target_link_libraries(chaser ${HFO_LIBRARIES})

add_custom_command(TARGET dqn POST_BUILD
  COMMAND ln -sf ${HFO_ROOT_DIR}/bin/HFO ${CMAKE_CURRENT_SOURCE_DIR}/bin/)

add_custom_command(TARGET dqn POST_BUILD
  COMMAND ln -sf ${HFO_ROOT_DIR}/bin/teams/base/config/formations-dt ${CMAKE_CURRENT_SOURCE_DIR}/bin/)
