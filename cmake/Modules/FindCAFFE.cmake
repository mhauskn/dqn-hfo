# - Try to find CAFFE
#
# The following variables are optionally searched for defaults
#  CAFFE_ROOT_DIR:            Base directory where all GLOG components are found
#
# The following are set after configuration is done:
#  CAFFE_FOUND
#  CAFFE_INCLUDE_DIRS
#  CAFFE_LIBRARIES

include(FindPackageHandleStandardArgs)

set(CAFFE_ROOT_DIR "" CACHE PATH "Folder containing CAFFE")

find_path(CAFFE_INCLUDE_DIR caffe/caffe.hpp
  PATHS ${CAFFE_ROOT_DIR}
  PATH_SUFFIXES
  include)

find_library(CAFFE_LIBRARY caffe
  PATHS ${CAFFE_ROOT_DIR}
  PATH_SUFFIXES
  build/lib)

find_package_handle_standard_args(CAFFE DEFAULT_MSG
  CAFFE_INCLUDE_DIR CAFFE_LIBRARY)

if(CAFFE_FOUND)
  set(CAFFE_INCLUDE_DIRS ${CAFFE_INCLUDE_DIR})
  set(CAFFE_LIBRARIES ${CAFFE_LIBRARY})
endif()