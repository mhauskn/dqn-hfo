# - Try to find HFO
#
# The following variables are optionally searched for defaults
#  HFO_ROOT_DIR:            Base directory where all GLOG components are found
#
# The following are set after configuration is done:
#  HFO_FOUND
#  HFO_INCLUDE_DIRS
#  HFO_LIBRARIES

include(FindPackageHandleStandardArgs)

set(HFO_ROOT_DIR "" CACHE PATH "Folder containing HFO")

find_path(HFO_INCLUDE_DIR HFO.hpp
  PATHS ${HFO_ROOT_DIR}
  PATH_SUFFIXES
  src)

find_library(HFO_LIBRARY hfo
  PATHS ${HFO_ROOT_DIR}
  PATH_SUFFIXES
  lib)

find_package_handle_standard_args(HFO DEFAULT_MSG
  HFO_INCLUDE_DIR HFO_LIBRARY)

if(HFO_FOUND)
  set(HFO_INCLUDE_DIRS ${HFO_INCLUDE_DIR})
  set(HFO_LIBRARIES ${HFO_LIBRARY})
endif()