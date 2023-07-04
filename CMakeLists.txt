cmake_minimum_required(VERSION 3.20)
set(project_name "VSI.AndroidSL")
project(${project_name})

option(SL_DIST_BUILD "Build from distributed package"   ON)
option(PUBLIC_TIM_VX "Download TIM_VX from github"      ON)
option(RUN_NBG       "Turn on NBG feature"              ON)
option(USE_GRPC      "Use gRPC support, work with NBG"  OFF)

if(${RUN_NBG})
  add_definitions(-DRUN_NBG)
endif()

if(${USE_GRPC})
  add_definitions(-DUSE_GRPC)
endif()

if(${CMAKE_BUILD_TYPE})
  message(STATUS "Setting build type to Release, for debug builds use"
    "'-DCMAKE_BUILD_TYPE=Debug'.")
  set(CMAKE_BUILD_TYPE "Release")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/module")

if(NOT SL_DIST_BUILD)
    include(VersionFromGit)
endif()
include(ExternalProject)

set(CMAKE_CXX_STANDARD 17)

include(TimVxConfig)

if(NOT SL_DIST_BUILD)
include(SlangConfig)
endif()

##############################################################################
# Current project source
##############################################################################
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)
set(SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/VsiNeuralNetworksSupportLibraryimpl.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Model.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Execution.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/MapOperation.cpp
)
set(DEPENDENCIES ${CMAKE_CURRENT_BINARY_DIR}/tim-vx-install/lib/libtim-vx.so)
add_library(VsiSupportLibrary SHARED ${SRC})
target_link_libraries(VsiSupportLibrary ${DEPENDENCIES})
target_include_directories(VsiSupportLibrary
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/deps/include
        ${CMAKE_CURRENT_BINARY_DIR}/tim-vx-install/include
    )

##############################################################################
# package setup
##############################################################################
if(NOT SL_DIST_BUILD)
version_from_git(
	LOG ON
	TIMESTAMP "%Y%d%H%M%S"
)

set(CPACK_PACKAGE_VERSION_MAJOR "${VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${VERSION_PATCH}")
set(CPACK_SOURCE_TBZ2 "OFF")
set(CPACK_SOURCE_TXZ  "OFF")
set(CPACK_SOURCE_TZ   "OFF")
set(CPACK_SOURCE_PACKAGE_FILE_NAME ${project_name}-${GIT_TAG}-${GIT_HASH})
include(CPack)

endif()