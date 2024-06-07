#    Copyright (c) 2021 Vivante Corporation
#
#    Permission is hereby granted, free of charge, to any person obtaining a
#    copy of this software and associated documentation files (the "Software"),
#    to deal in the Software without restriction, including without limitation
#    the rights to use, copy, modify, merge, publish, distribute, sublicense,
#    and/or sell copies of the Software, and to permit persons to whom the
#    Software is furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in
#    all copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#    DEALINGS IN THE SOFTWARE
#
set( repo_addr https://github.com/VeriSilicon/TIM-VX.git)
if(NOT PUBLIC_TIM_VX)
    set(repo_addr git@gitlab-cn.verisilicon.com:npu_sw/verisilicon/tim-vx.git)
endif()

if(EXT_TIM_VX)
    set(repo_addr ${EXT_TIM_VX})
endif()

if(NOT TIM_VX_TAG)
set(TIM_VX_TAG "main")
endif()

message(STATUS "use TIM_VX from ${repo_addr}")
message(STATUS "TIM_VX version: ${TIM_VX_TAG}")
message(STATUS "TIM_VX build with external sdk -> ${EXT_VIV_SDK}")

if(USE_GRPC)
    ExternalProject_Add(tim-vx
        SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/deps/tim-vx
        GIT_REPOSITORY ${repo_addr}
        GIT_TAG ${TIM_VX_TAG}
        CMAKE_ARGS
            "-DTIM_VX_ENABLE_PLATFORM=ON"
            "-DEXTERNAL_VIV_SDK=${EXT_VIV_SDK}"
            "-DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/tim-vx-install/${LIBABI}"
            "-DCMAKE_TOOLCHAIN_FILE:FILEPATH=${CMAKE_TOOLCHAIN_FILE}"
            "-DANDROID_ABI=${ANDROID_ABI}"
            "-DTIM_VX_DBG_ENABLE_TENSOR_HNDL=${ENABLE_TENSOR_HANDLE}"
            "-DTIM_VX_ENABLE_GRPC=ON"
            "-DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
            "-DProtobuf_DIR=${Protobuf_DIR}"
            "-DgRPC_DIR=${gRPC_DIR}"
            "-Dabsl_DIR=${absl_DIR}"
            "-DANDROID_PLATFORM=${ANDROID_PLATFORM}"
    )
else()
    list(APPEND ARGS "-DTIM_VX_ENABLE_PLATFORM=ON")
    list(APPEND ARGS "-DEXTERNAL_VIV_SDK=${CMAKE_SOURCE_DIR}/prebuilt/android_arm64")
    list(APPEND ARGS "-DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/tim-vx-install/")
    list(APPEND ARGS "-DCMAKE_TOOLCHAIN_FILE:FILEPATH=${CMAKE_TOOLCHAIN_FILE}")
    list(APPEND ARGS "-DANDROID_ABI=${ANDROID_ABI}")

    if(NOT ANDROID_PLATFORM)
        list(APPEND ARGS "-DANDROID_PLATFORM=${ANDROID_PLATFORM}")
    else()
        list(APPEND ARGS "-DANDROID_PLATFORM=android-34")
    endif()

    message(STATUS "TIM-VX cmake args = ${ARGS}")

    ExternalProject_Add(tim-vx
        SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/deps/tim-vx
        GIT_REPOSITORY ${repo_addr}
        GIT_TAG ${TIM_VX_TAG}
        CMAKE_ARGS
            "-DTIM_VX_ENABLE_PLATFORM=ON"
            "-DEXTERNAL_VIV_SDK=${EXT_VIV_SDK}"
            "-DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/tim-vx-install/${LIBABI}"
            "-DCMAKE_TOOLCHAIN_FILE:FILEPATH=${CMAKE_TOOLCHAIN_FILE}"
            "-DANDROID_ABI=${ANDROID_ABI}"
            "-DTIM_VX_DBG_ENABLE_TENSOR_HNDL=${ENABLE_TENSOR_HANDLE}"
            "-DANDROID_PLATFORM=${ANDROID_PLATFORM}"
            "-DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}"
    )
endif()
