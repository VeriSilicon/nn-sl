# NOTICE: only valid for -DSL_DIST_BUILD=OFF, for build from distributed package
# VSI already packed slang library properly inside of android.sl

if(NOT SLANG_TARGET_PID)
message(FATAL_ERROR "Slang require concrete PID name")
endif()

ExternalProject_Add(Slang
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/deps/slang
    GIT_REPOSITORY git@gitlab-cn.verisilicon.com:ipd_vip_vosp/slang.git
    GIT_TAG master
    CMAKE_ARGS
        "-DHW_SPEC=${SLANG_TARGET_PID}"
        "-DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/deps/"
)