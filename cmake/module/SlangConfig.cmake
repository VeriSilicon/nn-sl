# NOTICE: only valid for -DSL_DIST_BUILD=OFF, for build from distributed package
# VSI already packed slang library properly inside of android.sl
set( repo_addr git@gitlab-cn.verisilicon.com:ipd_vip_vosp/slang.git)
if(INTERNAL_BUILD)
    set(repo_addr https://gitlab-cn.verisilicon.com/ipd_vip_vosp_ro/slang.git)
endif()

if(NOT SLANG_TARGET_PID)
message(FATAL_ERROR "Slang require concrete PID name")
endif()

ExternalProject_Add(Slang
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/deps/slang
    GIT_REPOSITORY ${repo_addr}
    GIT_TAG master
    CMAKE_ARGS
        "-DHW_SPEC=${SLANG_TARGET_PID}"
        "-DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_SOURCE_DIR}/deps/"
)