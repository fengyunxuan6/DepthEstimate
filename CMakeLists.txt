cmake_minimum_required(VERSION 3.5)

get_filename_component(ProjectId ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${ProjectId})
#================== Add BY  ZJL =====================#
#获取当前所在文件
get_filename_component(CMAKE_MODULE_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
#获取上级目录
get_filename_component(MVS_PATH "${CMAKE_MODULE_DIR}" DIRECTORY)
get_filename_component(LF_WORK_PATH "${MVS_PATH}" DIRECTORY)
#message("-----LF_WORK_PATH:  "  ${LF_WORK_PATH})
#即LF_WORK_PATH = "/home/zjl/work/Code/LF"
#添加头文件搜索路径
include_directories(${LF_WORK_PATH}/ThirdPartyLib/boost_1_71_0/include)
#对不同构建设置
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set(libpleno_DIR ${LF_WORK_PATH}/Common/LibPleno/lib/Debug/cmake/pleno)
    link_directories(${LF_WORK_PATH}/ThirdPartyLib/opencv-4.2.0/Debug/lib;
            ${LF_WORK_PATH}/ThirdPartyLib/yaml-cpp/lib;
            ${LF_WORK_PATH}/ThirdPartyLib/
    )
    #设置 boost引用路径
    set(Boost_INCLUDE_DIR ${LF_WORK_PATH}/ThirdPartyLib/boost_1_71_0/include)
    set(Boost_LIBRARY_DIR ${LF_WORK_PATH}/ThirdPartyLib/boost_1_71_0/debug/lib)
elseif (${CMAKE_BUILD_TYPE} STREQUAL "Release")
    set(libpleno_DIR ${LF_WORK_PATH}/Common/LibPleno/lib/Release/cmake/pleno)
    link_directories(${LF_WORK_PATH}/ThirdPartyLib/opencv-4.2.0/Release/lib;
            ${LF_WORK_PATH}/ThirdPartyLib/yaml-cpp/lib)
    #设置 boost引用路径
    set(Boost_INCLUDE_DIR ${LF_WORK_PATH}/ThirdPartyLib/boost_1_71_0/include)
    set(Boost_LIBRARY_DIR ${LF_WORK_PATH}/ThirdPartyLib/boost_1_71_0/release/lib)
elseif (${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
    set(libpleno_DIR ${LF_WORK_PATH}/Common/LibPleno/lib/RelWithDebInfo/cmake/pleno)
    link_directories(${LF_WORK_PATH}/ThirdPartyLib/opencv-4.2.0/Release/lib;
            ${LF_WORK_PATH}/ThirdPartyLib/yaml-cpp/lib)
    #设置 boost引用路径
    set(Boost_INCLUDE_DIR ${LF_WORK_PATH}/ThirdPartyLib/boost_1_71_0/include)
    set(Boost_LIBRARY_DIR ${LF_WORK_PATH}/ThirdPartyLib/boost_1_71_0/release/lib)
endif ()
##################################################################################3

message("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
message("BLADE: Blur Aware Depth Estimation with a Multi-focus Plenoptic camera")
message("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

add_subdirectory(src/depth)
add_subdirectory(src/distances)
add_subdirectory(src/evaluate)
add_subdirectory(src/export)
add_subdirectory(src/lidarcamera)
add_subdirectory(src/scaling)


