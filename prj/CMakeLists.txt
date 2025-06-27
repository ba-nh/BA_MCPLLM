cmake_minimum_required(VERSION 3.10)
project(rkllm_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(CURL REQUIRED libcurl)
pkg_check_modules(JSONCPP REQUIRED jsoncpp)

set(SOURCE_FILES_1 src/llm_demo.cpp)
add_executable(llm_demo ${SOURCE_FILES_1})

set(SOURCE_FILES_2 src/multimodel_demo.cpp)
add_executable(multimodel_demo ${SOURCE_FILES_2})

set(RKLLM_API_PATH "${CMAKE_SOURCE_DIR}/${CMAKE_SYSTEM_NAME}/librkllm_api")
include_directories(${RKLLM_API_PATH}/include)
include_directories(${CURL_INCLUDE_DIRS})
include_directories(${JSONCPP_INCLUDE_DIRS})

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(RKLLM_RT_LIB ${RKLLM_API_PATH}/${CMAKE_ANDROID_ARCH_ABI}/librkllmrt.so)
    find_package(OpenMP REQUIRED)
    target_link_libraries(llm_demo  ${RKLLM_RT_LIB} log OpenMP::OpenMP_CXX ${CURL_LIBRARIES} ${JSONCPP_LIBRARIES})
    target_link_libraries(multimodel_demo  ${RKLLM_RT_LIB} log OpenMP::OpenMP_CXX ${CURL_LIBRARIES} ${JSONCPP_LIBRARIES})
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(RKLLM_RT_LIB ${RKLLM_API_PATH}/aarch64/librkllmrt.so)
    target_link_libraries(llm_demo  ${RKLLM_RT_LIB} ${CURL_LIBRARIES} ${JSONCPP_LIBRARIES})
    target_link_libraries(multimodel_demo  ${RKLLM_RT_LIB} ${CURL_LIBRARIES} ${JSONCPP_LIBRARIES})
endif()


