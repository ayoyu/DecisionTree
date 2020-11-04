cmake_minimum_required(VERSION 3.0.0)
project(Algo VERSION 0.1.0)

include(CTest)
enable_testing()
set(source_dir "${PROJECT_SOURCE_DIR}/src/")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -std=c++14")
file(GLOB source_files "${source_dir}/tree/*.cpp")
add_library(Algo SHARED ${source_files})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
