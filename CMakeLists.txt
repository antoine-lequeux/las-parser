cmake_minimum_required(VERSION 3.25)
project(LAS_Parser VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
FetchContent_Declare(
    CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.6.2
)
set(CLI11_BUILD_TESTS OFF CACHE INTERNAL "")
set(CLI11_BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(CLI11_BUILD_DOCS OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(CLI11)

add_executable(las_parser src/main.cpp)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(las_parser PRIVATE
        -Wall -Wextra -Wpedantic
        -march=native
        $<$<CONFIG:Release>:-O3 -ffast-math -flto>
        $<$<CONFIG:Debug>:-O0 -g3>
    )
endif()

target_link_libraries(las_parser PRIVATE CLI11::CLI11)