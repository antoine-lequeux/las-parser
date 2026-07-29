cmake_minimum_required(VERSION 3.25)
project(LAS_Parser VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
FetchContent_Declare(
    lyra
    GIT_REPOSITORY https://github.com/bfgroup/Lyra.git
    GIT_TAG        1.8.0
)
FetchContent_MakeAvailable(lyra)

add_executable(las_parser src/main.cpp)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(las_parser PRIVATE
        -Wall -Wextra -Wpedantic
        -fno-exceptions
        -march=native
        $<$<CONFIG:Release>:-O3 -ffast-math -flto>
        $<$<CONFIG:Debug>:-O0 -g3>
    )
endif()

target_link_libraries(las_parser PRIVATE bfg::lyra)