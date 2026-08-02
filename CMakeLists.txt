cmake_minimum_required(VERSION 4.2)

project(LAS_Parser VERSION 0.1.0 LANGUAGES CXX)

enable_testing()

# ---------------------------------------------------------
# Global project configuration
# ---------------------------------------------------------

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Default build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Use ccache if available for build speed
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER   ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

# Global LTO/IPO for Release
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_ok)
if(ipo_ok)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

# ---------------------------------------------------------
# Global compiler flags
# ---------------------------------------------------------

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Disable exceptions and RTTI globally so dependencies build without them
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>"
        "$<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>"
    )

    # Architecture / SIMD baseline
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
        # Includes all modern windows/linux computers 
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=x86-64-v3>)
    elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        # Includes all Silicon Apple chips from M1 to M5)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mcpu=apple-m1>)
    endif()
endif()

# ---------------------------------------------------------
# FetchContent setup
# ---------------------------------------------------------
include(FetchContent)

# Disable test/benchmark builds for EVE
set(EVE_BUILD_TEST           OFF CACHE BOOL "" FORCE)
set(EVE_BUILD_BENCHMARKS     OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    lyra
    GIT_REPOSITORY https://github.com/bfgroup/Lyra.git
    GIT_TAG        1.8.0
    SYSTEM
)

FetchContent_Declare(
    eve
    GIT_REPOSITORY https://github.com/jfalcou/eve.git
    GIT_TAG        v2023.02.15
    SYSTEM
)

# Silence all warnings during third-party library compilation.
set(BACKUP_C_FLAGS "${CMAKE_C_FLAGS}")
set(BACKUP_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
endif()

FetchContent_MakeAvailable(lyra eve)

# Restore original flags for the actual project
set(CMAKE_C_FLAGS "${BACKUP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${BACKUP_CXX_FLAGS}")

# ---------------------------------------------------------
# Executable target
# ---------------------------------------------------------
add_executable(las_parser src/main.cpp)

# Shared cross-platform links
target_link_libraries(las_parser PRIVATE
    bfg::lyra
    eve::eve
)

# ---------------------------------------------------------
# Executable-only optimizations & strict warnings
# ---------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")

    # Warnings
    target_compile_options(las_parser PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wnon-virtual-dtor
    )

endif()

# Optimization flags
target_compile_options(las_parser PRIVATE
    "$<$<CONFIG:Release>:-O3>"
    "$<$<CONFIG:Debug>:-O0>"
    "$<$<CONFIG:Debug>:-g3>"
)

# ---------------------------------------------------------
# Test executable
# ---------------------------------------------------------
add_executable(las_test
    src/test_main.cpp
)

target_link_libraries(las_test PRIVATE
    eve::eve
)

target_compile_options(las_test PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wnon-virtual-dtor
)

target_compile_options(las_test PRIVATE
    "$<$<CONFIG:Release>:-O3>"
    "$<$<CONFIG:Debug>:-O0>"
    "$<$<CONFIG:Debug>:-g3>"
)

add_test(NAME las_test COMMAND las_test "${CMAKE_SOURCE_DIR}/test.las")