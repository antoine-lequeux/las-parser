cmake_minimum_required(VERSION 3.20)
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

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    # Disable exceptions and RTTI globally so dependencies build without them
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>"
        "$<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>"
    )

    option(USE_AVX512 "Compile with AVX-512 support" OFF)

    # Architecture / SIMD baseline
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
        if(USE_AVX512)
            # Includes all modern x86-64 computers supporting AVX-512
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=x86-64-v4>)
            message(STATUS "> x86_64 architecture detected (AVX-512 / x86-64-v4)")
        else()
            # Includes all modern x86-64 computers
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=x86-64-v3>)
            message(STATUS "> x86_64 architecture detected (AVX2 / x86-64-v3)")
        endif()
    elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        # Includes all Silicon Apple chips from M1 to M5)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mcpu=apple-m1>)
        message(STATUS "> Apple Silicon architecture detected (ARM64 / NEON)")
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    # Disable exceptions and RTTI globally
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CXX>:/EHs-c->"
        "$<$<COMPILE_LANGUAGE:CXX>:/GR->"
        "$<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>"
        # Force UTF-8 for MSVC
        "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>"
    )

    option(USE_AVX512 "Compile with AVX-512 support" OFF)

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
        if(USE_AVX512)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
            message(STATUS "> x86_64 architecture detected (AVX-512 / x86-64-v4)")
        else()
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
            message(STATUS "> x86_64 architecture detected (AVX2 / x86-64-v3)")
        endif()
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
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
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
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")

    # Warnings
    target_compile_options(las_parser PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wconversion
        -Wsign-conversion
        -Wnon-virtual-dtor
        -Wno-missing-field-initializers
        -Wno-missing-braces
        -Wold-style-cast
        -Wdouble-promotion
        -Wcast-align
        -Wimplicit-fallthrough
        -Wunreachable-code
        -Wnull-dereference
        -Wundef
        -Wunused-macros
        -Wextra-semi
        -Wzero-as-null-pointer-constant
        -Wsuggest-override
    )

endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(las_parser PRIVATE
        -Wheader-hygiene
    )

endif()

# Optimization flags
target_compile_options(las_parser PRIVATE
    "$<$<CONFIG:Debug>:-O0>"
    "$<$<CONFIG:Debug>:-g3>"
)

install(TARGETS las_parser DESTINATION bin)

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
    "$<$<CONFIG:Debug>:-O0>"
    "$<$<CONFIG:Debug>:-g3>"
)

add_test(NAME las_test COMMAND las_test "${CMAKE_SOURCE_DIR}/test.las")