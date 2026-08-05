#include "cli.hpp"
#include <print>

int main(int argc, const char** argv)
{
    if constexpr (eve::current_api == eve::neon)
        std::println("Using NEON on ARM64");
    else if constexpr (eve::current_api == eve::avx2)
        std::println("Using AVX2 on x86-64");
    else if constexpr (eve::current_api == eve::avx512)
        std::println("Using AVX512 on x86-64");
    else
        std::println("WARNING: Using unknown SIMD backend");

    laspar::launch_cli(argc, argv);

    return 0;
}