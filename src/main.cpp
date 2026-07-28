#include "benchmark.hpp"
#include "memory_mapper.hpp"
#include <print>

using namespace laspar;

int main()
{
    auto file_result = MemoryMappedFile::open("test.las");

    if (!file_result)
    {
        std::println(stderr, "Fatal error: {}", file_result.error());
        return 1;
    }

    auto header_result = validate_las_header(file_result->data(), file_result->size());
    if (!header_result)
    {
        std::println(stderr, "Invalid LAS file: {}", header_result.error());
        return 1;
    }

    const auto& view = header_result.value();

    run_benchmark(view, file_result->data());

    return 0;
}