#include "header_view.hpp"
#include "memory_mapper.hpp"
#include <print>
#include <string_view>

int main()
{
    auto file_result = laspar::MemoryMappedFile::open("test.las");

    if (!file_result)
    {
        std::println(stderr, "Fatal error: {}", file_result.error());
        return 1;
    }

    auto header_result = laspar::validate_las_header(file_result->data(), file_result->size());
    if (!header_result)
    {
        std::println(stderr, "Invalid LAS file: {}", header_result.error());
        return 1;
    }

    const auto& view = header_result.value();
    std::print(
        "Successfully mapped LAS file:\n"
        "  Points: {}\n"
        "  Offset: {} bytes\n",
        view.point_count, view.point_data_offset
    );

    return 0;
}