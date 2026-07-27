#include "las_data.hpp"
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

    const laspar::MemoryMappedFile& las_file = file_result.value();
    const auto* header = reinterpret_cast<const laspar::LasHeader*>(las_file.data());

    std::string_view signature(header->signature);
    if (signature != "LASF")
    {
        std::println(stderr, "Error: Not a valid LAS file (Missing LASF signature).");
        return 1;
    }

    std::print("Successfully mapped LAS file:\n"
               "  Signature: {}\n"
               "  Version:   {}.{}\n"
               "  Points:    {}\n"
               "  Offset:    {} bytes\n",
               signature, header->version_major, header->version_minor, header->number_of_point_records,
               header->offset_to_point_data);

    return 0;
}