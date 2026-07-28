#include "memory_mapper.hpp"
#include "scalar_processing.hpp"
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

    std::println("Processing {} points...", view.point_count);

    auto start_time = std::chrono::high_resolution_clock::now();
    laspar::BoundingBox bbox = laspar::compute_bounding_box_scalar(file_result->data(), view);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::println(
        "\nProcessed bounding box:\n"
        "  X: [{:.3}, {:.3}]\n"
        "  Y: [{:.3}, {:.3}]\n"
        "  Z: [{:.3}, {:.3}]\n"
        "Time taken: {:.3} seconds",
        bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y, bbox.min_z, bbox.max_z, elapsed.count()
    );

    std::println(
        "\nStored bounding box:\n"
        "  X: [{:.3}, {:.3}]\n"
        "  Y: [{:.3}, {:.3}]\n"
        "  Z: [{:.3}, {:.3}]\n",
        view.header->min_x, view.header->max_x, view.header->min_y, view.header->max_y, view.header->min_z,
        view.header->max_z
    );

    return 0;
}