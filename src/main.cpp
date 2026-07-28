#include "memory_mapper.hpp"
#include "scalar_processing.hpp"
#include <chrono>
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

    std::println("Starting scalar bounding box calculation...");

    auto start_time = std::chrono::high_resolution_clock::now();
    BoundingBox bbox = compute_bounding_box_scalar(file_result->data(), view);
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end_time - start_time;
    double seconds = elapsed.count();

    u64 total_bytes_processed = view.point_count * view.point_record_length;

    double gigabytes = static_cast<double>(total_bytes_processed) / (1024.0 * 1024.0 * 1024.0);
    double throughput_gb_s = gigabytes / seconds;
    double million_points_s = (static_cast<double>(view.point_count) / 1'000'000.0) / seconds;

    std::println(
        "\nProcessed bounding box:\n"
        "  X: [{:.3}, {:.3}]\n"
        "  Y: [{:.3}, {:.3}]\n"
        "  Z: [{:.3}, {:.3}]\n",
        bbox.min_x, bbox.max_x, bbox.min_y, bbox.max_y, bbox.min_z, bbox.max_z
    );

    std::println(
        "\nPerformance report (scalar):\n"
        "  Points   : {}\n"
        "  Time     : {:.4} seconds\n"
        "  Speed    : {:.3} Mp/s\n"
        "  Bandwidth: {:.3} GB/s\n",
        view.point_count, seconds, million_points_s, throughput_gb_s
    );

    return 0;
}