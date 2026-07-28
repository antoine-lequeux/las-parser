#pragma once

#include <chrono>
#include <print>

#include "avx2_processing.hpp"
#include "header_view.hpp"
#include "scalar_processing.hpp"

namespace laspar
{

template <typename Func>
inline std::pair<double, BoundingBox>
benchmark(Func&& compute_func, const char* name, u64 point_count, u32 stride) noexcept
{
    auto start = std::chrono::high_resolution_clock::now();
    BoundingBox bbox = compute_func();
    auto end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();

    u64 total_bytes = point_count * stride;
    double gb_s = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / seconds;
    double mps = (point_count / 1'000'000.0) / seconds;

    std::println("{:<10}| {:<8.2f} Mp/s | {:<6.2f} Gb/s | Time: {:.4f}s", name, mps, gb_s, seconds);

    return {seconds, bbox};
}

inline void run_benchmark(const HeaderView& view, const u8* data) noexcept
{
    std::println("Benchmarking {} points...", view.point_count);
    std::println("-------------------------------------------------------");

    auto [time_scalar, bbox_scalar] = benchmark(
        [&]() { return compute_bounding_box_scalar(data, view); }, "Scalar", view.point_count, view.point_record_length
    );

    auto [time_avx2, bbox_avx2] = benchmark(
        [&]() { return compute_bounding_box_avx2(data, view); }, "AVX2", view.point_count, view.point_record_length
    );

    std::println("-------------------------------------------------------");
    std::println("AVX2 is {:.2f}x faster then scalar.\n", (time_scalar / time_avx2));

    std::println(
        "Processed bounding box (scalar):\n"
        "  X: [{:.3}, {:.3}]\n"
        "  Y: [{:.3}, {:.3}]\n"
        "  Z: [{:.3}, {:.3}]\n",
        bbox_scalar.min_x, bbox_scalar.max_x, bbox_scalar.min_y, bbox_scalar.max_y, bbox_scalar.min_z, bbox_scalar.max_z
    );

    std::println(
        "Processed bounding box (AVX2):\n"
        "  X: [{:.3}, {:.3}]\n"
        "  Y: [{:.3}, {:.3}]\n"
        "  Z: [{:.3}, {:.3}]\n",
        bbox_avx2.min_x, bbox_avx2.max_x, bbox_avx2.min_y, bbox_avx2.max_y, bbox_avx2.min_z, bbox_avx2.max_z
    );
}
} // namespace laspar