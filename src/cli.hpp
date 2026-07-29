#include <array>
#include <chrono>
#include <lyra/lyra.hpp>
#include <print>
#include <sstream>
#include <vector>

#include "avx2_processing.hpp"
#include "format_dispatcher.hpp"
#include "header_view.hpp"
#include "memory_mapper.hpp"

namespace laspar
{

inline int launch_cli(int argc, const char** argv)
{
    std::string input_file;
    std::vector<int> keep_classes;
    bool show_help = false;
    bool do_bbox = false;
    bool do_count = false;

    auto cli = lyra::help(show_help) | lyra::arg(input_file, "input_file")("The LAS file to process").required() |
               lyra::opt(keep_classes, "class")["-k"]["--keep-class"]("Filter by classification (can be chained)") |
               lyra::opt(do_bbox)["-b"]["--bbox"]("Compute bounding box of filtered points") |
               lyra::opt(do_count)["-c"]["--count"]("Count filtered points by class");

    auto cli_result = cli.parse({argc, argv});
    if (!cli_result)
    {
        std::println(stderr, "CLI ERROR: {}", cli_result.message());
        return 1;
    }
    if (show_help)
    {
        std::ostringstream oss;
        oss << cli;
        std::println("{}", oss.str());
        return 0;
    }

    std::array<u8, 256> filter_mask = {0};
    std::array<double, 256> blend_mask = {0.0};
    const double lane_pass = std::bit_cast<double>(~u64 {0});

    const bool has_filter = !keep_classes.empty();
    if (has_filter)
    {
        for (i32 c : keep_classes)
        {
            if (c >= 0 && c < 256)
            {
                filter_mask[c] = 1;
                blend_mask[c] = lane_pass;
            }
        }
    }
    else
    {
        filter_mask.fill(1);
        blend_mask.fill(lane_pass);
    }

    auto file_result = MemoryMappedFile::open(input_file);
    if (!file_result)
    {
        std::println(stderr, "File Error: {}", file_result.error());
        return 1;
    }

    auto header_result = validate_las_header(file_result->data(), file_result->size());
    if (!header_result)
    {
        std::println(stderr, "Header Error: {}", header_result.error());
        return 1;
    }
    const auto& view = header_result.value();

    const u8 format_id = view.header->point_data_record_format & 0x3Fu;
    const u32 classification_offset = (format_id <= 5) ? 15u : 16u;
    const u8 classification_mask = (format_id <= 5) ? 0x1Fu : 0xFFu;

    std::vector<u64> class_counts(256, 0);
    BoundingBox bbox;
    u64 points_processed = 0;

    auto start = std::chrono::high_resolution_clock::now();

    if (do_bbox)
    {
        if (has_filter && do_count)
        {
            bbox = compute_bounding_box_avx2<true, true>(
                file_result->data(), view, filter_mask, blend_mask, classification_offset, classification_mask,
                points_processed, class_counts
            );
        }
        else if (has_filter && !do_count)
        {
            bbox = compute_bounding_box_avx2<true, false>(
                file_result->data(), view, filter_mask, blend_mask, classification_offset, classification_mask,
                points_processed, class_counts
            );
        }
        else if (!has_filter && do_count)
        {
            bbox = compute_bounding_box_avx2<false, true>(
                file_result->data(), view, filter_mask, blend_mask, classification_offset, classification_mask,
                points_processed, class_counts
            );
        }
        else
        {
            bbox = compute_bounding_box_avx2<false, false>(
                file_result->data(), view, filter_mask, blend_mask, classification_offset, classification_mask,
                points_processed, class_counts
            );
        }
    }
    else if (has_filter || do_count)
    {
        auto run_result = dispatch_by_format(file_result->data(), view, [&](const auto& pt) {
            uint8_t c = pt.get_classification();
            if (has_filter && !filter_mask[c]) return;
            points_processed++;
            if (do_count) class_counts[c]++;
        });
        if (!run_result)
        {
            std::println(stderr, "Processing Error: {}", run_result.error());
            return 1;
        }
    }
    else
    {
        points_processed = view.point_count;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();

    std::println("Processed {} points in {:.4f}s\n", points_processed, seconds);

    if (do_bbox && points_processed > 0)
    {
        auto f = [](double v) { return std::format("{:.2f}", v); };
        std::string xs0 = f(bbox.min_x), xs1 = f(bbox.max_x);
        std::string ys0 = f(bbox.min_y), ys1 = f(bbox.max_y);
        std::string zs0 = f(bbox.min_z), zs1 = f(bbox.max_z);

        usize w0 = std::max({xs0.size(), ys0.size(), zs0.size()});
        usize w1 = std::max({xs1.size(), ys1.size(), zs1.size()});

        std::println("Bounding box:");
        std::println("  X: [{:<{}}, {:<{}}]", xs0, w0, xs1, w1);
        std::println("  Y: [{:<{}}, {:<{}}]", ys0, w0, ys1, w1);
        std::println("  Z: [{:<{}}, {:<{}}]\n", zs0, w0, zs1, w1);
    }

    if (do_count && points_processed > 0)
    {
        std::println("Classification report:");
        for (usize i = 0; i < class_counts.size(); ++i)
            if (class_counts[i] > 0) std::println("  Class {:<3} -> {:>8} points", i, class_counts[i]);
    }

    return 0;
}
} // namespace laspar