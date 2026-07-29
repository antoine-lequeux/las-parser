#include <array>
#include <chrono>
#include <cmath>
#include <lyra/lyra.hpp>
#include <print>
#include <sstream>
#include <vector>

#include "avx2_processing.hpp"
#include "header_view.hpp"
#include "memory_mapper.hpp"

template <>
struct std::formatter<lyra::cli> : std::formatter<std::string_view>
{
    auto format(const lyra::cli& cli, std::format_context& ctx) const
    {
        std::ostringstream ss;
        ss << cli;
        return std::formatter<std::string_view>::format(ss.str(), ctx);
    }
};

namespace laspar
{

// The classification for all point formats is defined here:
// https://paulbourke.net/dataformats/laz/LAS_1_4_r15.pdf (p.19 & p.30).
inline const char* get_asprs_class_name(u8 format_id, i32 class_id)
{
    // Common classes.
    switch (class_id)
    {
        case 0: return "Created, Never Classified";
        case 1: return "Unclassified";
        case 2: return "Ground";
        case 3: return "Low Vegetation";
        case 4: return "Medium Vegetation";
        case 5: return "High Vegetation";
        case 6: return "Building";
        case 7: return "Low Point (Noise)";
        case 9: return "Water";
    }

    // Classes for formats 0 to 5.
    if (format_id <= 5)
    {
        switch (class_id)
        {
            case 8: return "Model Key-Point (Mass Point)";
            case 12: return "Overlap Points";
            default: return "Reserved for ASPRS Def.";
        }
    }
    // Classes for formats 6 to 10.
    else
    {
        switch (class_id)
        {
            case 8: return "Reserved";
            case 10: return "Rail";
            case 11: return "Road Surface";
            case 12: return "Reserved";
            case 13: return "Wire - Guard (Shield)";
            case 14: return "Wire - Conductor (Phase)";
            case 15: return "Transmission Tower";
            case 16: return "Wire-Structure Connector";
            case 17: return "Bridge Deck";
            case 18: return "High Noise";
            case 19: return "Overhead Structure";
            case 20: return "Ignored Ground";
            case 21: return "Snow";
            case 22: return "Temporal Exclusion";
            default:
                if (class_id >= 64 && class_id <= 255) return "User Definable";
                return "Reserved";
        }
    }
}

inline void print_class_histogram(const std::vector<u64>& class_counts, u8 format_id, u16 max_bar_width = 100)
{
    u64 max_count = 0;
    for (u64 count : class_counts) max_count = std::max(max_count, count);
    if (max_count == 0) return;

    std::println("Classification report:");

    for (usize i = 0; i < class_counts.size(); ++i)
    {
        if (class_counts[i] == 0) continue;

        double ratio = static_cast<double>(class_counts[i]) / max_count;
        i32 bar_width = static_cast<i32>(std::round(ratio * max_bar_width));
        if (bar_width == 0) bar_width = 1;

        std::string bar(bar_width, '#');

        i32 space_count = std::max(0, static_cast<i32>(max_bar_width + 1) - bar_width);
        std::string padding(space_count, ' ');

        std::println(
            "  {:<6} {:28} | {}{}{}", std::format("[C{}]", i), get_asprs_class_name(format_id, i), bar, padding,
            class_counts[i]
        );
    }
    std::println("");
}

inline void print_elev_histogram(
    const std::vector<u64>& z_bins, u64 underflow_count, u64 overflow_count, double mean, double std_dev,
    double hist_min, double hist_max, u16 max_bar_width = 100
)
{
    const i32 num_bins = static_cast<i32>(z_bins.size());
    const double bin_step = (hist_max > hist_min) ? (hist_max - hist_min) / num_bins : 1.0;

    u64 max_count = std::max(underflow_count, overflow_count);
    for (u64 count : z_bins) max_count = std::max(max_count, count);
    if (max_count == 0) return;

    std::println("Elevation histogram (mean: {:.2f}, std dev: {:.2f}):", mean, std_dev);

    auto print_bar = [&](std::string_view label, u64 count) {
        double ratio = static_cast<double>(count) / max_count;
        i32 bar_width = static_cast<i32>(std::round(ratio * max_bar_width));
        if (count > 0 && bar_width == 0) bar_width = 1;

        i32 space_count = std::max(0, static_cast<i32>(max_bar_width + 1) - bar_width);
        std::string bar(bar_width, '#');
        std::string padding(space_count, ' ');

        std::println("  {:>16} | {}{}{}", label, bar, padding, count);
    };

    if (underflow_count > 0) print_bar(std::format("< {:.1f} ", hist_min), underflow_count);

    for (i32 i = 0; i < num_bins; i++)
    {
        double bin_lo = hist_min + (i * bin_step);
        double bin_hi = bin_lo + bin_step;
        print_bar(std::format("[{:.1f}, {:.1f}]", bin_lo, bin_hi), z_bins[i]);
    }

    if (overflow_count > 0) print_bar(std::format("> {:.1f} ", hist_max), overflow_count);

    std::println("");
}

inline int launch_cli(int argc, const char** argv)
{
    std::string input_file;
    std::vector<int> keep_classes;
    bool show_help = false;
    bool do_bbox = false;
    bool do_count = false;
    bool do_elev = false;
    u16 hist_width = 100;
    u16 nb_bins = 20;

    auto cli =
        lyra::help(show_help) | lyra::arg(input_file, "input_file")("The LAS file to process").required() |
        lyra::opt(keep_classes, "class")["-k"]["--keep-class"]("Filter by classification (can be chained)") |
        lyra::opt(do_bbox)["-b"]["--bbox"]("Compute bounding box of filtered points") |
        lyra::opt(do_count)["-c"]["--count"](
            "Show histogram of filtered points by class; optionally set max histogram bar width (default: 100)"
        ) |
        lyra::opt(do_elev)["-e"]["--elevation"](
            "Show histogram of filtered points by elevation; optionally set max histogram bar width (default: 100)"
        ) |
        lyra::opt(hist_width, "width")["-w"]["--hist-width"]("Max histogram bar width in characters (default: 100)") |
        lyra::opt(nb_bins, "bins")["--bins"]("Number of bins in the elevation histogram (default: 20)");

    auto cli_result = cli.parse({argc, argv});
    if (!cli_result)
    {
        std::println(stderr, "CLI ERROR: {}", cli_result.message());
        return 1;
    }
    if (show_help)
    {
        std::println("\n{}", cli);
        return 0;
    }

    nb_bins = std::max(static_cast<u16>(1), nb_bins);

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

    {
        auto start = std::chrono::high_resolution_clock::now();

        // Use the data once so it's cached and the point processing doesn't hit a page fault.
        const u8* p = file_result->data() + view.point_data_offset;
        const u64 size = static_cast<u64>(view.point_count) * view.point_record_length;
        auto* p1 = static_cast<const volatile u8*>(p);
        const usize page_size = 4096;
        for (usize i = 0; i < size; i += page_size) (void)p1[i];

        auto end = std::chrono::high_resolution_clock::now();
        double seconds = std::chrono::duration<double>(end - start).count();
        std::println("Loaded file data in {:.4f} sec", seconds);
    }

    const u8 format_id = view.header->point_data_record_format & 0x3Fu;
    const u32 classification_offset = (format_id <= 5) ? 15u : 16u;
    const u8 classification_mask = (format_id <= 5) ? 0x1Fu : 0xFFu;

    std::vector<u64> class_counts(256, 0);

    auto start = std::chrono::high_resolution_clock::now();

    ProcessResult pr = dispatch_avx2(
        has_filter, do_count, do_bbox, do_elev, file_result->data(), view, filter_mask, blend_mask,
        classification_offset, classification_mask, class_counts
    );

    const u64 points_processed = pr.points_processed;

    auto end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();

    std::println("Processed {} points in {:.4f} sec\n", points_processed, seconds);

    if (do_bbox && points_processed > 0)
    {
        auto f = [](double v) { return std::format("{:.2f}", v); };
        std::string xs0 = f(pr.bbox.min_x), xs1 = f(pr.bbox.max_x);
        std::string ys0 = f(pr.bbox.min_y), ys1 = f(pr.bbox.max_y);
        std::string zs0 = f(pr.bbox.min_z), zs1 = f(pr.bbox.max_z);

        usize w0 = std::max({xs0.size(), ys0.size(), zs0.size()});
        usize w1 = std::max({xs1.size(), ys1.size(), zs1.size()});

        std::println("Bounding box:");
        std::println("  X: [{:<{}}, {:<{}}]", xs0, w0, xs1, w1);
        std::println("  Y: [{:<{}}, {:<{}}]", ys0, w0, ys1, w1);
        std::println("  Z: [{:<{}}, {:<{}}]\n", zs0, w0, zs1, w1);
    }

    if (do_elev && points_processed > 0)
    {
        const double n = static_cast<double>(points_processed);
        const double mean = pr.sum_z / n;
        const double mean_sq = pr.sum_z2 / n;
        const double variance = std::max(0.0, mean_sq - (mean * mean));
        const double std_dev = std::sqrt(variance);

        const double raw_min = mean - 3.0 * std_dev;
        const double raw_max = mean + 3.0 * std_dev;
        const double hist_min = raw_min;
        const double hist_max = raw_max;

        std::vector<u64> z_bins(nb_bins, 0);
        u64 underflow_count = 0, overflow_count = 0;

        const double range = (hist_max > hist_min) ? hist_max - hist_min : 1.0;
        const double bin_step = range / nb_bins;

        const u8* p2 = file_result->data() + view.point_data_offset;
        for (u64 i = 0; i < view.point_count; ++i)
        {
            if (has_filter)
            {
                const u8 c = p2[classification_offset] & classification_mask;
                if (!filter_mask[c])
                {
                    p2 += view.point_record_length;
                    continue;
                }
            }
            const auto* pt = reinterpret_cast<const LasPointCoordinates*>(p2);
            const double z = (pt->z * view.header->z_scale_factor) + view.header->z_offset;

            if (z < hist_min)
                ++underflow_count;
            else if (z >= hist_max)
                ++overflow_count;
            else
            {
                i32 bin = static_cast<i32>((z - hist_min) / bin_step);
                z_bins[std::min(bin, nb_bins - 1)]++;
            }
            p2 += view.point_record_length;
        }

        print_elev_histogram(z_bins, underflow_count, overflow_count, mean, std_dev, hist_min, hist_max, hist_width);
    }

    if (do_count && points_processed > 0) print_class_histogram(class_counts, format_id, hist_width);

    return 0;
}
} // namespace laspar