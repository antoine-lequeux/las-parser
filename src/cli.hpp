#include <array>
#include <chrono>
#include <cmath>
#include <lyra/lyra.hpp>
#include <optional>
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

        std::print("  {:<6} {:28} | ", std::format("[C{}]", i), get_asprs_class_name(format_id, i));
        for (i32 b = 0; b < bar_width; ++b) std::print("█");
        std::println(" {:L}", class_counts[i]);
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

        std::print("  {:>16} | ", label);
        for (i32 b = 0; b < bar_width; ++b) std::print("█");
        std::println(" {:L}", count);
    };

    if (underflow_count > 0) print_bar(std::format("< {:.2f} ", hist_min), underflow_count);

    for (i32 i = 0; i < num_bins; i++)
    {
        double bin_lo = hist_min + (i * bin_step);
        double bin_hi = bin_lo + bin_step;
        print_bar(std::format("[{:.2f}, {:.2f}]", bin_lo, bin_hi), z_bins[i]);
    }

    if (overflow_count > 0) print_bar(std::format("> {:.2f} ", hist_max), overflow_count);

    std::println("");
}

inline int launch_cli(int argc, const char** argv)
{
    std::string input_file;
    std::vector<int> keep_classes;
    std::optional<double> opt_xmin, opt_xmax, opt_ymin, opt_ymax, opt_zmin, opt_zmax;
    bool show_help = false;
    bool do_bbox = false;
    bool do_count = false;
    bool do_elev = false;
    u16 hist_width = 100;
    u16 nb_bins = 20;

    auto cli =
        lyra::help(show_help) | lyra::arg(input_file, "input_file")("The LAS file to process").required() |
        lyra::opt(keep_classes, "class")["-k"]["--keep-class"]("Filter by classification (can be chained)") |
        lyra::opt(opt_xmin, "value")["--xmin"]("Filter points with X >= value") |
        lyra::opt(opt_xmax, "value")["--xmax"]("Filter points with X <= value") |
        lyra::opt(opt_ymin, "value")["--ymin"]("Filter points with Y >= value") |
        lyra::opt(opt_ymax, "value")["--ymax"]("Filter points with Y <= value") |
        lyra::opt(opt_zmin, "value")["--zmin"]("Filter points with Z >= value") |
        lyra::opt(opt_zmax, "value")["--zmax"]("Filter points with Z <= value") |
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

    const bool has_class_filter = !keep_classes.empty();
    const bool has_coord_filter = opt_xmin.has_value() || opt_xmax.has_value() || opt_ymin.has_value() ||
                                  opt_ymax.has_value() || opt_zmin.has_value() || opt_zmax.has_value();

    const double filter_xmin = opt_xmin.value_or(std::numeric_limits<double>::lowest());
    const double filter_xmax = opt_xmax.value_or(std::numeric_limits<double>::max());
    const double filter_ymin = opt_ymin.value_or(std::numeric_limits<double>::lowest());
    const double filter_ymax = opt_ymax.value_or(std::numeric_limits<double>::max());
    const double filter_zmin = opt_zmin.value_or(std::numeric_limits<double>::lowest());
    const double filter_zmax = opt_zmax.value_or(std::numeric_limits<double>::max());

    if (has_class_filter)
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

    struct TimingRecord
    {
        std::string description;
        double seconds;
        std::string suffix = "";
    };
    std::vector<TimingRecord> timings;

    {
        auto start = std::chrono::high_resolution_clock::now();

        // Use the data once so it's cached and the point processing doesn't hit a page fault.
        const u8* p = file_result->data() + view.point_data_offset;
        const u64 size = static_cast<u64>(view.point_count) * view.point_record_length;
        auto* p1 = static_cast<const volatile u8*>(p);
        const usize page_size = 4096;
        for (usize i = 0; i < size; i += page_size) (void)p1[i];

        auto end = std::chrono::high_resolution_clock::now();
        timings.push_back({"Pre-loaded file data in", std::chrono::duration<double>(end - start).count()});
    }

    const u8 format_id = view.header->point_data_record_format & 0x3Fu;
    const u32 classification_offset = (format_id <= 5) ? 15u : 16u;
    const u8 classification_mask = (format_id <= 5) ? 0x1Fu : 0xFFu;

    std::vector<u64> class_counts(256, 0);

    auto start_process = std::chrono::high_resolution_clock::now();

    if (do_elev || do_count || do_bbox)
    {
        ProcessResult pr = dispatch_avx2(
            has_class_filter, has_coord_filter, do_count, do_bbox, do_elev, file_result->data(), view, filter_mask,
            blend_mask, classification_offset, classification_mask, class_counts, filter_xmin, filter_xmax, filter_ymin,
            filter_ymax, filter_zmin, filter_zmax
        );

        const u64 points_processed = pr.points_processed;

        auto end_process = std::chrono::high_resolution_clock::now();
        double process_seconds = std::chrono::duration<double>(end_process - start_process).count();

        double mp_s = (view.point_count / 1'000'000.0) / process_seconds;
        double gb_s = ((view.point_count * view.point_record_length) / 1'000'000'000.0) / process_seconds;
        std::string process_suffix = std::format(" ({:.1f} Mp/s, {:.2f} GB/s)", mp_s, gb_s);

        timings.push_back({std::format("Processed {} points in", points_processed), process_seconds, process_suffix});

        std::vector<u64> z_bins(nb_bins, 0);
        u64 underflow_count = 0, overflow_count = 0;
        double elev_mean = 0, elev_std_dev = 0, hist_min = 0, hist_max = 0;

        if (do_elev && points_processed > 0)
        {
            auto start1 = std::chrono::high_resolution_clock::now();
            const double n = static_cast<double>(points_processed);
            elev_mean = pr.sum_z / n;
            const double mean_sq = pr.sum_z2 / n;
            const double variance = std::max(0.0, mean_sq - (elev_mean * elev_mean));
            elev_std_dev = std::sqrt(variance);

            const double raw_min = elev_mean - 3.0 * elev_std_dev;
            const double raw_max = elev_mean + 3.0 * elev_std_dev;
            hist_min = raw_min;
            hist_max = raw_max;

            const double range = (hist_max > hist_min) ? hist_max - hist_min : 1.0;
            const double bin_step = range / nb_bins;

            dispatch_histogram_avx2(
                has_class_filter, has_coord_filter, file_result->data(), view, filter_mask, blend_mask,
                classification_offset, classification_mask, filter_xmin, filter_xmax, filter_ymin, filter_ymax,
                filter_zmin, filter_zmax, hist_min, hist_max, bin_step, nb_bins, z_bins, underflow_count, overflow_count
            );

            auto end1 = std::chrono::high_resolution_clock::now();
            timings.push_back({"Computed elev. hist. in", std::chrono::duration<double>(end1 - start1).count()});
        }

        usize max_len = 0;
        for (const auto& t : timings) max_len = std::max(max_len, t.description.size());

        for (const auto& t : timings)
        {
            usize dashes = max_len - t.description.size() + 3;
            std::println("{} {} {:.4f} sec{}", t.description, std::string(dashes, '-'), t.seconds, t.suffix);
        }
        std::println();

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
            print_elev_histogram(
                z_bins, underflow_count, overflow_count, elev_mean, elev_std_dev, hist_min, hist_max, hist_width
            );
        }

        if (do_count && points_processed > 0) print_class_histogram(class_counts, format_id, hist_width);
    }
    else
    {
        std::println("No work to do. Use '-b', '-e', or '-c' to perform operations on points.");
    }

    return 0;
}
} // namespace laspar