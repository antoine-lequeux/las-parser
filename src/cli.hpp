#include <array>
#include <chrono>
#include <cmath>
#include <lyra/lyra.hpp>
#include <print>
#include <sstream>
#include <vector>

#include "ascii_exporter.hpp"
#include "file_writer.hpp"
#include "header_utils.hpp"
#include "header_view.hpp"
#include "memory_mapper.hpp"
#include "platform.hpp"
#include "simd_processing.hpp"
#include "types.hpp"

namespace std
{
template <>
struct formatter<lyra::cli>
{
    constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }

    auto format(const lyra::cli& cli, auto& ctx) const
    {
        std::ostringstream ss;
        ss << cli;
        return std::format_to(ctx.out(), "{}", ss.str());
    }
};
} // namespace std

namespace laspar
{

// The classification for all point formats is defined here:
// https://paulbourke.net/dataformats/laz/LAS_1_4_r15.pdf (p.19 & p.30).
inline const char* get_asprs_class_name(u8 format_id, usize class_id)
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

        f64 ratio = as<f64>(class_counts[i]) / as<f64>(max_count);
        i32 bar_width = as<i32>(std::round(ratio * max_bar_width));
        if (bar_width == 0) bar_width = 1;

        std::print("  {:<6} {:28} | ", std::format("[C{}]", i), get_asprs_class_name(format_id, i));
        for (i32 b = 0; b < bar_width; ++b) std::print("█");
        std::println(" {}", fmt_num(class_counts[i]));
    }
    std::println("");
}

inline void print_elev_histogram(
    const std::vector<u64>& z_bins, u64 underflow_count, u64 overflow_count, f64 mean, f64 std_dev, f64 hist_min,
    f64 hist_max, u16 max_bar_width = 100
)
{
    const usize num_bins = z_bins.size();
    const f64 bin_step = (hist_max > hist_min) ? (hist_max - hist_min) / as<f64>(num_bins) : 1.0;

    u64 max_count = std::max(underflow_count, overflow_count);
    for (u64 count : z_bins) max_count = std::max(max_count, count);
    if (max_count == 0) return;

    std::println("Elevation histogram (mean: {:.2f}, std dev: {:.2f}):", mean, std_dev);

    auto print_bar = [&](std::string_view label, u64 count) {
        f64 ratio = as<f64>(count) / as<f64>(max_count);
        i32 bar_width = as<i32>(std::round(ratio * max_bar_width));
        if (count > 0 && bar_width == 0) bar_width = 1;

        std::print("  {:>16} | ", label);
        for (i32 b = 0; b < bar_width; ++b) std::print("█");
        std::println(" {}", fmt_num(count));
    };

    if (underflow_count > 0) print_bar(std::format("< {:.2f} ", hist_min), underflow_count);

    for (usize i = 0; i < num_bins; i++)
    {
        f64 bin_lo = hist_min + (as<f64>(i) * bin_step);
        f64 bin_hi = bin_lo + bin_step;
        print_bar(std::format("[{:.2f}, {:.2f}]", bin_lo, bin_hi), z_bins[i]);
    }

    if (overflow_count > 0) print_bar(std::format("> {:.2f} ", hist_max), overflow_count);

    std::println("");
}

inline int launch_cli(int argc, const char** argv)
{
    std::string input_file;
    std::string write_kept;
    std::string write_dropped;
    std::string export_csv_file;
    std::string export_xyz_file;
    std::vector<usize> keep_classes;
    std::optional<f64> opt_xmin, opt_xmax, opt_ymin, opt_ymax, opt_zmin, opt_zmax;
    bool show_help = false;
    bool do_bbox = false;
    bool do_count = false;
    bool do_elev = false;
    bool do_header = false;
    bool do_lint = false;
    u16 hist_width = 100;
    u16 nb_bins = 20;
    u64 keep_every = 1;

    auto cli =
        lyra::help(show_help) | lyra::arg(input_file, "input_file")("The LAS file to process").required() |
        lyra::opt(keep_classes, "class")["-k"]["--keep-class"]("Filter by classification (can be chained)") |
        lyra::opt(write_kept, "file")["--write-kept"]("Write points that survived the filter to a new LAS file") |
        lyra::opt(write_dropped, "file")["--write-dropped"](
            "Write points that did not survive the filter to a new LAS file"
        ) |
        lyra::opt(export_csv_file, "file")["--export-csv"]("Export filtered points to a CSV file") |
        lyra::opt(export_xyz_file, "file")["--export-xyz"]("Export filtered points to an XYZ file") |
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
        lyra::opt(do_header)["-H"]["--header"]("Print the LAS header metadata") |
        lyra::opt(do_lint)["-L"]["--lint"]("Check the LAS header for corruption or mismatch") |
        lyra::opt(keep_every, "N")["--keep-every"]("Keep 1 in every N points; can be used with other filters") |
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

    nb_bins = std::max(as<u16>(1), nb_bins);

    std::array<u8, 256> filter_mask = {};
    std::array<f64, 256> blend_mask = {};

    const bool has_class_filter = !keep_classes.empty();
    const bool has_coord_filter = opt_xmin || opt_xmax || opt_ymin || opt_ymax || opt_zmin || opt_zmax;

    const f64 filter_xmin = opt_xmin.value_or(std::numeric_limits<f64>::lowest());
    const f64 filter_xmax = opt_xmax.value_or(std::numeric_limits<f64>::max());
    const f64 filter_ymin = opt_ymin.value_or(std::numeric_limits<f64>::lowest());
    const f64 filter_ymax = opt_ymax.value_or(std::numeric_limits<f64>::max());
    const f64 filter_zmin = opt_zmin.value_or(std::numeric_limits<f64>::lowest());
    const f64 filter_zmax = opt_zmax.value_or(std::numeric_limits<f64>::max());

    if (has_class_filter)
    {
        for (usize c : keep_classes)
        {
            if (c < 256)
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
    const auto& view = *header_result;

    if (do_header) print_header(*view.header, input_file);

    if (do_lint)
    {
        bool passed = lint_header(*view.header, file_result->size());
        if (!passed) return 1;
    }

    struct TimingRecord
    {
        std::string description;
        f64 seconds;
        std::string suffix = "";
    };
    std::vector<TimingRecord> timings;

    {
        auto start = std::chrono::high_resolution_clock::now();

        // Use the data once so it's cached and the point processing doesn't hit a page fault.
        const u8* p = file_result->data() + view.point_data_offset;
        const u64 size = as<u64>(view.point_count) * view.point_record_length;

        platform::prefetch_memory(p, size);

        auto end = std::chrono::high_resolution_clock::now();
        timings.push_back({"Pre-loaded file data in", std::chrono::duration<f64>(end - start).count()});
    }

    bool do_write_kept = !write_kept.empty();
    bool do_write_dropped = !write_dropped.empty();

    BufferedFileWriter kept_writer, dropped_writer;

    if (do_write_kept)
    {
        if (!kept_writer.open(write_kept))
        {
            std::println(stderr, "Error: Could not open output file '{}' for writing kept points.", write_kept);
            return 1;
        }
        // Reserve space for header.
        kept_writer.write(file_result->data(), view.point_data_offset);
    }

    if (do_write_dropped)
    {
        if (!dropped_writer.open(write_dropped))
        {
            std::println(stderr, "Error: Could not open output file '{}' for writing dropped points.", write_dropped);
            return 1;
        }
        // Reserve space for header.
        dropped_writer.write(file_result->data(), view.point_data_offset);
    }

    const u8 format_id = view.header->point_data_record_format & 0x3Fu;
    const u32 classification_offset = (format_id <= 5) ? 15u : 16u;
    const u8 classification_mask = (format_id <= 5) ? 0x1Fu : 0xFFu;

    std::vector<u64> class_counts(256, 0);

    auto start_process = std::chrono::high_resolution_clock::now();

    bool do_export_csv = !export_csv_file.empty();
    bool do_export_xyz = !export_xyz_file.empty();

    if (do_elev || do_count || do_bbox || do_write_kept || do_write_dropped || do_export_csv || do_export_xyz)
    {
        bool has_decimation = (keep_every > 1);
        ProcessResult pr = process_points_simd(
            has_class_filter, has_coord_filter, do_count, do_bbox, do_elev, do_write_kept, do_write_dropped,
            has_decimation, file_result->data(), view, filter_mask, blend_mask, classification_offset,
            classification_mask, class_counts, filter_xmin, filter_xmax, filter_ymin, filter_ymax, filter_zmin,
            filter_zmax, keep_every, &kept_writer, &dropped_writer
        );

        const u64 points_processed = pr.points_processed;

        auto end_process = std::chrono::high_resolution_clock::now();
        f64 total_process_seconds = std::chrono::duration<f64>(end_process - start_process).count();
        f64 compute_seconds = std::max(0.0001, total_process_seconds - pr.io_time_seconds);

        f64 mp_s = (as<f64>(view.point_count) / 1'000'000.0) / compute_seconds;
        f64 gb_s = (as<f64>(view.point_count * view.point_record_length) / 1'000'000'000.0) / compute_seconds;
        std::string process_suffix = std::format(" ({:.1f} Mp/s, {:.2f} GB/s)", mp_s, gb_s);

        timings.push_back(
            {std::format("Processed {} points in", fmt_num(points_processed)), compute_seconds, process_suffix}
        );

        if (do_write_kept || do_write_dropped)
        {
            u64 points_written = 0;
            if (do_write_kept) points_written += pr.points_processed;
            if (do_write_dropped) points_written += pr.points_dropped;

            f64 io_mp_s = (as<f64>(points_written) / 1'000'000.0) / std::max(0.0001, pr.io_time_seconds);
            f64 io_gb_s = (as<f64>(points_written * view.point_record_length) / 1'000'000'000.0) /
                          std::max(0.0001, pr.io_time_seconds);
            std::string io_suffix = std::format(" ({:.1f} Mp/s, {:.2f} GB/s)", io_mp_s, io_gb_s);
            timings.push_back(
                {std::format("Wrote {} points to disk in", points_written), pr.io_time_seconds, io_suffix}
            );
        }

        std::vector<u64> z_bins(nb_bins, 0);
        u64 underflow_count = 0, overflow_count = 0;
        f64 elev_mean = 0, elev_std_dev = 0, hist_min = 0, hist_max = 0;

        if (do_elev && points_processed > 0)
        {
            auto start1 = std::chrono::high_resolution_clock::now();
            const f64 n = as<f64>(points_processed);
            elev_mean = pr.sum_z / n;
            const f64 mean_sq = pr.sum_z2 / n;
            const f64 variance = std::max(0.0, mean_sq - (elev_mean * elev_mean));
            elev_std_dev = std::sqrt(variance);

            const f64 raw_min = elev_mean - 3.0 * elev_std_dev;
            const f64 raw_max = elev_mean + 3.0 * elev_std_dev;
            hist_min = raw_min;
            hist_max = raw_max;

            const f64 range = (hist_max > hist_min) ? hist_max - hist_min : 1.0;
            const f64 bin_step = range / nb_bins;

            build_elev_histogram_simd(
                has_class_filter, has_coord_filter, has_decimation, file_result->data(), view, filter_mask, blend_mask,
                classification_offset, classification_mask, filter_xmin, filter_xmax, filter_ymin, filter_ymax,
                filter_zmin, filter_zmax, keep_every, hist_min, hist_max, bin_step, nb_bins, z_bins, underflow_count,
                overflow_count
            );

            auto end1 = std::chrono::high_resolution_clock::now();
            timings.push_back({"Computed elev. hist. in", std::chrono::duration<f64>(end1 - start1).count()});
        }

        if (do_write_kept || do_write_dropped)
        {
            auto start2 = std::chrono::high_resolution_clock::now();

            if (do_write_kept)
            {
                LasHeader kept_header = *view.header;
                update_header_for_write(kept_header, pr.points_processed, pr.bbox);
                kept_writer.seek(0);
                kept_writer.write(&kept_header, sizeof(LasHeader));
                kept_writer.close();
            }
            if (do_write_dropped)
            {
                LasHeader dropped_header = *view.header;
                update_header_for_write(dropped_header, pr.points_dropped, pr.dropped_bbox);
                dropped_writer.seek(0);
                dropped_writer.write(&dropped_header, sizeof(LasHeader));
                dropped_writer.close();
            }

            auto end2 = std::chrono::high_resolution_clock::now();
            timings.push_back({"Updated output files in", std::chrono::duration<f64>(end2 - start2).count()});
        }

        if (do_export_csv)
        {
            auto start = std::chrono::high_resolution_clock::now();
            auto res = export_csv(
                export_csv_file, has_class_filter, has_coord_filter, has_decimation, file_result->data(), view,
                filter_mask, classification_offset, classification_mask, filter_xmin, filter_xmax, filter_ymin,
                filter_ymax, filter_zmin, filter_zmax, keep_every
            );
            if (!res)
                std::println(stderr, "Error exporting CSV: {}", res.error());
            else
            {
                auto end = std::chrono::high_resolution_clock::now();
                f64 io = *res;
                f64 comp = std::chrono::duration<f64>(end - start).count() - io;
                timings.push_back({std::format("Exported {} points to CSV in", fmt_num(points_processed)), comp});
            }
        }

        if (do_export_xyz)
        {
            auto start = std::chrono::high_resolution_clock::now();
            auto res = export_xyz(
                export_xyz_file, has_class_filter, has_coord_filter, has_decimation, file_result->data(), view,
                filter_mask, classification_offset, classification_mask, filter_xmin, filter_xmax, filter_ymin,
                filter_ymax, filter_zmin, filter_zmax, keep_every
            );
            if (!res)
                std::println(stderr, "Error exporting XYZ: {}", res.error());
            else
            {
                auto end = std::chrono::high_resolution_clock::now();
                f64 io = *res;
                f64 comp = std::chrono::duration<f64>(end - start).count() - io;
                timings.push_back({std::format("Exported {} points to XYZ in", fmt_num(points_processed)), comp});
            }
        }

        usize max_len = 0;
        for (const auto& t : timings) max_len = std::max(max_len, t.description.size());

        for (const auto& t : timings)
        {
            usize dashes = max_len - t.description.size() + 3;
            std::println("{} {} {:.4f} sec{}", t.description, std::string(dashes, '-'), t.seconds, t.suffix);
        }
        std::println("");

        if (do_bbox && points_processed > 0)
        {
            auto f = [](f64 v) { return std::format("{:.2f}", v); };
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
    else if (!do_header && !do_lint)
    {
        std::println("No work to do.");
    }

    return 0;
}
} // namespace laspar