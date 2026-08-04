#include "memory_mapper.hpp"
#include "scalar_processing.hpp"
#include "simd_processing.hpp"
#include <cassert>
#include <cstdio>
#include <print>

using namespace laspar;

bool compare_results(const ProcessResult& r1, const ProcessResult& r2)
{
    bool ok = true;
    if (r1.points_processed != r2.points_processed)
    {
        std::print("\nPoints processed: scalar={} avx2={}", r1.points_processed, r2.points_processed);
        ok = false;
    }
    if (r1.points_dropped != r2.points_dropped)
    {
        std::print("\nPoints dropped: scalar={} avx2={}", r1.points_dropped, r2.points_dropped);
        ok = false;
    }

    if (std::abs(r1.sum_z - r2.sum_z) / std::max(1.0, std::abs(r1.sum_z)) > 1e-5)
    {
        std::print("\nSum Z: scalar={} avx2={}", r1.sum_z, r2.sum_z);
        ok = false;
    }
    if (std::abs(r1.sum_z2 - r2.sum_z2) / std::max(1.0, std::abs(r1.sum_z2)) > 1e-5)
    {
        std::print("\nSum Z2: scalar={} avx2={}", r1.sum_z2, r2.sum_z2);
        ok = false;
    }

    if (r1.points_processed > 0)
    {
        if (std::abs(r1.bbox.min_x - r2.bbox.min_x) > 1e-4)
        {
            std::print("\nbbox.min_x: scalar={} avx2={}", r1.bbox.min_x, r2.bbox.min_x);
            ok = false;
        }
        if (std::abs(r1.bbox.max_x - r2.bbox.max_x) > 1e-4)
        {
            std::print("\nbbox.max_x: scalar={} avx2={}", r1.bbox.max_x, r2.bbox.max_x);
            ok = false;
        }
        if (std::abs(r1.bbox.min_y - r2.bbox.min_y) > 1e-4)
        {
            std::print("\nbbox.min_y: scalar={} avx2={}", r1.bbox.min_y, r2.bbox.min_y);
            ok = false;
        }
        if (std::abs(r1.bbox.max_y - r2.bbox.max_y) > 1e-4)
        {
            std::print("\nbbox.max_y: scalar={} avx2={}", r1.bbox.max_y, r2.bbox.max_y);
            ok = false;
        }
        if (std::abs(r1.bbox.min_z - r2.bbox.min_z) > 1e-4)
        {
            std::print("\nbbox.min_z: scalar={} avx2={}", r1.bbox.min_z, r2.bbox.min_z);
            ok = false;
        }
        if (std::abs(r1.bbox.max_z - r2.bbox.max_z) > 1e-4)
        {
            std::print("\nbbox.max_z: scalar={} avx2={}", r1.bbox.max_z, r2.bbox.max_z);
            ok = false;
        }
    }

    if (r1.points_dropped > 0)
    {
        if (std::abs(r1.dropped_bbox.min_x - r2.dropped_bbox.min_x) > 1e-4)
        {
            std::print("\ndropped_bbox.min_x: scalar={} avx2={}", r1.dropped_bbox.min_x, r2.dropped_bbox.min_x);
            ok = false;
        }
        if (std::abs(r1.dropped_bbox.max_x - r2.dropped_bbox.max_x) > 1e-4)
        {
            std::print("\ndropped_bbox.max_x: scalar={} avx2={}", r1.dropped_bbox.max_x, r2.dropped_bbox.max_x);
            ok = false;
        }
        if (std::abs(r1.dropped_bbox.min_y - r2.dropped_bbox.min_y) > 1e-4)
        {
            std::print("\ndropped_bbox.min_y: scalar={} avx2={}", r1.dropped_bbox.min_y, r2.dropped_bbox.min_y);
            ok = false;
        }
        if (std::abs(r1.dropped_bbox.max_y - r2.dropped_bbox.max_y) > 1e-4)
        {
            std::print("\ndropped_bbox.max_y: scalar={} avx2={}", r1.dropped_bbox.max_y, r2.dropped_bbox.max_y);
            ok = false;
        }
        if (std::abs(r1.dropped_bbox.min_z - r2.dropped_bbox.min_z) > 1e-4)
        {
            std::print("\ndropped_bbox.min_z: scalar={} avx2={}", r1.dropped_bbox.min_z, r2.dropped_bbox.min_z);
            ok = false;
        }
        if (std::abs(r1.dropped_bbox.max_z - r2.dropped_bbox.max_z) > 1e-4)
        {
            std::print("\ndropped_bbox.max_z: scalar={} avx2={}", r1.dropped_bbox.max_z, r2.dropped_bbox.max_z);
            ok = false;
        }
    }
    return ok;
}

inline constinit i32 num_tests = 0;
inline constinit i32 passed_tests = 0;

inline void run_test_case(
    bool has_class_filter, bool has_coord_filter, bool do_count, bool do_bbox, bool do_elev, bool write_kept,
    bool write_dropped, bool has_decimation, const String& name, const u8* data, const HeaderView& view,
    const std::array<u8, 256>& filter_mask, const std::array<f64, 256>& blend_mask, u32 classification_offset,
    u8 classification_byte_mask, f64 filter_xmin, f64 filter_xmax, f64 filter_ymin, f64 filter_ymax, f64 filter_zmin,
    f64 filter_zmax, u64 keep_every
)
{
    num_tests++;
    std::print("Running test: {} ... ", name);
    std::fflush(stdout);

    std::vector<u64> counts_scalar(256, 0);
    std::vector<u64> counts_simd(256, 0);

    ProcessResult res_scalar = process_points_scalar(
        has_class_filter, has_coord_filter, do_count, do_bbox, do_elev, has_decimation, data, view, filter_mask,
        classification_offset, classification_byte_mask, counts_scalar, filter_xmin, filter_xmax, filter_ymin,
        filter_ymax, filter_zmin, filter_zmax, keep_every
    );

    ProcessResult res_simd = process_points_simd(
        has_class_filter, has_coord_filter, do_count, do_bbox, do_elev, write_kept, write_dropped, has_decimation, data,
        view, filter_mask, blend_mask, classification_offset, classification_byte_mask, counts_simd, filter_xmin,
        filter_xmax, filter_ymin, filter_ymax, filter_zmin, filter_zmax, keep_every, nullptr, nullptr
    );

    if (!compare_results(res_scalar, res_simd))
    {
        std::println("FAILED (ProcessResult mismatch)");
        return;
    }

    if (do_count)
    {
        for (usize i = 0; i < 256; ++i)
        {
            if (counts_scalar[i] != counts_simd[i])
            {
                std::println("FAILED (Class count mismatch at class {})", i);
                return;
            }
        }
    }

    if (do_elev && res_scalar.points_processed > 0)
    {
        const f64 n = as<f64>(res_scalar.points_processed);
        f64 elev_mean = res_scalar.sum_z / n;
        f64 mean_sq = res_scalar.sum_z2 / n;
        f64 variance = std::max(0.0, mean_sq - (elev_mean * elev_mean));
        f64 elev_std_dev = std::sqrt(variance);

        f64 hist_min = elev_mean - 3.0 * elev_std_dev;
        f64 hist_max = elev_mean + 3.0 * elev_std_dev;
        f64 range = (hist_max > hist_min) ? hist_max - hist_min : 1.0;
        i32 nb_bins = 20;
        f64 bin_step = range / nb_bins;

        std::vector<u64> bins_scalar(nb_bins, 0);
        std::vector<u64> bins_simd(nb_bins, 0);
        u64 underflow_scalar = 0, overflow_scalar = 0;
        u64 underflow_simd = 0, overflow_simd = 0;

        build_elev_histogram_scalar(
            has_class_filter, has_coord_filter, has_decimation, data, view, filter_mask, classification_offset,
            classification_byte_mask, filter_xmin, filter_xmax, filter_ymin, filter_ymax, filter_zmin, filter_zmax,
            keep_every, hist_min, hist_max, bin_step, nb_bins, bins_scalar, underflow_scalar, overflow_scalar
        );

        build_elev_histogram_simd(
            has_class_filter, has_coord_filter, has_decimation, data, view, filter_mask, blend_mask,
            classification_offset, classification_byte_mask, filter_xmin, filter_xmax, filter_ymin, filter_ymax,
            filter_zmin, filter_zmax, keep_every, hist_min, hist_max, bin_step, nb_bins, bins_simd, underflow_simd,
            overflow_simd
        );

        if (underflow_scalar != underflow_simd || overflow_scalar != overflow_simd)
        {
            std::println("FAILED (Histogram over/underflow mismatch)");
            return;
        }

        for (usize i = 0; i < bins_scalar.size(); ++i)
        {
            if (bins_scalar[i] != bins_simd[i])
            {
                std::println("FAILED (Histogram bin {} mismatch: scalar={} simd={})", i, bins_scalar[i], bins_simd[i]);
                return;
            }
        }
    }

    std::println("PASSED");
    passed_tests++;
}

int main(int argc, char** argv)
{
    using namespace laspar;
    if (argc < 2)
    {
        std::println(stderr, "Usage: las_test <test_file.las>");
        return 1;
    }

    auto file_result = MemoryMappedFile::open(argv[1]);
    if (!file_result)
    {
        std::println(stderr, "Failed to open test file: {}", argv[1]);
        return 1;
    }

    auto header_result = validate_las_header(file_result->data(), file_result->size());
    if (!header_result)
    {
        std::println(stderr, "Invalid LAS header.");
        return 1;
    }

    const auto& view = *header_result;
    const u8 format_id = view.header->point_data_record_format & 0x3Fu;
    const u32 classification_offset = (format_id <= 5) ? 15u : 16u;
    const u8 classification_mask = (format_id <= 5) ? 0x1Fu : 0xFFu;

    std::array<u8, 256> filter_mask_all = {};
    filter_mask_all.fill(1);
    std::array<f64, 256> blend_mask_all = {};
    blend_mask_all.fill(lane_pass);

    std::array<u8, 256> filter_mask_class2 = {};
    filter_mask_class2[2] = 1;
    std::array<f64, 256> blend_mask_class2 = {};
    blend_mask_class2[2] = lane_pass;

    f64 min_f = std::numeric_limits<f64>::lowest();
    f64 max_f = std::numeric_limits<f64>::max();

    run_test_case(
        false, false, true, true, true, false, false, false, "Full process (no filters)", file_result->data(), view,
        filter_mask_all, blend_mask_all, classification_offset, classification_mask, min_f, max_f, min_f, max_f, min_f,
        max_f, 1
    );

    run_test_case(
        true, false, true, true, true, false, false, false, "Class filter (class 2)", file_result->data(), view,
        filter_mask_class2, blend_mask_class2, classification_offset, classification_mask, min_f, max_f, min_f, max_f,
        min_f, max_f, 1
    );

    run_test_case(
        false, true, true, true, true, false, false, false, "Coord filter", file_result->data(), view, filter_mask_all,
        blend_mask_all, classification_offset, classification_mask, 651500.0, max_f, min_f, max_f, min_f, max_f, 1
    );

    run_test_case(
        true, true, true, true, true, false, false, false, "Class 2 + coord filter", file_result->data(), view,
        filter_mask_class2, blend_mask_class2, classification_offset, classification_mask, 651500.0, max_f, min_f,
        max_f, min_f, 70.0, 1
    );

    run_test_case(
        true, true, true, true, true, false, false, true, "Decimation (keep every 10)", file_result->data(), view,
        filter_mask_all, blend_mask_all, classification_offset, classification_mask, min_f, max_f, min_f, max_f, min_f,
        max_f, 10
    );

    run_test_case(
        true, false, true, true, true, false, false, true, "Decimation + class filter", file_result->data(), view,
        filter_mask_class2, blend_mask_class2, classification_offset, classification_mask, min_f, max_f, min_f, max_f,
        min_f, max_f, 15
    );

    run_test_case(
        false, true, true, true, true, false, false, true, "Decimation + coord filter", file_result->data(), view,
        filter_mask_all, blend_mask_all, classification_offset, classification_mask, 651200.0, max_f, min_f, max_f,
        min_f, max_f, 5
    );

    run_test_case(
        true, true, false, false, false, false, false, false, "Only filters", file_result->data(), view,
        filter_mask_class2, blend_mask_class2, classification_offset, classification_mask, min_f, 651500.0, min_f,
        max_f, min_f, max_f, 1
    );

    run_test_case(
        true, true, true, true, true, false, false, true, "All features ON", file_result->data(), view,
        filter_mask_class2, blend_mask_class2, classification_offset, classification_mask, min_f, 651500.0, min_f,
        6862500.0, min_f, max_f, 7
    );

    run_test_case(
        true, true, true, true, true, false, false, true, "Heavy decimation (keep every 123)", file_result->data(),
        view, filter_mask_all, blend_mask_all, classification_offset, classification_mask, min_f, max_f, min_f, max_f,
        min_f, max_f, 123
    );

    std::array<u8, 256> filter_mask_none = {};
    std::array<f64, 256> blend_mask_none = {};
    run_test_case(
        true, true, true, true, true, false, false, true, "Drop everything", file_result->data(), view,
        filter_mask_none, blend_mask_none, classification_offset, classification_mask, min_f, max_f, min_f, max_f,
        min_f, max_f, 1
    );

    run_test_case(
        false, false, true, false, false, false, false, false, "Only count (no filters)", file_result->data(), view,
        filter_mask_all, blend_mask_all, classification_offset, classification_mask, min_f, max_f, min_f, max_f, min_f,
        max_f, 1
    );

    std::println("\n===============================");
    std::println("TESTS PASSED: {} / {}", passed_tests, num_tests);
    std::println("===============================");

    return passed_tests != num_tests;
}