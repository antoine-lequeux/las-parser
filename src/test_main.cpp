#include "header_utils.hpp"
#include "header_view.hpp"
#include "memory_mapper.hpp"
#include "scalar_processing.hpp"
#include "simd_processing.hpp"
#if defined(_MSC_VER)
    #define BOOST_UT_DISABLE_MODULE
#endif
#include <boost/ut.hpp>

#include <cassert>
#include <cstdio>
#include <print>
#include <vector>

using namespace laspar;
using namespace boost::ut;

auto approx_eq(f64 expected, f64 tolerance = 1e-4)
{
    return [=](f64 actual) { return std::abs(expected - actual) <= tolerance; };
}

void verify_written_file(
    const std::string& path, u64 expected_points, const BoundingBox& expected_bbox,
    const std::vector<u64>& expected_counts, bool do_count
)
{
    auto mmap = MemoryMappedFile::open(path);
    expect(mmap.has_value() >> fatal) << "Failed to open written file: " << path;

    auto hdr = validate_las_header(mmap->data(), mmap->size());
    expect(hdr.has_value() >> fatal) << "Written file has invalid header or is corrupt";

    expect(eq(hdr->point_count, expected_points));

    // Check bounding box
    if (expected_points == 0)
    {
        expect(approx_eq(0.0)(hdr->header->min_x));
        expect(approx_eq(0.0)(hdr->header->max_x));
        expect(approx_eq(0.0)(hdr->header->min_y));
        expect(approx_eq(0.0)(hdr->header->max_y));
        expect(approx_eq(0.0)(hdr->header->min_z));
        expect(approx_eq(0.0)(hdr->header->max_z));
    }
    else
    {
        expect(approx_eq(expected_bbox.min_x)(hdr->header->min_x));
        expect(approx_eq(expected_bbox.max_x)(hdr->header->max_x));
        expect(approx_eq(expected_bbox.min_y)(hdr->header->min_y));
        expect(approx_eq(expected_bbox.max_y)(hdr->header->max_y));
        expect(approx_eq(expected_bbox.min_z)(hdr->header->min_z));
        expect(approx_eq(expected_bbox.max_z)(hdr->header->max_z));
    }

    if (expected_points == 0) return;

    // Run scalar process to verify file content
    std::vector<u64> actual_counts(256, 0);
    std::array<u8, 256> filter_mask_all = {};
    filter_mask_all.fill(1);

    auto view = *hdr;
    const u8 format_id = view.header->point_data_record_format & 0x3Fu;
    const u32 class_off = (format_id <= 5) ? 15u : 16u;
    const u8 class_mask = (format_id <= 5) ? 0x1Fu : 0xFFu;

    f64 min_f = std::numeric_limits<f64>::lowest();
    f64 max_f = std::numeric_limits<f64>::max();

    auto res = process_points_scalar(
        false, false, do_count, true, false, false, mmap->data(), view, filter_mask_all, class_off, class_mask,
        actual_counts, min_f, max_f, min_f, max_f, min_f, max_f, 1
    );

    expect(eq(res.points_processed, expected_points));
    expect(approx_eq(expected_bbox.min_x)(res.bbox.min_x));
    expect(approx_eq(expected_bbox.max_x)(res.bbox.max_x));
    expect(approx_eq(expected_bbox.min_y)(res.bbox.min_y));
    expect(approx_eq(expected_bbox.max_y)(res.bbox.max_y));
    expect(approx_eq(expected_bbox.min_z)(res.bbox.min_z));
    expect(approx_eq(expected_bbox.max_z)(res.bbox.max_z));

    if (do_count)
        for (usize i = 0; i < 256; ++i) expect(eq(actual_counts[i], expected_counts[i]));
}

int main(int argc, char** argv)
{
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
        std::println(stderr, "Invalid LAS header: {}", header_result.error());
        return 1;
    }

    const auto& view = *header_result;
    const u8 format_id = view.header->point_data_record_format & 0x3Fu;
    const u32 classification_offset = (format_id <= 5) ? 15u : 16u;
    const u8 classification_mask = (format_id <= 5) ? 0x1Fu : 0xFFu;

    // Run an initial scalar pass to extract dataset parameters
    std::vector<u64> counts_all(256, 0);
    std::array<u8, 256> filter_mask_all = {};
    filter_mask_all.fill(1);

    f64 min_f = std::numeric_limits<f64>::lowest();
    f64 max_f = std::numeric_limits<f64>::max();

    ProcessResult res_all = process_points_scalar(
        false, false, true, true, false, false, file_result->data(), view, filter_mask_all, classification_offset,
        classification_mask, counts_all, min_f, max_f, min_f, max_f, min_f, max_f, 1
    );

    f64 med_x = (res_all.bbox.min_x + res_all.bbox.max_x) / 2.0;
    f64 med_y = (res_all.bbox.min_y + res_all.bbox.max_y) / 2.0;
    f64 med_z = (res_all.bbox.min_z + res_all.bbox.max_z) / 2.0;

    std::vector<u8> active_classes;
    for (usize i = 0; i < 256; ++i)
        if (counts_all[i] > 0) active_classes.push_back(static_cast<u8>(i));

    u8 test_class_1 = active_classes.size() > 0 ? active_classes[0] : 0;
    u8 test_class_2 = active_classes.size() > 1 ? active_classes[1] : test_class_1;

    std::array<u8, 256> filter_mask_c1 = {};
    filter_mask_c1[test_class_1] = 1;
    std::array<f64, 256> blend_mask_c1 = {};
    blend_mask_c1[test_class_1] = lane_pass;

    std::array<u8, 256> filter_mask_c1_c2 = {};
    filter_mask_c1_c2[test_class_1] = 1;
    filter_mask_c1_c2[test_class_2] = 1;
    std::array<f64, 256> blend_mask_c1_c2 = {};
    blend_mask_c1_c2[test_class_1] = lane_pass;
    blend_mask_c1_c2[test_class_2] = lane_pass;

    std::string test_filter = "";
    if (argc >= 3) test_filter = argv[2];

    auto run_test_case = [&](bool has_class_filter, bool has_coord_filter, bool do_count, bool do_bbox, bool do_elev,
                             bool write_kept, bool write_dropped, bool has_decimation, const std::string& name,
                             const std::array<u8, 256>& f_mask, const std::array<f64, 256>& b_mask, f64 f_xmin,
                             f64 f_xmax, f64 f_ymin, f64 f_ymax, f64 f_zmin, f64 f_zmax, u64 keep_every) {
        if (!test_filter.empty() && name != test_filter) return;

        test(name) = [=, &file_result, &view]() mutable {
            std::vector<u64> counts_scalar(256, 0);
            std::vector<u64> counts_simd(256, 0);

            ProcessResult res_scalar = process_points_scalar(
                has_class_filter, has_coord_filter, do_count, do_bbox, do_elev, has_decimation, file_result->data(),
                view, f_mask, classification_offset, classification_mask, counts_scalar, f_xmin, f_xmax, f_ymin, f_ymax,
                f_zmin, f_zmax, keep_every
            );

            BufferedFileWriter kept_writer;
            BufferedFileWriter dropped_writer;

            std::string kept_path = "kept_test.las";
            std::string dropped_path = "dropped_test.las";

            if (write_kept)
            {
                expect(kept_writer.open(kept_path)) << "Could not open kept_test.las";
                kept_writer.write(file_result->data(), view.point_data_offset);
            }
            if (write_dropped)
            {
                expect(dropped_writer.open(dropped_path)) << "Could not open dropped_test.las";
                dropped_writer.write(file_result->data(), view.point_data_offset);
            }

            ProcessResult res_simd = process_points_simd(
                has_class_filter, has_coord_filter, do_count, do_bbox, do_elev, write_kept, write_dropped,
                has_decimation, file_result->data(), view, f_mask, b_mask, classification_offset, classification_mask,
                counts_simd, f_xmin, f_xmax, f_ymin, f_ymax, f_zmin, f_zmax, keep_every, &kept_writer, &dropped_writer
            );

            // Update headers properly if files were written
            if (write_kept)
            {
                LasHeader kept_header = *view.header;
                update_header_for_write(kept_header, res_simd.points_processed, res_simd.bbox);
                kept_writer.seek(0);
                kept_writer.write(reinterpret_cast<const u8*>(&kept_header), sizeof(LasHeader));
                kept_writer.close();
            }

            if (write_dropped)
            {
                LasHeader dropped_header = *view.header;
                update_header_for_write(dropped_header, res_simd.points_dropped, res_simd.dropped_bbox);
                dropped_writer.seek(0);
                dropped_writer.write(reinterpret_cast<const u8*>(&dropped_header), sizeof(LasHeader));
                dropped_writer.close();
            }

            expect(eq(res_scalar.points_processed, res_simd.points_processed));
            expect(eq(res_scalar.points_dropped, res_simd.points_dropped));

            if (do_elev)
            {
                expect(approx_eq(res_scalar.sum_z, 1.0)(res_simd.sum_z));
                expect(approx_eq(res_scalar.sum_z2, 1000.0)(res_simd.sum_z2));
            }

            if (do_bbox && res_scalar.points_processed > 0)
            {
                expect(approx_eq(res_scalar.bbox.min_x)(res_simd.bbox.min_x));
                expect(approx_eq(res_scalar.bbox.max_x)(res_simd.bbox.max_x));
                expect(approx_eq(res_scalar.bbox.min_y)(res_simd.bbox.min_y));
                expect(approx_eq(res_scalar.bbox.max_y)(res_simd.bbox.max_y));
                expect(approx_eq(res_scalar.bbox.min_z)(res_simd.bbox.min_z));
                expect(approx_eq(res_scalar.bbox.max_z)(res_simd.bbox.max_z));
            }

            if (do_bbox && write_dropped && res_scalar.points_dropped > 0)
            {
                expect(approx_eq(res_scalar.dropped_bbox.min_x)(res_simd.dropped_bbox.min_x));
                expect(approx_eq(res_scalar.dropped_bbox.max_x)(res_simd.dropped_bbox.max_x));
                expect(approx_eq(res_scalar.dropped_bbox.min_y)(res_simd.dropped_bbox.min_y));
                expect(approx_eq(res_scalar.dropped_bbox.max_y)(res_simd.dropped_bbox.max_y));
                expect(approx_eq(res_scalar.dropped_bbox.min_z)(res_simd.dropped_bbox.min_z));
                expect(approx_eq(res_scalar.dropped_bbox.max_z)(res_simd.dropped_bbox.max_z));
            }

            if (do_count)
                for (usize i = 0; i < 256; ++i)
                    expect(eq(counts_scalar[i], counts_simd[i])) << "Class count mismatch at class " << i;

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
                    has_class_filter, has_coord_filter, has_decimation, file_result->data(), view, f_mask,
                    classification_offset, classification_mask, f_xmin, f_xmax, f_ymin, f_ymax, f_zmin, f_zmax,
                    keep_every, hist_min, hist_max, bin_step, nb_bins, bins_scalar, underflow_scalar, overflow_scalar
                );

                build_elev_histogram_simd(
                    has_class_filter, has_coord_filter, has_decimation, file_result->data(), view, f_mask, b_mask,
                    classification_offset, classification_mask, f_xmin, f_xmax, f_ymin, f_ymax, f_zmin, f_zmax,
                    keep_every, hist_min, hist_max, bin_step, nb_bins, bins_simd, underflow_simd, overflow_simd
                );

                auto approx_count = [](u64 expected, u64 actual) {
                    return expected >= actual ? (expected - actual <= 10) : (actual - expected <= 10);
                };

                expect(approx_count(underflow_scalar, underflow_simd)) << "Underflow mismatch";
                expect(approx_count(overflow_scalar, overflow_simd)) << "Overflow mismatch";
                for (i32 i = 0; i < nb_bins; ++i)
                    expect(approx_count(bins_scalar[i], bins_simd[i])) << "Z bin mismatch at bin " << i;
            }

            if (write_kept)
            {
                LasHeader kept_header = *view.header;
                update_header_for_write(kept_header, res_scalar.points_processed, res_scalar.bbox);
                kept_writer.seek(0);
                kept_writer.write(&kept_header, sizeof(LasHeader));
                kept_writer.close();
            }

            if (write_dropped)
            {
                LasHeader dropped_header = *view.header;
                update_header_for_write(dropped_header, res_scalar.points_dropped, res_scalar.dropped_bbox);
                dropped_writer.seek(0);
                dropped_writer.write(&dropped_header, sizeof(LasHeader));
                dropped_writer.close();
            }

            if (write_kept)
                verify_written_file(kept_path, res_scalar.points_processed, res_scalar.bbox, counts_scalar, do_count);

            if (write_dropped)
            {
                verify_written_file(
                    dropped_path, res_scalar.points_dropped, res_scalar.dropped_bbox, counts_scalar, false
                );
            }
        };
    };

    // 1. Full pass
    run_test_case(
        false, false, true, true, true, false, false, false, "Full process (no filters)", filter_mask_all,
        std::array<f64, 256> {}, min_f, max_f, min_f, max_f, min_f, max_f, 1
    );

    // 2. Class filter (1 class)
    run_test_case(
        true, false, true, true, true, false, false, false, "Class filter (1 class)", filter_mask_c1, blend_mask_c1,
        min_f, max_f, min_f, max_f, min_f, max_f, 1
    );

    // 3. Class filter (2 classes) + write kept
    run_test_case(
        true, false, true, true, true, true, false, false, "Class filter (2 classes) + write kept", filter_mask_c1_c2,
        blend_mask_c1_c2, min_f, max_f, min_f, max_f, min_f, max_f, 1
    );

    // 4. Coord filter (X > med_x)
    run_test_case(
        false, true, true, true, true, false, false, false, "Coord filter (X > med_x)", filter_mask_all,
        std::array<f64, 256> {}, med_x, max_f, min_f, max_f, min_f, max_f, 1
    );

    // 5. Coord filter (Y < med_y, Z > med_z)
    run_test_case(
        false, true, true, true, true, false, false, false, "Coord filter (Y < med_y, Z > med_z)", filter_mask_all,
        std::array<f64, 256> {}, min_f, max_f, min_f, med_y, med_z, max_f, 1
    );

    // 6. Class + coord + write dropped
    run_test_case(
        true, true, true, true, true, false, true, false, "Class + coord + write dropped", filter_mask_c1,
        blend_mask_c1, med_x, max_f, min_f, med_y, min_f, max_f, 1
    );

    // 7. Decimation (keep every 10)
    run_test_case(
        false, false, true, true, true, false, false, true, "Decimation (keep every 10)", filter_mask_all,
        std::array<f64, 256> {}, min_f, max_f, min_f, max_f, min_f, max_f, 10
    );

    // 8. Decimation + class filter + write kept
    run_test_case(
        true, false, true, true, true, true, false, true, "Decimation + class filter + write Kept", filter_mask_c1,
        blend_mask_c1, min_f, max_f, min_f, max_f, min_f, max_f, 15
    );

    // 9. Only filters (no count/elev/bbox)
    run_test_case(
        true, true, false, false, false, false, false, false, "Only filters (no count/elev/bbox)", filter_mask_c1,
        blend_mask_c1, min_f, max_f, min_f, max_f, min_f, max_f, 1
    );

    // 10. All features ON + write both
    run_test_case(
        true, true, true, true, true, true, true, true, "All features ON + write both", filter_mask_c1_c2,
        blend_mask_c1_c2, min_f, max_f, min_f, max_f, min_f, max_f, 7
    );

    // 11. Heavy decimation
    run_test_case(
        true, true, true, true, true, false, false, true, "Heavy decimation (keep every 123)", filter_mask_c1,
        blend_mask_c1, min_f, max_f, min_f, max_f, min_f, max_f, 123
    );

    // 12. Extremely tight bounding box (might yield 0 points)
    run_test_case(
        false, true, true, true, true, true, true, false, "Extremely tight coord filter", filter_mask_all,
        std::array<f64, 256> {}, med_x, med_x + 0.0001, med_y, med_y + 0.0001, min_f, max_f, 1
    );

    return 0;
}