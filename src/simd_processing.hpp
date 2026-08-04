#pragma once
#include "file_writer.hpp"
#include "header_view.hpp"
#include "las_data.hpp"
#include <array>
#include <bit>
#include <eve/module/core.hpp>
#include <eve/module/math.hpp>
#include <vector>

namespace laspar
{

template <std::unsigned_integral T>
[[nodiscard]] constexpr u32 popcount(T val)
{
    return as<u32>(std::popcount(val));
}

constexpr f64 lane_pass = std::bit_cast<f64>(~u64 {0});

alignas(32) inline const f64 decimation_mask_lut_data[16][4] = {
    {0.0, 0.0, 0.0, 0.0},
    {lane_pass, 0.0, 0.0, 0.0},
    {0.0, lane_pass, 0.0, 0.0},
    {lane_pass, lane_pass, 0.0, 0.0},
    {0.0, 0.0, lane_pass, 0.0},
    {lane_pass, 0.0, lane_pass, 0.0},
    {0.0, lane_pass, lane_pass, 0.0},
    {lane_pass, lane_pass, lane_pass, 0.0},
    {0.0, 0.0, 0.0, lane_pass},
    {lane_pass, 0.0, 0.0, lane_pass},
    {0.0, lane_pass, 0.0, lane_pass},
    {lane_pass, lane_pass, 0.0, lane_pass},
    {0.0, 0.0, lane_pass, lane_pass},
    {lane_pass, 0.0, lane_pass, lane_pass},
    {0.0, lane_pass, lane_pass, lane_pass},
    {lane_pass, lane_pass, lane_pass, lane_pass}
};

inline ProcessResult process_points_simd(
    bool has_class_filter, bool has_coord_filter, bool do_count, bool do_bbox, bool do_elev, bool write_kept,
    bool write_dropped, bool has_decimation, const u8* file_data, const HeaderView& view,
    const std::array<u8, 256>& filter_mask, const std::array<f64, 256>& blend_mask, u32 classification_offset,
    u8 classification_byte_mask, std::vector<u64>& out_class_counts, f64 filter_xmin, f64 filter_xmax, f64 filter_ymin,
    f64 filter_ymax, f64 filter_zmin, f64 filter_zmax, u64 keep_every, BufferedFileWriter* kept_writer,
    BufferedFileWriter* dropped_writer
)
{
    using wf64 = eve::wide<f64, eve::fixed<4>>;
    using wi32 = eve::wide<i32, eve::fixed<4>>;
    using wl64 = eve::logical<wf64>;

    ProcessResult result {};
    u64 current_offset = 0;

    const wf64 x_scale {view.header->x_scale_factor};
    const wf64 y_scale {view.header->y_scale_factor};
    const wf64 z_scale {view.header->z_scale_factor};

    const wf64 x_offset {view.header->x_offset};
    const wf64 y_offset {view.header->y_offset};
    const wf64 z_offset {view.header->z_offset};

    wf64 min_x, max_x, min_y, max_y, min_z, max_z;
    wf64 dropped_min_x, dropped_max_x, dropped_min_y, dropped_max_y, dropped_min_z, dropped_max_z;
    const wf64 pos_inf {std::numeric_limits<f64>::max()};
    const wf64 neg_inf {std::numeric_limits<f64>::lowest()};

    if (do_bbox || write_kept)
    {
        min_x = pos_inf;
        max_x = neg_inf;
        min_y = pos_inf;
        max_y = neg_inf;
        min_z = pos_inf;
        max_z = neg_inf;
    }
    if (write_dropped)
    {
        dropped_min_x = pos_inf;
        dropped_max_x = neg_inf;
        dropped_min_y = pos_inf;
        dropped_max_y = neg_inf;
        dropped_min_z = pos_inf;
        dropped_max_z = neg_inf;
    }

    wf64 sum_z_vec {0.0}, sum_z2_vec {0.0}, zero_pd {0.0};

    const u8* p = file_data + view.point_data_offset;
    const u32 stride = view.point_record_length;

    const wf64 vxmin {filter_xmin};
    const wf64 vxmax {filter_xmax};
    const wf64 vymin {filter_ymin};
    const wf64 vymax {filter_ymax};
    const wf64 vzmin {filter_zmin};
    const wf64 vzmax {filter_zmax};

    u64 i = 0;
    const u64 batch_limit = view.point_count - (view.point_count % 4);
    u64 passed_count = 0;
    u64 dropped_count = 0;

    for (; i < batch_limit; i += 4)
    {
        const auto* pt0 = reinterpret_cast<const LasPointCoordinates*>(p);
        const auto* pt1 = reinterpret_cast<const LasPointCoordinates*>(p + stride);
        const auto* pt2 = reinterpret_cast<const LasPointCoordinates*>(p + stride * 2);
        const auto* pt3 = reinterpret_cast<const LasPointCoordinates*>(p + stride * 3);

        wi32 vx_int {pt0->x, pt1->x, pt2->x, pt3->x};
        wi32 vy_int {pt0->y, pt1->y, pt2->y, pt3->y};
        wi32 vz_int {pt0->z, pt1->z, pt2->z, pt3->z};

        wf64 vx = eve::fma(eve::convert(vx_int, eve::as<f64>()), x_scale, x_offset);
        wf64 vy = eve::fma(eve::convert(vy_int, eve::as<f64>()), y_scale, y_offset);
        wf64 vz = eve::fma(eve::convert(vz_int, eve::as<f64>()), z_scale, z_offset);

        if (has_class_filter || has_coord_filter || do_count)
        {
            u8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            if (has_class_filter || do_count)
            {
                c0 = p[classification_offset] & classification_byte_mask;
                c1 = p[stride + classification_offset] & classification_byte_mask;
                c2 = p[stride * 2 + classification_offset] & classification_byte_mask;
                c3 = p[stride * 3 + classification_offset] & classification_byte_mask;
            }

            wf64 blend_val {lane_pass};
            if (has_class_filter && has_coord_filter)
            {
                wf64 class_blend {blend_mask[c0], blend_mask[c1], blend_mask[c2], blend_mask[c3]};
                wl64 coord_blend =
                    (vx >= vxmin) && (vx <= vxmax) && (vy >= vymin) && (vy <= vymax) && (vz >= vzmin) && (vz <= vzmax);
                blend_val = eve::if_else(coord_blend, class_blend, zero_pd);
            }
            else if (has_class_filter)
            {
                blend_val = wf64 {blend_mask[c0], blend_mask[c1], blend_mask[c2], blend_mask[c3]};
            }
            else if (has_coord_filter)
            {
                wl64 coord_blend =
                    (vx >= vxmin) && (vx <= vxmax) && (vy >= vymin) && (vy <= vymax) && (vz >= vzmin) && (vz <= vzmax);
                blend_val = eve::if_else(coord_blend, wf64(lane_pass), zero_pd);
            }

            if (has_class_filter || has_coord_filter)
            {
                wl64 blend = eve::bit_cast(blend_val, eve::as<wl64>());
                u32 mask = eve::top_bits(blend).as_int();

                if (has_decimation)
                {
                    if (mask != 0)
                    {
                        u32 num_passed = popcount(mask);
                        if (current_offset != 0 && current_offset + num_passed <= keep_every)
                        {
                            current_offset += num_passed;
                            if (current_offset == keep_every) current_offset = 0;
                            mask = 0;
                            blend = wl64 {false};
                        }
                        else
                        {
                            u32 new_mask = 0;
                            for (i32 b = 0; b < 4; ++b)
                            {
                                if (mask & (1 << b))
                                {
                                    if (current_offset == 0) new_mask |= (1 << b);
                                    current_offset++;
                                    if (current_offset == keep_every) current_offset = 0;
                                }
                            }
                            if (new_mask != mask)
                            {
                                mask = new_mask;
                                blend = eve::bit_cast(wf64(&decimation_mask_lut_data[mask][0]), eve::as<wl64>());
                            }
                        }
                    }
                }

                u32 pop = popcount(mask);
                passed_count += pop;
                dropped_count += 4 - pop;

                if (write_kept)
                {
                    if (mask == 15)
                        kept_writer->write(p, stride * 4);
                    else
                    {
                        if (mask & 1) kept_writer->write(p, stride);
                        if (mask & 2) kept_writer->write(p + stride, stride);
                        if (mask & 4) kept_writer->write(p + stride * 2, stride);
                        if (mask & 8) kept_writer->write(p + stride * 3, stride);
                    }
                }

                if (write_dropped)
                {
                    if (mask == 0)
                        dropped_writer->write(p, stride * 4);
                    else
                    {
                        if (!(mask & 1)) dropped_writer->write(p, stride);
                        if (!(mask & 2)) dropped_writer->write(p + stride, stride);
                        if (!(mask & 4)) dropped_writer->write(p + stride * 2, stride);
                        if (!(mask & 8)) dropped_writer->write(p + stride * 3, stride);
                    }
                }

                if (do_elev)
                {
                    wf64 vz_filtered = eve::if_else(blend, vz, zero_pd);
                    sum_z_vec = sum_z_vec + vz_filtered;
                    sum_z2_vec = eve::fma(vz_filtered, vz_filtered, sum_z2_vec);
                }

                if (do_bbox || write_kept)
                {
                    min_x = eve::min(min_x, eve::if_else(blend, vx, pos_inf));
                    max_x = eve::max(max_x, eve::if_else(blend, vx, neg_inf));
                    min_y = eve::min(min_y, eve::if_else(blend, vy, pos_inf));
                    max_y = eve::max(max_y, eve::if_else(blend, vy, neg_inf));
                    min_z = eve::min(min_z, eve::if_else(blend, vz, pos_inf));
                    max_z = eve::max(max_z, eve::if_else(blend, vz, neg_inf));
                }

                if (write_dropped)
                {
                    wl64 dropped_blend = !blend;
                    dropped_min_x = eve::min(dropped_min_x, eve::if_else(dropped_blend, vx, pos_inf));
                    dropped_max_x = eve::max(dropped_max_x, eve::if_else(dropped_blend, vx, neg_inf));
                    dropped_min_y = eve::min(dropped_min_y, eve::if_else(dropped_blend, vy, pos_inf));
                    dropped_max_y = eve::max(dropped_max_y, eve::if_else(dropped_blend, vy, neg_inf));
                    dropped_min_z = eve::min(dropped_min_z, eve::if_else(dropped_blend, vz, pos_inf));
                    dropped_max_z = eve::max(dropped_max_z, eve::if_else(dropped_blend, vz, neg_inf));
                }

                if (do_count)
                {
                    if (mask & 1) out_class_counts[c0]++;
                    if (mask & 2) out_class_counts[c1]++;
                    if (mask & 4) out_class_counts[c2]++;
                    if (mask & 8) out_class_counts[c3]++;
                }
            }
            else // Only DoCount is true
            {
                passed_count += 4;
                if (write_kept) kept_writer->write(p, stride * 4);

                if (do_elev)
                {
                    sum_z_vec = sum_z_vec + vz;
                    sum_z2_vec = eve::fma(vz, vz, sum_z2_vec);
                }
                if (do_bbox || write_kept)
                {
                    min_x = eve::min(min_x, vx);
                    max_x = eve::max(max_x, vx);
                    min_y = eve::min(min_y, vy);
                    max_y = eve::max(max_y, vy);
                    min_z = eve::min(min_z, vz);
                    max_z = eve::max(max_z, vz);
                }
                out_class_counts[c0]++;
                out_class_counts[c1]++;
                out_class_counts[c2]++;
                out_class_counts[c3]++;
            }
        }
        else
        {
            passed_count += 4;
            if (write_kept) kept_writer->write(p, stride * 4);

            if (do_elev)
            {
                sum_z_vec = sum_z_vec + vz;
                sum_z2_vec = eve::fma(vz, vz, sum_z2_vec);
            }

            if (do_bbox || write_kept)
            {
                min_x = eve::min(min_x, vx);
                max_x = eve::max(max_x, vx);
                min_y = eve::min(min_y, vy);
                max_y = eve::max(max_y, vy);
                min_z = eve::min(min_z, vz);
                max_z = eve::max(max_z, vz);
            }
        }

        p += stride * 4;
    }

    // Horizontal reduction of SIMD accumulators.
    if (do_elev)
    {
        result.sum_z = eve::reduce(sum_z_vec, eve::add);
        result.sum_z2 = eve::reduce(sum_z2_vec, eve::add);
    }

    if (do_bbox || write_kept)
    {
        result.bbox.min_x = eve::reduce(min_x, eve::min);
        result.bbox.max_x = eve::reduce(max_x, eve::max);
        result.bbox.min_y = eve::reduce(min_y, eve::min);
        result.bbox.max_y = eve::reduce(max_y, eve::max);
        result.bbox.min_z = eve::reduce(min_z, eve::min);
        result.bbox.max_z = eve::reduce(max_z, eve::max);
    }

    if (write_dropped)
    {
        result.dropped_bbox.min_x = eve::reduce(dropped_min_x, eve::min);
        result.dropped_bbox.max_x = eve::reduce(dropped_max_x, eve::max);
        result.dropped_bbox.min_y = eve::reduce(dropped_min_y, eve::min);
        result.dropped_bbox.max_y = eve::reduce(dropped_max_y, eve::max);
        result.dropped_bbox.min_z = eve::reduce(dropped_min_z, eve::min);
        result.dropped_bbox.max_z = eve::reduce(dropped_max_z, eve::max);
    }

    // Process the remaining points using scalar math.
    for (; i < view.point_count; ++i)
    {
        bool passed = true;
        u8 c = 0;

        if (has_class_filter || do_count) c = p[classification_offset] & classification_byte_mask;

        const auto* pt = reinterpret_cast<const LasPointCoordinates*>(p);
        const f64 x = (pt->x * view.header->x_scale_factor) + view.header->x_offset;
        const f64 y = (pt->y * view.header->y_scale_factor) + view.header->y_offset;
        const f64 z = (pt->z * view.header->z_scale_factor) + view.header->z_offset;

        if (has_class_filter)
        {
            if (!filter_mask[c]) passed = false;
        }

        if (has_coord_filter)
        {
            if (x < filter_xmin || x > filter_xmax || y < filter_ymin || y > filter_ymax || z < filter_zmin ||
                z > filter_zmax)
            {
                passed = false;
            }
        }

        if (passed)
        {
            if (has_decimation)
            {
                if (current_offset != 0)
                {
                    current_offset++;
                    if (current_offset == keep_every) current_offset = 0;
                    passed = false;
                }
                else
                {
                    current_offset++;
                    if (current_offset == keep_every) current_offset = 0;
                }
            }
        }

        if (passed)
        {
            passed_count++;
            if (write_kept) kept_writer->write(p, stride);

            if (do_count) out_class_counts[c]++;
            if (do_elev)
            {
                result.sum_z += z;
                result.sum_z2 += z * z;
            }
            if (do_bbox || write_kept)
            {
                result.bbox.min_x = std::min(result.bbox.min_x, x);
                result.bbox.max_x = std::max(result.bbox.max_x, x);
                result.bbox.min_y = std::min(result.bbox.min_y, y);
                result.bbox.max_y = std::max(result.bbox.max_y, y);
                result.bbox.min_z = std::min(result.bbox.min_z, z);
                result.bbox.max_z = std::max(result.bbox.max_z, z);
            }
        }
        else
        {
            dropped_count++;
            if (write_dropped)
            {
                dropped_writer->write(p, stride);
                result.dropped_bbox.min_x = std::min(result.dropped_bbox.min_x, x);
                result.dropped_bbox.max_x = std::max(result.dropped_bbox.max_x, x);
                result.dropped_bbox.min_y = std::min(result.dropped_bbox.min_y, y);
                result.dropped_bbox.max_y = std::max(result.dropped_bbox.max_y, y);
                result.dropped_bbox.min_z = std::min(result.dropped_bbox.min_z, z);
                result.dropped_bbox.max_z = std::max(result.dropped_bbox.max_z, z);
            }
        }

        p += stride;
    }

    result.points_processed = passed_count;
    result.points_dropped = dropped_count;
    if (write_kept) result.io_time_seconds += kept_writer->get_io_seconds();
    if (write_dropped) result.io_time_seconds += dropped_writer->get_io_seconds();
    return result;
}

inline void build_elev_histogram_simd(
    bool has_class_filter, bool has_coord_filter, bool has_decimation, const u8* file_data, const HeaderView& view,
    const std::array<u8, 256>& filter_mask, const std::array<f64, 256>& blend_mask, u32 classification_offset,
    u8 classification_byte_mask, f64 filter_xmin, f64 filter_xmax, f64 filter_ymin, f64 filter_ymax, f64 filter_zmin,
    f64 filter_zmax, u64 keep_every, f64 hist_min, f64 hist_max, f64 bin_step, i32 nb_bins,
    std::vector<u64>& out_z_bins, u64& out_underflow, u64& out_overflow
)
{
    using wf64 = eve::wide<f64, eve::fixed<4>>;
    using wi32 = eve::wide<i32, eve::fixed<4>>;
    using wl64 = eve::logical<wf64>;

    u64 current_offset = 0;
    const wf64 x_scale {view.header->x_scale_factor};
    const wf64 y_scale {view.header->y_scale_factor};
    const wf64 z_scale {view.header->z_scale_factor};
    const wf64 x_offset {view.header->x_offset};
    const wf64 y_offset {view.header->y_offset};
    const wf64 z_offset {view.header->z_offset};

    const wf64 vxmin {filter_xmin};
    const wf64 vxmax {filter_xmax};
    const wf64 vymin {filter_ymin};
    const wf64 vymax {filter_ymax};
    const wf64 vzmin {filter_zmin};
    const wf64 vzmax {filter_zmax};

    const u8* p = file_data + view.point_data_offset;
    const u32 stride = view.point_record_length;

    u64 i = 0;
    const u64 batch_limit = view.point_count - (view.point_count % 4);

    for (; i < batch_limit; i += 4)
    {
        const auto* pt0 = reinterpret_cast<const LasPointCoordinates*>(p);
        const auto* pt1 = reinterpret_cast<const LasPointCoordinates*>(p + stride);
        const auto* pt2 = reinterpret_cast<const LasPointCoordinates*>(p + stride * 2);
        const auto* pt3 = reinterpret_cast<const LasPointCoordinates*>(p + stride * 3);

        wi32 vz_int {pt0->z, pt1->z, pt2->z, pt3->z};
        wf64 vz = eve::fma(eve::convert(vz_int, eve::as<f64>()), z_scale, z_offset);

        u32 mask = 15;

        if (has_class_filter || has_coord_filter)
        {
            wi32 vx_int {pt0->x, pt1->x, pt2->x, pt3->x};
            wi32 vy_int {pt0->y, pt1->y, pt2->y, pt3->y};
            wf64 vx = eve::fma(eve::convert(vx_int, eve::as<f64>()), x_scale, x_offset);
            wf64 vy = eve::fma(eve::convert(vy_int, eve::as<f64>()), y_scale, y_offset);

            u8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            if (has_class_filter)
            {
                c0 = p[classification_offset] & classification_byte_mask;
                c1 = p[stride + classification_offset] & classification_byte_mask;
                c2 = p[stride * 2 + classification_offset] & classification_byte_mask;
                c3 = p[stride * 3 + classification_offset] & classification_byte_mask;
            }

            wl64 blend;
            if (has_class_filter && has_coord_filter)
            {
                wf64 class_blend {blend_mask[c0], blend_mask[c1], blend_mask[c2], blend_mask[c3]};
                wl64 coord_blend =
                    (vx >= vxmin) && (vx <= vxmax) && (vy >= vymin) && (vy <= vymax) && (vz >= vzmin) && (vz <= vzmax);
                blend = coord_blend && eve::bit_cast(class_blend, eve::as<wl64>());
            }
            else if (has_class_filter)
            {
                wf64 class_blend {blend_mask[c0], blend_mask[c1], blend_mask[c2], blend_mask[c3]};
                blend = eve::bit_cast(class_blend, eve::as<wl64>());
            }
            else if (has_coord_filter)
            {
                blend =
                    (vx >= vxmin) && (vx <= vxmax) && (vy >= vymin) && (vy <= vymax) && (vz >= vzmin) && (vz <= vzmax);
            }

            mask = eve::top_bits(blend).as_int();

            if (has_decimation)
            {
                if (mask != 0)
                {
                    u32 num_passed = popcount(mask);
                    if (current_offset != 0 && current_offset + num_passed <= keep_every)
                    {
                        current_offset += num_passed;
                        if (current_offset == keep_every) current_offset = 0;
                        mask = 0;
                    }
                    else
                    {
                        u32 new_mask = 0;
                        for (i32 b = 0; b < 4; ++b)
                        {
                            if (mask & (1 << b))
                            {
                                if (current_offset == 0) new_mask |= (1 << b);
                                current_offset++;
                                if (current_offset == keep_every) current_offset = 0;
                            }
                        }
                        mask = new_mask;
                    }
                }
            }
        }

        if (mask)
        {
            alignas(32) f64 z_vals[4];
            eve::store(vz, z_vals);

            auto bin_val = [&](f64 z) {
                if (z < hist_min)
                    out_underflow++;
                else if (z >= hist_max)
                    out_overflow++;
                else
                    out_z_bins[as<usize>(std::min(as<i32>((z - hist_min) / bin_step), nb_bins - 1))]++;
            };

            if (mask & 1) bin_val(z_vals[0]);
            if (mask & 2) bin_val(z_vals[1]);
            if (mask & 4) bin_val(z_vals[2]);
            if (mask & 8) bin_val(z_vals[3]);
        }

        p += stride * 4;
    }

    for (; i < view.point_count; ++i)
    {
        bool passed = true;
        u8 c = 0;

        if (has_class_filter) c = p[classification_offset] & classification_byte_mask;

        const auto* pt = reinterpret_cast<const LasPointCoordinates*>(p);
        const f64 x = (pt->x * view.header->x_scale_factor) + view.header->x_offset;
        const f64 y = (pt->y * view.header->y_scale_factor) + view.header->y_offset;
        const f64 z = (pt->z * view.header->z_scale_factor) + view.header->z_offset;

        if (has_class_filter)
            if (!filter_mask[c]) passed = false;
        if (has_coord_filter)
        {
            if (x < filter_xmin || x > filter_xmax || y < filter_ymin || y > filter_ymax || z < filter_zmin ||
                z > filter_zmax)
                passed = false;
        }

        if (passed)
        {
            if (z < hist_min)
                out_underflow++;
            else if (z >= hist_max)
                out_overflow++;
            else
                out_z_bins[as<usize>(std::min(as<i32>((z - hist_min) / bin_step), nb_bins - 1))]++;
        }

        p += stride;
    }
}

} // namespace laspar
