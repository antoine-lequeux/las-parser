#pragma once
#include "file_writer.hpp"
#include "header_view.hpp"
#include "las_data.hpp"
#include <array>
#include <bit>
#include <immintrin.h>
#include <vector>

namespace laspar
{
// Get the smallest double out of 4.
inline f64 hmin_pd(__m256d v)
{
    __m256d perm = _mm256_permute2f128_pd(v, v, 1);
    __m256d min1 = _mm256_min_pd(v, perm);
    __m256d perm2 = _mm256_permute_pd(min1, 5);
    __m256d min2 = _mm256_min_pd(min1, perm2);
    return _mm256_cvtsd_f64(min2);
}

// Get the largest double out of 4.
inline f64 hmax_pd(__m256d v)
{
    __m256d perm = _mm256_permute2f128_pd(v, v, 1);
    __m256d max1 = _mm256_max_pd(v, perm);
    __m256d perm2 = _mm256_permute_pd(max1, 5);
    __m256d max2 = _mm256_max_pd(max1, perm2);
    return _mm256_cvtsd_f64(max2);
}

// Get the sum of all 4 doubles.
inline f64 hsum_pd(__m256d v)
{
    __m128d lo = _mm256_extractf128_pd(v, 0);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d sum1 = _mm_add_pd(lo, hi);
    __m128d sum2 = _mm_add_pd(sum1, _mm_unpackhi_pd(sum1, sum1));
    return _mm_cvtsd_f64(sum2);
}

constexpr f64 lane_pass = std::bit_cast<f64>(~u64 {0});

alignas(32) const f64 decimation_mask_lut_data[16][4] = {
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

// AVX2 point processing function to get all data required in the command line.
template <
    bool HasClassFilter, bool HasCoordFilter, bool DoCount, bool DoBBox, bool DoElev, bool WriteKept, bool WriteDropped,
    bool HasDecimation>
inline ProcessResult process_points_avx2(
    const u8* file_data, const HeaderView& view, const std::array<u8, 256>& filter_mask,
    const std::array<f64, 256>& blend_mask, u32 classification_offset, u8 classification_byte_mask,
    std::vector<u64>& out_class_counts, f64 filter_xmin, f64 filter_xmax, f64 filter_ymin, f64 filter_ymax,
    f64 filter_zmin, f64 filter_zmax, u64 keep_every, BufferedFileWriter* kept_writer,
    BufferedFileWriter* dropped_writer
)
{
    ProcessResult result {};
    u64 current_offset = 0;

    // Store the scale values into 256-bit registers holding 4 doubles each.
    const __m256d x_scale = _mm256_set1_pd(view.header->x_scale_factor);
    const __m256d y_scale = _mm256_set1_pd(view.header->y_scale_factor);
    const __m256d z_scale = _mm256_set1_pd(view.header->z_scale_factor);

    // Same for offset values.
    const __m256d x_offset = _mm256_set1_pd(view.header->x_offset);
    const __m256d y_offset = _mm256_set1_pd(view.header->y_offset);
    const __m256d z_offset = _mm256_set1_pd(view.header->z_offset);

    __m256d min_x, max_x, min_y, max_y, min_z, max_z;
    __m256d dropped_min_x, dropped_max_x, dropped_min_y, dropped_max_y, dropped_min_z, dropped_max_z;
    const __m256d pos_inf = _mm256_set1_pd(std::numeric_limits<f64>::max());
    const __m256d neg_inf = _mm256_set1_pd(std::numeric_limits<f64>::lowest());

    if constexpr (DoBBox || WriteKept)
    {
        min_x = pos_inf;
        max_x = neg_inf;
        min_y = pos_inf;
        max_y = neg_inf;
        min_z = pos_inf;
        max_z = neg_inf;
    }
    if constexpr (WriteDropped)
    {
        dropped_min_x = pos_inf;
        dropped_max_x = neg_inf;
        dropped_min_y = pos_inf;
        dropped_max_y = neg_inf;
        dropped_min_z = pos_inf;
        dropped_max_z = neg_inf;
    }

    __m256d sum_z_vec, sum_z2_vec, zero_pd;
    if constexpr (DoElev)
    {
        sum_z_vec = _mm256_setzero_pd();
        sum_z2_vec = _mm256_setzero_pd();
        zero_pd = _mm256_setzero_pd();
    }

    const u8* p = file_data + view.point_data_offset;
    const u32 stride = view.point_record_length;

    const __m256d vxmin = _mm256_set1_pd(filter_xmin);
    const __m256d vxmax = _mm256_set1_pd(filter_xmax);
    const __m256d vymin = _mm256_set1_pd(filter_ymin);
    const __m256d vymax = _mm256_set1_pd(filter_ymax);
    const __m256d vzmin = _mm256_set1_pd(filter_zmin);
    const __m256d vzmax = _mm256_set1_pd(filter_zmax);

    u64 i = 0;
    const u64 batch_limit = view.point_count - (view.point_count % 4);
    u64 passed_count = 0;
    u64 dropped_count = 0;

    for (; i < batch_limit; i += 4)
    {
        __m128i pt0_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        __m128i pt1_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + stride));
        __m128i pt2_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + stride * 2));
        __m128i pt3_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + stride * 3));

        __m128i tmp0 = _mm_unpacklo_epi32(pt0_vec, pt1_vec);
        __m128i tmp1 = _mm_unpackhi_epi32(pt0_vec, pt1_vec);
        __m128i tmp2 = _mm_unpacklo_epi32(pt2_vec, pt3_vec);
        __m128i tmp3 = _mm_unpackhi_epi32(pt2_vec, pt3_vec);

        __m128i vx_int = _mm_unpacklo_epi64(tmp0, tmp2);
        __m128i vy_int = _mm_unpackhi_epi64(tmp0, tmp2);
        __m128i vz_int = _mm_unpacklo_epi64(tmp1, tmp3);

        __m256d vx = _mm256_fmadd_pd(_mm256_cvtepi32_pd(vx_int), x_scale, x_offset);
        __m256d vy = _mm256_fmadd_pd(_mm256_cvtepi32_pd(vy_int), y_scale, y_offset);
        __m256d vz = _mm256_fmadd_pd(_mm256_cvtepi32_pd(vz_int), z_scale, z_offset);

        if constexpr (HasClassFilter || HasCoordFilter || DoCount)
        {
            u8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            if constexpr (HasClassFilter || DoCount)
            {
                c0 = p[classification_offset] & classification_byte_mask;
                c1 = p[stride + classification_offset] & classification_byte_mask;
                c2 = p[stride * 2 + classification_offset] & classification_byte_mask;
                c3 = p[stride * 3 + classification_offset] & classification_byte_mask;
            }

            __m256d blend;
            if constexpr (HasClassFilter && HasCoordFilter)
            {
                __m256d class_blend = _mm256_set_pd(blend_mask[c3], blend_mask[c2], blend_mask[c1], blend_mask[c0]);

                __m256d m_xmin = _mm256_cmp_pd(vx, vxmin, _CMP_GE_OQ);
                __m256d m_xmax = _mm256_cmp_pd(vx, vxmax, _CMP_LE_OQ);
                __m256d m_ymin = _mm256_cmp_pd(vy, vymin, _CMP_GE_OQ);
                __m256d m_ymax = _mm256_cmp_pd(vy, vymax, _CMP_LE_OQ);
                __m256d m_zmin = _mm256_cmp_pd(vz, vzmin, _CMP_GE_OQ);
                __m256d m_zmax = _mm256_cmp_pd(vz, vzmax, _CMP_LE_OQ);

                __m256d m_x = _mm256_and_pd(m_xmin, m_xmax);
                __m256d m_y = _mm256_and_pd(m_ymin, m_ymax);
                __m256d m_z = _mm256_and_pd(m_zmin, m_zmax);
                __m256d coord_blend = _mm256_and_pd(m_x, _mm256_and_pd(m_y, m_z));

                blend = _mm256_and_pd(class_blend, coord_blend);
            }
            else if constexpr (HasClassFilter)
            {
                blend = _mm256_set_pd(blend_mask[c3], blend_mask[c2], blend_mask[c1], blend_mask[c0]);
            }
            else if constexpr (HasCoordFilter)
            {
                __m256d m_xmin = _mm256_cmp_pd(vx, vxmin, _CMP_GE_OQ);
                __m256d m_xmax = _mm256_cmp_pd(vx, vxmax, _CMP_LE_OQ);
                __m256d m_ymin = _mm256_cmp_pd(vy, vymin, _CMP_GE_OQ);
                __m256d m_ymax = _mm256_cmp_pd(vy, vymax, _CMP_LE_OQ);
                __m256d m_zmin = _mm256_cmp_pd(vz, vzmin, _CMP_GE_OQ);
                __m256d m_zmax = _mm256_cmp_pd(vz, vzmax, _CMP_LE_OQ);

                __m256d m_x = _mm256_and_pd(m_xmin, m_xmax);
                __m256d m_y = _mm256_and_pd(m_ymin, m_ymax);
                __m256d m_z = _mm256_and_pd(m_zmin, m_zmax);
                blend = _mm256_and_pd(m_x, _mm256_and_pd(m_y, m_z));
            }

            if constexpr (HasClassFilter || HasCoordFilter)
            {
                i32 mask = _mm256_movemask_pd(blend);

                if constexpr (HasDecimation)
                {
                    if (mask != 0)
                    {
                        u32 num_passed = std::popcount(as<u32>(mask));
                        if (current_offset != 0 && current_offset + num_passed <= keep_every)
                        {
                            current_offset += num_passed;
                            if (current_offset == keep_every) current_offset = 0;
                            mask = 0;
                            blend = _mm256_setzero_pd();
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
                            if (new_mask != as<u32>(mask))
                            {
                                mask = new_mask;
                                blend = _mm256_load_pd(decimation_mask_lut_data[mask]);
                            }
                        }
                    }
                }

                u32 pop = std::popcount(as<u32>(mask));
                passed_count += pop;
                dropped_count += 4 - pop;

                if constexpr (WriteKept)
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

                if constexpr (WriteDropped)
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

                if constexpr (DoElev)
                {
                    __m256d vz_filtered = _mm256_blendv_pd(zero_pd, vz, blend);
                    sum_z_vec = _mm256_add_pd(sum_z_vec, vz_filtered);
                    sum_z2_vec = _mm256_fmadd_pd(vz_filtered, vz_filtered, sum_z2_vec);
                }

                if constexpr (DoBBox || WriteKept)
                {
                    min_x = _mm256_min_pd(min_x, _mm256_blendv_pd(pos_inf, vx, blend));
                    max_x = _mm256_max_pd(max_x, _mm256_blendv_pd(neg_inf, vx, blend));
                    min_y = _mm256_min_pd(min_y, _mm256_blendv_pd(pos_inf, vy, blend));
                    max_y = _mm256_max_pd(max_y, _mm256_blendv_pd(neg_inf, vy, blend));
                    min_z = _mm256_min_pd(min_z, _mm256_blendv_pd(pos_inf, vz, blend));
                    max_z = _mm256_max_pd(max_z, _mm256_blendv_pd(neg_inf, vz, blend));
                }

                if constexpr (WriteDropped)
                {
                    __m256d dropped_blend = _mm256_xor_pd(blend, _mm256_castsi256_pd(_mm256_set1_epi32(-1)));
                    dropped_min_x = _mm256_min_pd(dropped_min_x, _mm256_blendv_pd(pos_inf, vx, dropped_blend));
                    dropped_max_x = _mm256_max_pd(dropped_max_x, _mm256_blendv_pd(neg_inf, vx, dropped_blend));
                    dropped_min_y = _mm256_min_pd(dropped_min_y, _mm256_blendv_pd(pos_inf, vy, dropped_blend));
                    dropped_max_y = _mm256_max_pd(dropped_max_y, _mm256_blendv_pd(neg_inf, vy, dropped_blend));
                    dropped_min_z = _mm256_min_pd(dropped_min_z, _mm256_blendv_pd(pos_inf, vz, dropped_blend));
                    dropped_max_z = _mm256_max_pd(dropped_max_z, _mm256_blendv_pd(neg_inf, vz, dropped_blend));
                }

                if constexpr (DoCount)
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
                if constexpr (WriteKept) kept_writer->write(p, stride * 4);

                if constexpr (DoElev)
                {
                    sum_z_vec = _mm256_add_pd(sum_z_vec, vz);
                    sum_z2_vec = _mm256_fmadd_pd(vz, vz, sum_z2_vec);
                }
                if constexpr (DoBBox || WriteKept)
                {
                    min_x = _mm256_min_pd(min_x, vx);
                    max_x = _mm256_max_pd(max_x, vx);
                    min_y = _mm256_min_pd(min_y, vy);
                    max_y = _mm256_max_pd(max_y, vy);
                    min_z = _mm256_min_pd(min_z, vz);
                    max_z = _mm256_max_pd(max_z, vz);
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
            if constexpr (WriteKept) kept_writer->write(p, stride * 4);

            if constexpr (DoElev)
            {
                sum_z_vec = _mm256_add_pd(sum_z_vec, vz);
                sum_z2_vec = _mm256_fmadd_pd(vz, vz, sum_z2_vec);
            }

            if constexpr (DoBBox || WriteKept)
            {
                min_x = _mm256_min_pd(min_x, vx);
                max_x = _mm256_max_pd(max_x, vx);
                min_y = _mm256_min_pd(min_y, vy);
                max_y = _mm256_max_pd(max_y, vy);
                min_z = _mm256_min_pd(min_z, vz);
                max_z = _mm256_max_pd(max_z, vz);
            }
        }

        p += stride * 4;
    }

    // Horizontal reduction of SIMD accumulators.
    if constexpr (DoElev)
    {
        result.sum_z = hsum_pd(sum_z_vec);
        result.sum_z2 = hsum_pd(sum_z2_vec);
    }

    if constexpr (DoBBox || WriteKept)
    {
        result.bbox.min_x = hmin_pd(min_x);
        result.bbox.max_x = hmax_pd(max_x);
        result.bbox.min_y = hmin_pd(min_y);
        result.bbox.max_y = hmax_pd(max_y);
        result.bbox.min_z = hmin_pd(min_z);
        result.bbox.max_z = hmax_pd(max_z);
    }

    if constexpr (WriteDropped)
    {
        result.dropped_bbox.min_x = hmin_pd(dropped_min_x);
        result.dropped_bbox.max_x = hmax_pd(dropped_max_x);
        result.dropped_bbox.min_y = hmin_pd(dropped_min_y);
        result.dropped_bbox.max_y = hmax_pd(dropped_max_y);
        result.dropped_bbox.min_z = hmin_pd(dropped_min_z);
        result.dropped_bbox.max_z = hmax_pd(dropped_max_z);
    }

    // Process the remaining points using scalar math.
    for (; i < view.point_count; ++i)
    {
        bool passed = true;
        u8 c = 0;

        if constexpr (HasClassFilter || DoCount) c = p[classification_offset] & classification_byte_mask;

        const auto* pt = reinterpret_cast<const LasPointCoordinates*>(p);
        const f64 x = (pt->x * view.header->x_scale_factor) + view.header->x_offset;
        const f64 y = (pt->y * view.header->y_scale_factor) + view.header->y_offset;
        const f64 z = (pt->z * view.header->z_scale_factor) + view.header->z_offset;

        if constexpr (HasClassFilter)
        {
            if (!filter_mask[c]) passed = false;
        }

        if constexpr (HasCoordFilter)
        {
            if (x < filter_xmin || x > filter_xmax || y < filter_ymin || y > filter_ymax || z < filter_zmin ||
                z > filter_zmax)
            {
                passed = false;
            }
        }

        if (passed)
        {
            if constexpr (HasDecimation)
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
            if constexpr (WriteKept) kept_writer->write(p, stride);

            if constexpr (DoCount) out_class_counts[c]++;
            if constexpr (DoElev)
            {
                result.sum_z += z;
                result.sum_z2 += z * z;
            }
            if constexpr (DoBBox || WriteKept)
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
            if constexpr (WriteDropped)
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
    if constexpr (WriteKept) result.io_time_seconds += kept_writer->get_io_seconds();
    if constexpr (WriteDropped) result.io_time_seconds += dropped_writer->get_io_seconds();
    return result;
}

template <bool... Bools, typename Fn, typename... Rest>
decltype(auto) dispatch_bools(Fn&& fn, bool current, Rest... rest)
{
    if (current)
        if constexpr (sizeof...(rest) == 0)
            return fn.template operator()<Bools..., true>();
        else
            return dispatch_bools<Bools..., true>(std::forward<Fn>(fn), rest...);
    else if constexpr (sizeof...(rest) == 0)
        return fn.template operator()<Bools..., false>();
    else
        return dispatch_bools<Bools..., false>(std::forward<Fn>(fn), rest...);
}

inline ProcessResult dispatch_avx2(
    bool has_class_filter, bool has_coord_filter, bool do_count, bool do_bbox, bool do_elev, bool write_kept,
    bool write_dropped, bool has_decimation, const u8* file_data, const HeaderView& view,
    const std::array<u8, 256>& filter_mask, const std::array<f64, 256>& blend_mask, u32 classification_offset,
    u8 classification_byte_mask, std::vector<u64>& out_class_counts, f64 filter_xmin, f64 filter_xmax, f64 filter_ymin,
    f64 filter_ymax, f64 filter_zmin, f64 filter_zmax, u64 keep_every, BufferedFileWriter* kept_writer,
    BufferedFileWriter* dropped_writer
)
{
    auto runner = [&]<bool HCF, bool HCoF, bool DC, bool DB, bool DE, bool WK, bool WD, bool HD>() {
        return process_points_avx2<HCF, HCoF, DC, DB, DE, WK, WD, HD>(
            file_data, view, filter_mask, blend_mask, classification_offset, classification_byte_mask, out_class_counts,
            filter_xmin, filter_xmax, filter_ymin, filter_ymax, filter_zmin, filter_zmax, keep_every, kept_writer,
            dropped_writer
        );
    };

    return dispatch_bools(
        runner, has_class_filter, has_coord_filter, do_count, do_bbox, do_elev, write_kept, write_dropped,
        has_decimation
    );
}

template <bool HasClassFilter, bool HasCoordFilter, bool HasDecimation>
inline void build_elev_histogram_avx2(
    const u8* file_data, const HeaderView& view, const std::array<u8, 256>& filter_mask,
    const std::array<f64, 256>& blend_mask, u32 classification_offset, u8 classification_byte_mask, f64 filter_xmin,
    f64 filter_xmax, f64 filter_ymin, f64 filter_ymax, f64 filter_zmin, f64 filter_zmax, u64 keep_every, f64 hist_min,
    f64 hist_max, f64 bin_step, i32 nb_bins, std::vector<u64>& out_z_bins, u64& out_underflow, u64& out_overflow
)
{
    u64 current_offset = 0;
    const __m256d x_scale = _mm256_set1_pd(view.header->x_scale_factor);
    const __m256d y_scale = _mm256_set1_pd(view.header->y_scale_factor);
    const __m256d z_scale = _mm256_set1_pd(view.header->z_scale_factor);
    const __m256d x_offset = _mm256_set1_pd(view.header->x_offset);
    const __m256d y_offset = _mm256_set1_pd(view.header->y_offset);
    const __m256d z_offset = _mm256_set1_pd(view.header->z_offset);

    const __m256d vxmin = _mm256_set1_pd(filter_xmin);
    const __m256d vxmax = _mm256_set1_pd(filter_xmax);
    const __m256d vymin = _mm256_set1_pd(filter_ymin);
    const __m256d vymax = _mm256_set1_pd(filter_ymax);
    const __m256d vzmin = _mm256_set1_pd(filter_zmin);
    const __m256d vzmax = _mm256_set1_pd(filter_zmax);

    const u8* p = file_data + view.point_data_offset;
    const u32 stride = view.point_record_length;

    u64 i = 0;
    const u64 batch_limit = view.point_count - (view.point_count % 4);

    for (; i < batch_limit; i += 4)
    {
        __m128i pt0_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        __m128i pt1_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + stride));
        __m128i pt2_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + stride * 2));
        __m128i pt3_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + stride * 3));

        __m128i tmp0 = _mm_unpacklo_epi32(pt0_vec, pt1_vec);
        __m128i tmp1 = _mm_unpackhi_epi32(pt0_vec, pt1_vec);
        __m128i tmp2 = _mm_unpacklo_epi32(pt2_vec, pt3_vec);
        __m128i tmp3 = _mm_unpackhi_epi32(pt2_vec, pt3_vec);

        __m128i vz_int = _mm_unpacklo_epi64(tmp1, tmp3);
        __m256d vz = _mm256_fmadd_pd(_mm256_cvtepi32_pd(vz_int), z_scale, z_offset);

        i32 mask = 0xF;

        if constexpr (HasClassFilter || HasCoordFilter)
        {
            __m128i vx_int = _mm_unpacklo_epi64(tmp0, tmp2);
            __m128i vy_int = _mm_unpackhi_epi64(tmp0, tmp2);
            __m256d vx = _mm256_fmadd_pd(_mm256_cvtepi32_pd(vx_int), x_scale, x_offset);
            __m256d vy = _mm256_fmadd_pd(_mm256_cvtepi32_pd(vy_int), y_scale, y_offset);

            u8 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            if constexpr (HasClassFilter)
            {
                c0 = p[classification_offset] & classification_byte_mask;
                c1 = p[stride + classification_offset] & classification_byte_mask;
                c2 = p[stride * 2 + classification_offset] & classification_byte_mask;
                c3 = p[stride * 3 + classification_offset] & classification_byte_mask;
            }

            __m256d blend;
            if constexpr (HasClassFilter && HasCoordFilter)
            {
                __m256d class_blend = _mm256_set_pd(blend_mask[c3], blend_mask[c2], blend_mask[c1], blend_mask[c0]);
                __m256d m_xmin = _mm256_cmp_pd(vx, vxmin, _CMP_GE_OQ);
                __m256d m_xmax = _mm256_cmp_pd(vx, vxmax, _CMP_LE_OQ);
                __m256d m_ymin = _mm256_cmp_pd(vy, vymin, _CMP_GE_OQ);
                __m256d m_ymax = _mm256_cmp_pd(vy, vymax, _CMP_LE_OQ);
                __m256d m_zmin = _mm256_cmp_pd(vz, vzmin, _CMP_GE_OQ);
                __m256d m_zmax = _mm256_cmp_pd(vz, vzmax, _CMP_LE_OQ);

                __m256d m_x = _mm256_and_pd(m_xmin, m_xmax);
                __m256d m_y = _mm256_and_pd(m_ymin, m_ymax);
                __m256d m_z = _mm256_and_pd(m_zmin, m_zmax);
                blend = _mm256_and_pd(class_blend, _mm256_and_pd(m_x, _mm256_and_pd(m_y, m_z)));
            }
            else if constexpr (HasClassFilter)
            {
                blend = _mm256_set_pd(blend_mask[c3], blend_mask[c2], blend_mask[c1], blend_mask[c0]);
            }
            else if constexpr (HasCoordFilter)
            {
                __m256d m_xmin = _mm256_cmp_pd(vx, vxmin, _CMP_GE_OQ);
                __m256d m_xmax = _mm256_cmp_pd(vx, vxmax, _CMP_LE_OQ);
                __m256d m_ymin = _mm256_cmp_pd(vy, vymin, _CMP_GE_OQ);
                __m256d m_ymax = _mm256_cmp_pd(vy, vymax, _CMP_LE_OQ);
                __m256d m_zmin = _mm256_cmp_pd(vz, vzmin, _CMP_GE_OQ);
                __m256d m_zmax = _mm256_cmp_pd(vz, vzmax, _CMP_LE_OQ);

                __m256d m_x = _mm256_and_pd(m_xmin, m_xmax);
                __m256d m_y = _mm256_and_pd(m_ymin, m_ymax);
                __m256d m_z = _mm256_and_pd(m_zmin, m_zmax);
                blend = _mm256_and_pd(m_x, _mm256_and_pd(m_y, m_z));
            }
            mask = _mm256_movemask_pd(blend);

            if constexpr (HasDecimation)
            {
                if (mask != 0)
                {
                    u32 num_passed = std::popcount(as<u32>(mask));
                    if (current_offset != 0 && current_offset + num_passed <= keep_every)
                    {
                        current_offset += num_passed;
                        if (current_offset == keep_every) current_offset = 0;
                        mask = 0;
                    }
                    else
                    {
                        u32 new_mask = 0;
                        for (int b = 0; b < 4; ++b)
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
            _mm256_store_pd(z_vals, vz);

            auto bin_val = [&](f64 z) {
                if (z < hist_min)
                    out_underflow++;
                else if (z >= hist_max)
                    out_overflow++;
                else
                    out_z_bins[std::min(as<i32>((z - hist_min) / bin_step), nb_bins - 1)]++;
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

        if constexpr (HasClassFilter) c = p[classification_offset] & classification_byte_mask;

        const auto* pt = reinterpret_cast<const LasPointCoordinates*>(p);
        const f64 x = (pt->x * view.header->x_scale_factor) + view.header->x_offset;
        const f64 y = (pt->y * view.header->y_scale_factor) + view.header->y_offset;
        const f64 z = (pt->z * view.header->z_scale_factor) + view.header->z_offset;

        if constexpr (HasClassFilter)
            if (!filter_mask[c]) passed = false;
        if constexpr (HasCoordFilter)
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
                out_z_bins[std::min(as<i32>((z - hist_min) / bin_step), nb_bins - 1)]++;
        }

        p += stride;
    }
}

inline void dispatch_histogram_avx2(
    bool has_class_filter, bool has_coord_filter, bool has_decimation, const u8* file_data, const HeaderView& view,
    const std::array<u8, 256>& filter_mask, const std::array<f64, 256>& blend_mask, u32 classification_offset,
    u8 classification_byte_mask, f64 filter_xmin, f64 filter_xmax, f64 filter_ymin, f64 filter_ymax, f64 filter_zmin,
    f64 filter_zmax, u64 keep_every, f64 hist_min, f64 hist_max, f64 bin_step, i32 nb_bins,
    std::vector<u64>& out_z_bins, u64& out_underflow, u64& out_overflow
)
{
    auto runner = [&]<bool HCF, bool HCoF, bool HD>() {
        build_elev_histogram_avx2<HCF, HCoF, HD>(
            file_data, view, filter_mask, blend_mask, classification_offset, classification_byte_mask, filter_xmin,
            filter_xmax, filter_ymin, filter_ymax, filter_zmin, filter_zmax, keep_every, hist_min, hist_max, bin_step,
            nb_bins, out_z_bins, out_underflow, out_overflow
        );
    };

    dispatch_bools(runner, has_class_filter, has_coord_filter, has_decimation);
}

} // namespace laspar