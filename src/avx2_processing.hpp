#pragma once
#include "header_view.hpp"
#include "las_data.hpp"
#include <array>
#include <immintrin.h>
#include <vector>

namespace laspar
{
// Get the smallest double out of 4.
inline double hmin_pd(__m256d v)
{
    __m256d perm = _mm256_permute2f128_pd(v, v, 1);
    __m256d min1 = _mm256_min_pd(v, perm);
    __m256d perm2 = _mm256_permute_pd(min1, 5);
    __m256d min2 = _mm256_min_pd(min1, perm2);
    return _mm256_cvtsd_f64(min2);
}

// Get the largest double out of 4.
inline double hmax_pd(__m256d v)
{
    __m256d perm = _mm256_permute2f128_pd(v, v, 1);
    __m256d max1 = _mm256_max_pd(v, perm);
    __m256d perm2 = _mm256_permute_pd(max1, 5);
    __m256d max2 = _mm256_max_pd(max1, perm2);
    return _mm256_cvtsd_f64(max2);
}

// Get the sum of all 4 doubles.
inline double hsum_pd(__m256d v)
{
    __m128d lo = _mm256_extractf128_pd(v, 0);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d sum1 = _mm_add_pd(lo, hi);
    __m128d sum2 = _mm_add_pd(sum1, _mm_unpackhi_pd(sum1, sum1));
    return _mm_cvtsd_f64(sum2);
}

// AVX2 point processing function to get all data required in the command line.
template <bool HasFilter, bool DoCount, bool DoBBox, bool DoElev>
inline ProcessResult process_points_avx2(
    const u8* file_data, const HeaderView& view, const std::array<u8, 256>& filter_mask,
    const std::array<double, 256>& blend_mask, u32 classification_offset, u8 classification_byte_mask,
    std::vector<u64>& out_class_counts
)
{
    ProcessResult result {};

    // Store the scale values into 256-bit registers holding 4 doubles each.
    const __m256d x_scale = _mm256_set1_pd(view.header->x_scale_factor);
    const __m256d y_scale = _mm256_set1_pd(view.header->y_scale_factor);
    const __m256d z_scale = _mm256_set1_pd(view.header->z_scale_factor);

    // Same for offset values.
    const __m256d x_offset = _mm256_set1_pd(view.header->x_offset);
    const __m256d y_offset = _mm256_set1_pd(view.header->y_offset);
    const __m256d z_offset = _mm256_set1_pd(view.header->z_offset);

    __m256d min_x, max_x, min_y, max_y, min_z, max_z;
    const __m256d pos_inf = _mm256_set1_pd(std::numeric_limits<double>::max());
    const __m256d neg_inf = _mm256_set1_pd(std::numeric_limits<double>::lowest());

    if constexpr (DoBBox)
    {
        min_x = pos_inf;
        max_x = neg_inf;
        min_y = pos_inf;
        max_y = neg_inf;
        min_z = pos_inf;
        max_z = neg_inf;
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

    u64 i = 0;
    const u64 batch_limit = view.point_count - (view.point_count % 4);
    u64 passed_count = 0;

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

        if constexpr (HasFilter || DoCount)
        {
            u8 c0 = p[classification_offset] & classification_byte_mask;
            u8 c1 = p[stride + classification_offset] & classification_byte_mask;
            u8 c2 = p[stride * 2 + classification_offset] & classification_byte_mask;
            u8 c3 = p[stride * 3 + classification_offset] & classification_byte_mask;

            if constexpr (HasFilter)
            {
                passed_count += filter_mask[c0] + filter_mask[c1] + filter_mask[c2] + filter_mask[c3];

                __m256d blend = _mm256_set_pd(blend_mask[c3], blend_mask[c2], blend_mask[c1], blend_mask[c0]);

                if constexpr (DoElev)
                {
                    __m256d vz_filtered = _mm256_blendv_pd(zero_pd, vz, blend);
                    sum_z_vec = _mm256_add_pd(sum_z_vec, vz_filtered);
                    sum_z2_vec = _mm256_fmadd_pd(vz_filtered, vz_filtered, sum_z2_vec);
                }

                if constexpr (DoBBox)
                {
                    min_x = _mm256_min_pd(min_x, _mm256_blendv_pd(pos_inf, vx, blend));
                    max_x = _mm256_max_pd(max_x, _mm256_blendv_pd(neg_inf, vx, blend));
                    min_y = _mm256_min_pd(min_y, _mm256_blendv_pd(pos_inf, vy, blend));
                    max_y = _mm256_max_pd(max_y, _mm256_blendv_pd(neg_inf, vy, blend));
                    min_z = _mm256_min_pd(min_z, _mm256_blendv_pd(pos_inf, vz, blend));
                    max_z = _mm256_max_pd(max_z, _mm256_blendv_pd(neg_inf, vz, blend));
                }
            }
            else
            {
                passed_count += 4;

                if constexpr (DoElev)
                {
                    sum_z_vec = _mm256_add_pd(sum_z_vec, vz);
                    sum_z2_vec = _mm256_fmadd_pd(vz, vz, sum_z2_vec);
                }

                if constexpr (DoBBox)
                {
                    min_x = _mm256_min_pd(min_x, vx);
                    max_x = _mm256_max_pd(max_x, vx);
                    min_y = _mm256_min_pd(min_y, vy);
                    max_y = _mm256_max_pd(max_y, vy);
                    min_z = _mm256_min_pd(min_z, vz);
                    max_z = _mm256_max_pd(max_z, vz);
                }
            }

            if constexpr (DoCount)
            {
                if constexpr (HasFilter)
                {
                    if (filter_mask[c0]) out_class_counts[c0]++;
                    if (filter_mask[c1]) out_class_counts[c1]++;
                    if (filter_mask[c2]) out_class_counts[c2]++;
                    if (filter_mask[c3]) out_class_counts[c3]++;
                }
                else
                {
                    out_class_counts[c0]++;
                    out_class_counts[c1]++;
                    out_class_counts[c2]++;
                    out_class_counts[c3]++;
                }
            }
        }
        else
        {
            passed_count += 4;

            if constexpr (DoElev)
            {
                sum_z_vec = _mm256_add_pd(sum_z_vec, vz);
                sum_z2_vec = _mm256_fmadd_pd(vz, vz, sum_z2_vec);
            }

            if constexpr (DoBBox)
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

    if constexpr (DoBBox)
    {
        result.bbox.min_x = hmin_pd(min_x);
        result.bbox.max_x = hmax_pd(max_x);
        result.bbox.min_y = hmin_pd(min_y);
        result.bbox.max_y = hmax_pd(max_y);
        result.bbox.min_z = hmin_pd(min_z);
        result.bbox.max_z = hmax_pd(max_z);
    }

    // Process the remaining points using scalar math.
    for (; i < view.point_count; ++i)
    {
        if constexpr (HasFilter || DoCount)
        {
            u8 c = p[classification_offset] & classification_byte_mask;
            if constexpr (HasFilter)
            {
                if (!filter_mask[c])
                {
                    p += stride;
                    continue;
                }
            }
            if constexpr (DoCount) out_class_counts[c]++;
        }
        passed_count++;

        const auto* pt = reinterpret_cast<const LasPointCoordinates*>(p);
        const double x = (pt->x * view.header->x_scale_factor) + view.header->x_offset;
        const double y = (pt->y * view.header->y_scale_factor) + view.header->y_offset;
        const double z = (pt->z * view.header->z_scale_factor) + view.header->z_offset;

        if constexpr (DoElev)
        {
            result.sum_z += z;
            result.sum_z2 += z * z;
        }

        if constexpr (DoBBox)
        {
            result.bbox.min_x = std::min(result.bbox.min_x, x);
            result.bbox.max_x = std::max(result.bbox.max_x, x);
            result.bbox.min_y = std::min(result.bbox.min_y, y);
            result.bbox.max_y = std::max(result.bbox.max_y, y);
            result.bbox.min_z = std::min(result.bbox.min_z, z);
            result.bbox.max_z = std::max(result.bbox.max_z, z);
        }

        p += stride;
    }

    result.points_processed = passed_count;
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
    bool has_filter, bool do_count, bool do_bbox, bool do_elev, const u8* file_data, const HeaderView& view,
    const std::array<u8, 256>& filter_mask, const std::array<double, 256>& blend_mask, u32 classification_offset,
    u8 classification_byte_mask, std::vector<u64>& out_class_counts
)
{
    auto runner = [&]<bool HF, bool DC, bool DB, bool DE>() {
        return process_points_avx2<HF, DC, DB, DE>(
            file_data, view, filter_mask, blend_mask, classification_offset, classification_byte_mask, out_class_counts
        );
    };

    return dispatch_bools(runner, has_filter, do_count, do_bbox, do_elev);
}

} // namespace laspar