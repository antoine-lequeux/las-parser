#pragma once
#include "header_view.hpp"
#include "las_data.hpp"
#include <immintrin.h>

namespace laspar
{
// Get the smallest double out of 4.
inline double hmin_pd(__m256d v) noexcept
{
    __m256d perm = _mm256_permute2f128_pd(v, v, 1);
    __m256d min1 = _mm256_min_pd(v, perm);
    __m256d perm2 = _mm256_permute_pd(min1, 5);
    __m256d min2 = _mm256_min_pd(min1, perm2);
    return _mm256_cvtsd_f64(min2);
}

// Get the largest double out of 4.
inline double hmax_pd(__m256d v) noexcept
{
    __m256d perm = _mm256_permute2f128_pd(v, v, 1);
    __m256d max1 = _mm256_max_pd(v, perm);
    __m256d perm2 = _mm256_permute_pd(max1, 5);
    __m256d max2 = _mm256_max_pd(max1, perm2);
    return _mm256_cvtsd_f64(max2);
}

inline BoundingBox compute_bounding_box_avx2(const u8* file_data, const HeaderView& view) noexcept
{
    // Store the scale values into 256-bit registers holding 4 doubles each.
    const __m256d x_scale = _mm256_set1_pd(view.header->x_scale_factor);
    const __m256d y_scale = _mm256_set1_pd(view.header->y_scale_factor);
    const __m256d z_scale = _mm256_set1_pd(view.header->z_scale_factor);

    // Same for offset values.
    const __m256d x_offset = _mm256_set1_pd(view.header->x_offset);
    const __m256d y_offset = _mm256_set1_pd(view.header->y_offset);
    const __m256d z_offset = _mm256_set1_pd(view.header->z_offset);

    __m256d min_x = _mm256_set1_pd(std::numeric_limits<double>::max());
    __m256d max_x = _mm256_set1_pd(std::numeric_limits<double>::lowest());

    __m256d min_y = min_x, max_y = max_x;
    __m256d min_z = min_x, max_z = max_x;

    const u8* p = file_data + view.point_data_offset;
    const u32 stride = view.point_record_length;

    u64 i = 0;
    u64 batch_limit = view.point_count - (view.point_count % 4);

    for (; i < batch_limit; i += 4)
    {
        // Calculate pointers for the 4 points.
        const auto* pt0 = reinterpret_cast<const LasPointCoordinates*>(p);
        const auto* pt1 = reinterpret_cast<const LasPointCoordinates*>(p + stride);
        const auto* pt2 = reinterpret_cast<const LasPointCoordinates*>(p + stride * 2);
        const auto* pt3 = reinterpret_cast<const LasPointCoordinates*>(p + stride * 3);

        // Load 4 ints into a 128-bit register.
        __m128i vx_int = _mm_set_epi32(pt3->x, pt2->x, pt1->x, pt0->x);
        __m128i vy_int = _mm_set_epi32(pt3->y, pt2->y, pt1->y, pt0->y);
        __m128i vz_int = _mm_set_epi32(pt3->z, pt2->z, pt1->z, pt0->z);

        // Convert the 32-bit ints to 64-bit doubles.
        __m256d vx = _mm256_cvtepi32_pd(vx_int);
        __m256d vy = _mm256_cvtepi32_pd(vy_int);
        __m256d vz = _mm256_cvtepi32_pd(vz_int);

        // Compute vx = vx * x_scale + x_offset
        vx = _mm256_fmadd_pd(vx, x_scale, x_offset);
        vy = _mm256_fmadd_pd(vy, y_scale, y_offset);
        vz = _mm256_fmadd_pd(vz, z_scale, z_offset);

        min_x = _mm256_min_pd(min_x, vx);
        max_x = _mm256_max_pd(max_x, vx);

        min_y = _mm256_min_pd(min_y, vy);
        max_y = _mm256_max_pd(max_y, vy);

        min_z = _mm256_min_pd(min_z, vz);
        max_z = _mm256_max_pd(max_z, vz);

        // Jump 4 points.
        p += stride * 4;
    }

    // Get the smallest/largest double from each track.
    BoundingBox bbox;
    bbox.min_x = hmin_pd(min_x);
    bbox.max_x = hmax_pd(max_x);
    bbox.min_y = hmin_pd(min_y);
    bbox.max_y = hmax_pd(max_y);
    bbox.min_z = hmin_pd(min_z);
    bbox.max_z = hmax_pd(max_z);

    // Process the remaining points using scalar math.
    for (; i < view.point_count; ++i)
    {
        const auto* pt = reinterpret_cast<const LasPointCoordinates*>(p);

        double x = (pt->x * view.header->x_scale_factor) + view.header->x_offset;
        double y = (pt->y * view.header->y_scale_factor) + view.header->y_offset;
        double z = (pt->z * view.header->z_scale_factor) + view.header->z_offset;

        bbox.min_x = std::min(bbox.min_x, x);
        bbox.max_x = std::max(bbox.max_x, x);
        bbox.min_y = std::min(bbox.min_y, y);
        bbox.max_y = std::max(bbox.max_y, y);
        bbox.min_z = std::min(bbox.min_z, z);
        bbox.max_z = std::max(bbox.max_z, z);

        p += stride;
    }

    return bbox;
}
} // namespace laspar