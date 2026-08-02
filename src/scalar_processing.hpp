#pragma once
#include "header_view.hpp"
#include "las_data.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include <vector>

namespace laspar
{

template <bool HasClassFilter, bool HasCoordFilter, bool DoCount, bool DoBBox, bool DoElev, bool HasDecimation>
inline ProcessResult process_points_scalar(
    const u8* file_data, const HeaderView& view, const std::array<u8, 256>& filter_mask, u32 classification_offset,
    u8 classification_byte_mask, std::vector<u64>& out_class_counts, f64 filter_xmin, f64 filter_xmax, f64 filter_ymin,
    f64 filter_ymax, f64 filter_zmin, f64 filter_zmax, u64 keep_every
)
{
    ProcessResult result {};
    u64 current_offset = 0;

    f64 min_x = std::numeric_limits<f64>::max();
    f64 max_x = std::numeric_limits<f64>::lowest();
    f64 min_y = std::numeric_limits<f64>::max();
    f64 max_y = std::numeric_limits<f64>::lowest();
    f64 min_z = std::numeric_limits<f64>::max();
    f64 max_z = std::numeric_limits<f64>::lowest();

    f64 sum_z = 0.0;
    f64 sum_z2 = 0.0;

    const u8* p = file_data + view.point_data_offset;
    const u32 stride = view.point_record_length;
    const u64 point_count = view.point_count;

    u64 passed_count = 0;
    u64 dropped_count = 0;

    for (u64 i = 0; i < point_count; ++i)
    {
        i32 x_int = *reinterpret_cast<const i32*>(p);
        i32 y_int = *reinterpret_cast<const i32*>(p + 4);
        i32 z_int = *reinterpret_cast<const i32*>(p + 8);

        f64 x = static_cast<f64>(x_int) * view.header->x_scale_factor + view.header->x_offset;
        f64 y = static_cast<f64>(y_int) * view.header->y_scale_factor + view.header->y_offset;
        f64 z = static_cast<f64>(z_int) * view.header->z_scale_factor + view.header->z_offset;

        bool passed = true;
        u8 c = 0;

        if constexpr (HasClassFilter || DoCount) c = p[classification_offset] & classification_byte_mask;

        if constexpr (HasClassFilter)
        {
            if (filter_mask[c] == 0) passed = false;
        }

        if constexpr (HasCoordFilter)
        {
            if (x < filter_xmin || x > filter_xmax) passed = false;
            if (y < filter_ymin || y > filter_ymax) passed = false;
            if (z < filter_zmin || z > filter_zmax) passed = false;
        }

        if (passed && HasDecimation)
        {
            if (current_offset == 0)
                passed = true;
            else
                passed = false;
            current_offset++;
            if (current_offset == keep_every) current_offset = 0;
        }

        if (passed)
        {
            passed_count++;
            if constexpr (DoCount) out_class_counts[c]++;
            if constexpr (DoBBox)
            {
                min_x = std::min(min_x, x);
                max_x = std::max(max_x, x);
                min_y = std::min(min_y, y);
                max_y = std::max(max_y, y);
                min_z = std::min(min_z, z);
                max_z = std::max(max_z, z);
            }
            if constexpr (DoElev)
            {
                sum_z += z;
                sum_z2 += z * z;
            }
        }
        else
        {
            dropped_count++;
        }

        p += stride;
    }

    if constexpr (DoElev)
    {
        result.sum_z = sum_z;
        result.sum_z2 = sum_z2;
    }

    if constexpr (DoBBox)
    {
        result.bbox.min_x = min_x;
        result.bbox.max_x = max_x;
        result.bbox.min_y = min_y;
        result.bbox.max_y = max_y;
        result.bbox.min_z = min_z;
        result.bbox.max_z = max_z;
    }

    result.points_processed = passed_count;
    result.points_dropped = dropped_count;

    return result;
}

template <bool HasClassFilter, bool HasCoordFilter, bool HasDecimation>
inline void build_elev_histogram_scalar(
    const u8* file_data, const HeaderView& view, const std::array<u8, 256>& filter_mask, u32 classification_offset,
    u8 classification_byte_mask, f64 filter_xmin, f64 filter_xmax, f64 filter_ymin, f64 filter_ymax, f64 filter_zmin,
    f64 filter_zmax, u64 keep_every, f64 hist_min, f64 hist_max, f64 bin_step, i32 nb_bins,
    std::vector<u64>& out_z_bins, u64& out_underflow, u64& out_overflow
)
{
    const u8* p = file_data + view.point_data_offset;
    const u32 stride = view.point_record_length;
    const u64 point_count = view.point_count;

    u64 current_offset = 0;

    for (u64 i = 0; i < point_count; ++i)
    {
        i32 x_int = *reinterpret_cast<const i32*>(p);
        i32 y_int = *reinterpret_cast<const i32*>(p + 4);
        i32 z_int = *reinterpret_cast<const i32*>(p + 8);

        f64 x = static_cast<f64>(x_int) * view.header->x_scale_factor + view.header->x_offset;
        f64 y = static_cast<f64>(y_int) * view.header->y_scale_factor + view.header->y_offset;
        f64 z = static_cast<f64>(z_int) * view.header->z_scale_factor + view.header->z_offset;

        bool passed = true;
        u8 c = 0;

        if constexpr (HasClassFilter)
        {
            c = p[classification_offset] & classification_byte_mask;
            if (filter_mask[c] == 0) passed = false;
        }

        if constexpr (HasCoordFilter)
        {
            if (x < filter_xmin || x > filter_xmax) passed = false;
            if (y < filter_ymin || y > filter_ymax) passed = false;
            if (z < filter_zmin || z > filter_zmax) passed = false;
        }

        if (passed && HasDecimation)
        {
            if (current_offset == 0)
                passed = true;
            else
                passed = false;
            current_offset++;
            if (current_offset == keep_every) current_offset = 0;
        }

        if (passed)
        {
            if (z < hist_min)
                out_underflow++;
            else if (z >= hist_max)
                out_overflow++;
            else
                out_z_bins[std::min(static_cast<i32>((z - hist_min) / bin_step), nb_bins - 1)]++;
        }

        p += stride;
    }
}

} // namespace laspar