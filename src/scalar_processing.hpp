#pragma once

#include "header_view.hpp"

namespace laspar
{

inline BoundingBox compute_bounding_box_scalar(const u8* file_data, const HeaderView& view)
{
    BoundingBox bbox;

    const f64 x_scale = view.header->x_scale_factor;
    const f64 y_scale = view.header->y_scale_factor;
    const f64 z_scale = view.header->z_scale_factor;

    const f64 x_offset = view.header->x_offset;
    const f64 y_offset = view.header->y_offset;
    const f64 z_offset = view.header->z_offset;

    // Pointer on the first byte of point data.
    const u8* current_ptr = file_data + view.point_data_offset;

    for (u64 i = 0; i < view.point_count; ++i)
    {
        const auto* point = reinterpret_cast<const LasPointCoordinates*>(current_ptr);

        // Apply scale and offset.
        f64 x = (point->x * x_scale) + x_offset;
        f64 y = (point->y * y_scale) + y_offset;
        f64 z = (point->z * z_scale) + z_offset;

        bbox.min_x = std::min(bbox.min_x, x);
        bbox.max_x = std::max(bbox.max_x, x);

        bbox.min_y = std::min(bbox.min_y, y);
        bbox.max_y = std::max(bbox.max_y, y);

        bbox.min_z = std::min(bbox.min_z, z);
        bbox.max_z = std::max(bbox.max_z, z);

        // Jump to the next point.
        current_ptr += view.point_record_length;
    }

    return bbox;
}
} // namespace laspar