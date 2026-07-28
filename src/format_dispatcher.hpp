#pragma once

#include "header_view.hpp"
#include "las_data.hpp"
#include <expected>
#include <string>

namespace laspar
{

template <typename PointFormat, typename Func>
void iterate_points(const u8* file_data, const HeaderView& view, Func&& callback)
{
    const u8* p = file_data + view.point_data_offset;
    const u32 stride = view.point_record_length;

    for (u64 i = 0; i < view.point_count; ++i)
    {
        const auto* pt = reinterpret_cast<const PointFormat*>(p);
        callback(*pt);
        p += stride;
    }
}

template <typename Func>
std::expected<void, std::string> dispatch_by_format(const u8* file_data, const HeaderView& view, Func&& callback)
{
    u8 format_id = view.header->point_data_record_format & 0x3F;

    switch (format_id)
    {
        case 0: iterate_points<LasPointFormat0>(file_data, view, std::forward<Func>(callback)); break;
        case 1: iterate_points<LasPointFormat1>(file_data, view, std::forward<Func>(callback)); break;
        case 2: iterate_points<LasPointFormat2>(file_data, view, std::forward<Func>(callback)); break;
        case 3: iterate_points<LasPointFormat3>(file_data, view, std::forward<Func>(callback)); break;
        case 4: iterate_points<LasPointFormat4>(file_data, view, std::forward<Func>(callback)); break;
        case 5: iterate_points<LasPointFormat5>(file_data, view, std::forward<Func>(callback)); break;
        case 6: iterate_points<LasPointFormat6>(file_data, view, std::forward<Func>(callback)); break;
        case 7: iterate_points<LasPointFormat7>(file_data, view, std::forward<Func>(callback)); break;
        case 8: iterate_points<LasPointFormat8>(file_data, view, std::forward<Func>(callback)); break;
        case 9: iterate_points<LasPointFormat9>(file_data, view, std::forward<Func>(callback)); break;
        case 10: iterate_points<LasPointFormat10>(file_data, view, std::forward<Func>(callback)); break;

        default: return std::unexpected("Unsupported point format: " + std::to_string(format_id));
    }

    return {};
}
} // namespace laspar