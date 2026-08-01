#pragma once

#include "las_data.hpp"
#include <cstdint>

namespace laspar
{

struct HeaderView
{
    const LasHeader* header;
    u64 point_count;
    u32 point_data_offset;
    u16 point_record_length;
};

inline Result<HeaderView, StringView> validate_las_header(const u8* data, usize file_size)
{
    if (file_size < sizeof(LasHeader)) return Fail("File is smaller than the minimum LAS header size.");

    const auto* header = reinterpret_cast<const LasHeader*>(data);

    StringView sig(header->signature);
    if (sig != "LASF") return Fail("Invalid signature (expected 'LASF').");

    if (header->version_major != 1 || header->version_minor > 4) return Fail("Unsupported LAS version.");

    if (header->offset_to_point_data < header->header_size)
        return Fail("Corrupt header (point data offset is inside the header).");

    u64 total_points = header->number_of_point_records;
    if (total_points == 0 && header->legacy_number_of_point_records > 0)
        total_points = header->legacy_number_of_point_records;

    u16 point_len = header->point_data_record_length;
    if (point_len == 0) return Fail("Corrupt header (point record length cannot be 0).");

    u64 required_bytes = static_cast<uint64_t>(header->offset_to_point_data) + (total_points * point_len);
    if (file_size < required_bytes)
        return Fail("Truncated file (file size is smaller than expected point data bounds).");

    return HeaderView {
        .header = header,
        .point_count = total_points,
        .point_data_offset = header->offset_to_point_data,
        .point_record_length = point_len
    };
}
} // namespace laspar