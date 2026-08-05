#pragma once

#include "las_data.hpp"

namespace laspar
{

struct HeaderView
{
    const LasHeader* header;
    u64 point_count;
    u32 point_data_offset;
    u16 point_record_length;
};

inline Result<HeaderView> validate_las_header(const u8* data, usize file_size)
{
    if (file_size < sizeof(LasHeader)) return Fail(Error::FileSmallerThanMinHeaderSize);

    const auto* header = reinterpret_cast<const LasHeader*>(data);

    StringView sig(header->signature);
    if (sig != "LASF") return Fail(Error::InvalidSignature);

    if (header->version_major != 1 || header->version_minor > 4) return Fail(Error::UnsupportedLASVersion);

    if (header->offset_to_point_data < header->header_size) return Fail(Error::CorruptHeader);

    u64 total_points = header->number_of_point_records;
    if (total_points == 0 && header->legacy_number_of_point_records > 0)
        total_points = header->legacy_number_of_point_records;

    u16 point_len = header->point_data_record_length;
    if (point_len == 0) return Fail(Error::CorruptHeader);

    u64 required_bytes = as<uint64_t>(header->offset_to_point_data) + (total_points * point_len);
    if (file_size < required_bytes) return Fail(Error::TruncatedFile);

    return HeaderView {
        .header = header,
        .point_count = total_points,
        .point_data_offset = header->offset_to_point_data,
        .point_record_length = point_len
    };
}
} // namespace laspar