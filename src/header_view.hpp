#pragma once

#include "las_data.hpp"
#include <expected>

namespace laspar
{

struct HeaderView
{
    const LasHeader* header;
    u64 point_count;
    u32 point_data_offset;
    u16 point_record_length;
};

inline std::expected<HeaderView, Error> validate_las_header(const u8* data, usize file_size)
{
    if (file_size < sizeof(LasHeader)) return std::unexpected(Error::FileSmallerThanMinHeaderSize);

    const auto* header = reinterpret_cast<const LasHeader*>(data);

    std::string_view sig(header->signature);
    if (sig != "LASF") return std::unexpected(Error::InvalidSignature);

    if (header->version_major != 1 || header->version_minor > 4) return std::unexpected(Error::UnsupportedLASVersion);

    if (header->offset_to_point_data < header->header_size) return std::unexpected(Error::CorruptHeader);

    u64 total_points = (header->version_major == 1 && header->version_minor >= 4)
                           ? header->number_of_point_records
                           : header->legacy_number_of_point_records;

    u16 point_len = header->point_data_record_length;
    if (point_len == 0) return std::unexpected(Error::CorruptHeader);

    u64 required_bytes = as<u64>(header->offset_to_point_data) + (total_points * point_len);
    if (file_size < required_bytes) return std::unexpected(Error::TruncatedFile);

    return HeaderView {
        .header = header,
        .point_count = total_points,
        .point_data_offset = header->offset_to_point_data,
        .point_record_length = point_len
    };
}
} // namespace laspar