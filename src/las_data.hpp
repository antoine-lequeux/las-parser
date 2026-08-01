#pragma once

#include "types.hpp"
#include <array>
#include <limits>

namespace laspar
{
// Force 1-byte alignment to avoid padding between chunks of data.
#pragma pack(push, 1)

// Byte layout of the LAS 1.4 Public Header Block as defined here:
// https://paulbourke.net/dataformats/laz/LAS_1_4_r15.pdf (p.8).
struct LasHeader
{
    std::array<char, 4> signature;
    u16 file_source_id;
    u16 global_encoding;
    u32 guid_data_1;
    u16 guid_data_2;
    u16 guid_data_3;
    std::array<char, 8> guid_data_4;
    u8 version_major;
    u8 version_minor;
    std::array<char, 32> system_identifier;
    std::array<char, 32> generating_software;
    u16 creation_day_of_year;
    u16 creation_year;

    u16 header_size;
    u32 offset_to_point_data;
    u32 number_of_vlr;
    u8 point_data_record_format;
    u16 point_data_record_length;
    u32 legacy_number_of_point_records;
    std::array<u32, 5> legacy_number_of_points_by_return;

    f64 x_scale_factor;
    f64 y_scale_factor;
    f64 z_scale_factor;
    f64 x_offset;
    f64 y_offset;
    f64 z_offset;

    f64 max_x;
    f64 min_x;
    f64 max_y;
    f64 min_y;
    f64 max_z;
    f64 min_z;

    u64 start_of_waveform_data_packet_record;
    u64 start_of_first_evlr;
    u32 number_of_evlr;
    u64 number_of_point_records;
    std::array<u64, 15> number_of_points_by_return;
};
static_assert(sizeof(LasHeader) == 375, "LasHeader size must be exactly 375 bytes.");

// Byte layout of Point Data Record Format 0 as defined here:
// https://paulbourke.net/dataformats/laz/LAS_1_4_r15.pdf (p.16).
// Shared by formats 0 to 5.
struct LasPointBase
{
    i32 x;
    i32 y;
    i32 z;
    u16 intensity;

    // Return number (bits 0-2), nb of returns (3-5), scan dir flag (6), edge of flight line (7).
    u8 return_info;

    u8 classification;
    i8 scan_angle_rank;
    u8 user_data;
    u16 point_source_id;
};
static_assert(sizeof(LasPointBase) == 20, "LasPointBase must be exactly 20 bytes");

struct LasColor
{
    u16 red;
    u16 green;
    u16 blue;
};

struct LasWavePacket
{
    u8 descriptor_index;
    u64 byte_offset;
    u32 packet_size;
    float return_point_waveform_loc;
    float xt;
    float yt;
    float zt;
};

struct LasPointFormat0
{
    LasPointBase base;

    [[nodiscard]] i32 get_x() const { return base.x; }
    [[nodiscard]] i32 get_y() const { return base.y; }
    [[nodiscard]] i32 get_z() const { return base.z; }
    [[nodiscard]] u8 get_classification() const { return base.classification & 0x1F; }
};
struct LasPointFormat1
{
    LasPointBase base;
    f64 gps_time;

    [[nodiscard]] i32 get_x() const { return base.x; }
    [[nodiscard]] i32 get_y() const { return base.y; }
    [[nodiscard]] i32 get_z() const { return base.z; }
    [[nodiscard]] u8 get_classification() const { return base.classification & 0x1F; }
};
struct LasPointFormat2
{
    LasPointBase base;
    LasColor color;

    [[nodiscard]] i32 get_x() const { return base.x; }
    [[nodiscard]] i32 get_y() const { return base.y; }
    [[nodiscard]] i32 get_z() const { return base.z; }
    [[nodiscard]] u8 get_classification() const { return base.classification & 0x1F; }
};
struct LasPointFormat3
{
    LasPointBase base;
    f64 gps_time;
    LasColor color;

    [[nodiscard]] i32 get_x() const { return base.x; }
    [[nodiscard]] i32 get_y() const { return base.y; }
    [[nodiscard]] i32 get_z() const { return base.z; }
    [[nodiscard]] u8 get_classification() const { return base.classification & 0x1F; }
};
struct LasPointFormat4
{
    LasPointBase base;
    f64 gps_time;
    LasWavePacket wave;

    [[nodiscard]] i32 get_x() const { return base.x; }
    [[nodiscard]] i32 get_y() const { return base.y; }
    [[nodiscard]] i32 get_z() const { return base.z; }
    [[nodiscard]] u8 get_classification() const { return base.classification & 0x1F; }
};
struct LasPointFormat5
{
    LasPointBase base;
    f64 gps_time;
    LasColor color;
    LasWavePacket wave;

    [[nodiscard]] i32 get_x() const { return base.x; }
    [[nodiscard]] i32 get_y() const { return base.y; }
    [[nodiscard]] i32 get_z() const { return base.z; }
    [[nodiscard]] u8 get_classification() const { return base.classification & 0x1F; }
};

// Shared by formats 6 to 10.
struct LasPointBaseModern
{
    i32 x;
    i32 y;
    i32 z;
    u16 intensity;
    u8 return_info;
    u8 flags;
    u8 classification;
    u8 user_data;
    i16 scan_angle;
    u16 point_source_id;
    f64 gps_time;
};
static_assert(sizeof(LasPointBaseModern) == 30, "LasPointBaseModern must be exactly 30 bytes");

struct LasPointFormat6
{
    LasPointBaseModern base_modern;

    [[nodiscard]] i32 get_x() const { return base_modern.x; }
    [[nodiscard]] i32 get_y() const { return base_modern.y; }
    [[nodiscard]] i32 get_z() const { return base_modern.z; }
    [[nodiscard]] u8 get_classification() const { return base_modern.classification; }
};
struct LasPointFormat7
{
    LasPointBaseModern base_modern;
    LasColor color;

    [[nodiscard]] i32 get_x() const { return base_modern.x; }
    [[nodiscard]] i32 get_y() const { return base_modern.y; }
    [[nodiscard]] i32 get_z() const { return base_modern.z; }
    [[nodiscard]] u8 get_classification() const { return base_modern.classification; }
};
struct LasPointFormat8
{
    LasPointBaseModern base_modern;
    LasColor color;
    u16 nir;

    [[nodiscard]] i32 get_x() const { return base_modern.x; }
    [[nodiscard]] i32 get_y() const { return base_modern.y; }
    [[nodiscard]] i32 get_z() const { return base_modern.z; }
    [[nodiscard]] u8 get_classification() const { return base_modern.classification; }
};
struct LasPointFormat9
{
    LasPointBaseModern base_modern;
    LasWavePacket wave;

    [[nodiscard]] i32 get_x() const { return base_modern.x; }
    [[nodiscard]] i32 get_y() const { return base_modern.y; }
    [[nodiscard]] i32 get_z() const { return base_modern.z; }
    [[nodiscard]] u8 get_classification() const { return base_modern.classification; }
};
struct LasPointFormat10
{
    LasPointBaseModern base_modern;
    LasColor color;
    u16 nir;
    LasWavePacket wave;

    [[nodiscard]] i32 get_x() const { return base_modern.x; }
    [[nodiscard]] i32 get_y() const { return base_modern.y; }
    [[nodiscard]] i32 get_z() const { return base_modern.z; }
    [[nodiscard]] u8 get_classification() const { return base_modern.classification; }
};

struct LasPointCoordinates
{
    i32 x;
    i32 y;
    i32 z;
};

#pragma pack(pop)

struct BoundingBox
{
    f64 min_x = std::numeric_limits<f64>::max();
    f64 max_x = std::numeric_limits<f64>::lowest();

    f64 min_y = std::numeric_limits<f64>::max();
    f64 max_y = std::numeric_limits<f64>::lowest();

    f64 min_z = std::numeric_limits<f64>::max();
    f64 max_z = std::numeric_limits<f64>::lowest();
};

struct ProcessResult
{
    u64 points_processed = 0;
    u64 points_dropped = 0;

    BoundingBox bbox;
    BoundingBox dropped_bbox;

    f64 sum_z = 0.0;
    f64 sum_z2 = 0.0;

    f64 io_time_seconds = 0.0;
};

} // namespace laspar