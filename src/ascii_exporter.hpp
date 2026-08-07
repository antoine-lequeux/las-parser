#pragma once

#include "header_view.hpp"
#include "las_data.hpp"
#include "platform.hpp"
#include <charconv>
#include <cstring>
#include <expected>
#include <memory>
#include <string>
#include <string_view>

namespace laspar
{

class AsciiWriter
{
public:

    AsciiWriter() = default;
    ~AsciiWriter() { close(); }

    std::expected<void, Error> open(const std::string& path)
    {
        m_file = platform::open_file_write(path);
        if (!platform::is_valid_file_handle(m_file)) return std::unexpected(Error::FileMissingOrUnavailable);
        return {};
    }

    void close()
    {
        flush();
        if (platform::is_valid_file_handle(m_file))
        {
            platform::close_file(m_file);
            m_file = platform::INVALID_FILE_HANDLE;
        }
    }

    inline void write_char(char c)
    {
        if (m_pos >= m_buffer_size) flush();
        m_buffer[m_pos++] = c;
    }

    inline void write_str(std::string_view s)
    {
        const char* p = s.data();
        usize size = s.size();
        while (size > 0)
        {
            usize space = m_buffer_size - m_pos;
            usize to_copy = (size < space) ? size : space;
            std::memcpy(m_buffer.get() + m_pos, p, to_copy);
            m_pos += to_copy;
            p += to_copy;
            size -= to_copy;
            if (m_pos == m_buffer_size) flush();
        }
    }

    template <typename T>
    inline void write_num(T value)
    {
        if (m_pos + 64 > m_buffer_size) flush();
        auto [ptr, ec] = std::to_chars(m_buffer.get() + m_pos, m_buffer.get() + m_buffer_size, value);
        if (ec == std::errc()) m_pos = as<usize>(ptr - m_buffer.get());
    }

    void flush()
    {
        if (m_pos > 0 && platform::is_valid_file_handle(m_file))
        {
            f64 start = std::chrono::duration<f64>(std::chrono::steady_clock::now().time_since_epoch()).count();
            platform::write_file(m_file, reinterpret_cast<const u8*>(m_buffer.get()), m_pos);
            m_io_seconds +=
                std::chrono::duration<f64>(std::chrono::steady_clock::now().time_since_epoch()).count() - start;
            m_pos = 0;
        }
    }

    f64 get_io_seconds() const { return m_io_seconds; }

private:

    platform::FileHandle m_file = platform::INVALID_FILE_HANDLE;
    std::unique_ptr<char[]> m_buffer = std::make_unique<char[]>(1024 * 1024);
    usize m_buffer_size = 1024 * 1024;
    usize m_pos = 0;
    f64 m_io_seconds = 0.0;
};

inline bool keep_point(
    const LasPointCoordinates* pt, u8 classification, bool has_class_filter, bool has_coord_filter, bool has_decimation,
    const std::array<u8, 256>& filter_mask, f64 filter_xmin, f64 filter_xmax, f64 filter_ymin, f64 filter_ymax,
    f64 filter_zmin, f64 filter_zmax, const LasHeader& header, u64& current_decimation_offset, u64 keep_every
)
{
    if (has_class_filter && !filter_mask[classification]) return false;

    if (has_coord_filter)
    {
        f64 x = (pt->x * header.x_scale_factor) + header.x_offset;
        f64 y = (pt->y * header.y_scale_factor) + header.y_offset;
        f64 z = (pt->z * header.z_scale_factor) + header.z_offset;
        if (x < filter_xmin || x > filter_xmax || y < filter_ymin || y > filter_ymax || z < filter_zmin ||
            z > filter_zmax)
            return false;
    }

    if (has_decimation)
    {
        if (current_decimation_offset != 0)
        {
            current_decimation_offset++;
            if (current_decimation_offset == keep_every) current_decimation_offset = 0;
            return false;
        }
        current_decimation_offset++;
        if (current_decimation_offset == keep_every) current_decimation_offset = 0;
    }

    return true;
}

inline std::expected<f64, Error> export_xyz(
    const std::string& output_file, bool has_class_filter, bool has_coord_filter, bool has_decimation,
    const u8* file_data, const HeaderView& view, const std::array<u8, 256>& filter_mask, u32 classification_offset,
    u8 classification_byte_mask, f64 filter_xmin, f64 filter_xmax, f64 filter_ymin, f64 filter_ymax, f64 filter_zmin,
    f64 filter_zmax, u64 keep_every
)
{
    AsciiWriter writer;
    auto res = writer.open(output_file);
    if (!res) return std::unexpected(Error::FileMissingOrUnavailable);

    writer.write_str("X Y Z\n");

    const u8* p = file_data + view.point_data_offset;
    const u32 stride = view.point_record_length;
    u64 current_decimation = 0;

    const f64 xs = view.header->x_scale_factor, xo = view.header->x_offset;
    const f64 ys = view.header->y_scale_factor, yo = view.header->y_offset;
    const f64 zs = view.header->z_scale_factor, zo = view.header->z_offset;

    for (u64 i = 0; i < view.point_count; ++i)
    {
        u8 c = 0;
        if (has_class_filter) c = p[classification_offset] & classification_byte_mask;
        const auto* pt = reinterpret_cast<const LasPointCoordinates*>(p);

        if (keep_point(
                pt, c, has_class_filter, has_coord_filter, has_decimation, filter_mask, filter_xmin, filter_xmax,
                filter_ymin, filter_ymax, filter_zmin, filter_zmax, *view.header, current_decimation, keep_every
            ))
        {
            f64 x = (pt->x * xs) + xo;
            f64 y = (pt->y * ys) + yo;
            f64 z = (pt->z * zs) + zo;

            writer.write_num(x);
            writer.write_char(' ');
            writer.write_num(y);
            writer.write_char(' ');
            writer.write_num(z);
            writer.write_char('\n');
        }
        p += stride;
    }

    writer.close();
    return writer.get_io_seconds();
}

template <typename FormatStruct>
inline void write_csv_row(AsciiWriter& writer, const u8* p, const LasHeader& header)
{
    const auto* pt = reinterpret_cast<const FormatStruct*>(p);

    f64 x = (pt->get_x() * header.x_scale_factor) + header.x_offset;
    f64 y = (pt->get_y() * header.y_scale_factor) + header.y_offset;
    f64 z = (pt->get_z() * header.z_scale_factor) + header.z_offset;

    writer.write_num(x);
    writer.write_char(',');
    writer.write_num(y);
    writer.write_char(',');
    writer.write_num(z);
    writer.write_char(',');

    if constexpr (requires { pt->base; })
    {
        writer.write_num(pt->base.intensity);
        writer.write_char(',');
        writer.write_num(pt->base.return_info);
        writer.write_char(',');
        writer.write_num(pt->get_classification());
        writer.write_char(',');
        writer.write_num(pt->base.scan_angle_rank);
        writer.write_char(',');
        writer.write_num(pt->base.user_data);
        writer.write_char(',');
        writer.write_num(pt->base.point_source_id);
    }
    else if constexpr (requires { pt->base_modern; })
    {
        writer.write_num(pt->base_modern.intensity);
        writer.write_char(',');
        writer.write_num(pt->base_modern.return_info);
        writer.write_char(',');
        writer.write_num(pt->base_modern.flags);
        writer.write_char(',');
        writer.write_num(pt->get_classification());
        writer.write_char(',');
        writer.write_num(pt->base_modern.user_data);
        writer.write_char(',');
        writer.write_num(pt->base_modern.scan_angle);
        writer.write_char(',');
        writer.write_num(pt->base_modern.point_source_id);
        writer.write_char(',');
        writer.write_num(pt->base_modern.gps_time);
    }

    if constexpr (requires { pt->gps_time; } && requires { pt->base; })
    {
        writer.write_char(',');
        writer.write_num(pt->gps_time);
    }
    if constexpr (requires { pt->color; })
    {
        writer.write_char(',');
        writer.write_num(pt->color.red);
        writer.write_char(',');
        writer.write_num(pt->color.green);
        writer.write_char(',');
        writer.write_num(pt->color.blue);
    }
    if constexpr (requires { pt->nir; })
    {
        writer.write_char(',');
        writer.write_num(pt->nir);
    }
    if constexpr (requires { pt->wave; })
    {
        writer.write_char(',');
        writer.write_num(pt->wave.descriptor_index);
        writer.write_char(',');
        writer.write_num(pt->wave.byte_offset);
        writer.write_char(',');
        writer.write_num(pt->wave.packet_size);
        writer.write_char(',');
        writer.write_num(pt->wave.return_point_waveform_loc);
        writer.write_char(',');
        writer.write_num(pt->wave.xt);
        writer.write_char(',');
        writer.write_num(pt->wave.yt);
        writer.write_char(',');
        writer.write_num(pt->wave.zt);
    }
    writer.write_char('\n');
}

inline std::expected<f64, Error> export_csv(
    const std::string& output_file, bool has_class_filter, bool has_coord_filter, bool has_decimation,
    const u8* file_data, const HeaderView& view, const std::array<u8, 256>& filter_mask, u32 classification_offset,
    u8 classification_byte_mask, f64 filter_xmin, f64 filter_xmax, f64 filter_ymin, f64 filter_ymax, f64 filter_zmin,
    f64 filter_zmax, u64 keep_every
)
{
    AsciiWriter writer;
    auto res = writer.open(output_file);
    if (!res) return std::unexpected(Error::FileMissingOrUnavailable);

    u8 format_id = view.header->point_data_record_format & 0x3Fu;

    writer.write_str("x,y,z,intensity,return_info,");
    if (format_id >= 6) writer.write_str("flags,");
    writer.write_str("classification,");
    if (format_id < 6) writer.write_str("scan_angle_rank,");
    writer.write_str("user_data,");
    if (format_id >= 6) writer.write_str("scan_angle,");
    writer.write_str("point_source_id");

    if ((format_id >= 1 && format_id <= 5) || format_id >= 6) writer.write_str(",gps_time");
    if (format_id == 2 || format_id == 3 || format_id == 5 || format_id == 7 || format_id == 8 || format_id == 10)
        writer.write_str(",red,green,blue");
    if (format_id == 8 || format_id == 10) writer.write_str(",nir");
    if (format_id == 4 || format_id == 5 || format_id == 9 || format_id == 10)
        writer.write_str(",wave_descriptor,wave_byte_offset,wave_packet_size,wave_loc,wave_xt,wave_yt,wave_zt");
    writer.write_char('\n');

    const u8* p = file_data + view.point_data_offset;
    const u32 stride = view.point_record_length;
    u64 current_decimation = 0;

    for (u64 i = 0; i < view.point_count; ++i)
    {
        u8 c = 0;
        if (has_class_filter) c = p[classification_offset] & classification_byte_mask;
        const auto* pt = reinterpret_cast<const LasPointCoordinates*>(p);

        if (keep_point(
                pt, c, has_class_filter, has_coord_filter, has_decimation, filter_mask, filter_xmin, filter_xmax,
                filter_ymin, filter_ymax, filter_zmin, filter_zmax, *view.header, current_decimation, keep_every
            ))
        {
            switch (format_id)
            {
                case 0: write_csv_row<LasPointFormat0>(writer, p, *view.header); break;
                case 1: write_csv_row<LasPointFormat1>(writer, p, *view.header); break;
                case 2: write_csv_row<LasPointFormat2>(writer, p, *view.header); break;
                case 3: write_csv_row<LasPointFormat3>(writer, p, *view.header); break;
                case 4: write_csv_row<LasPointFormat4>(writer, p, *view.header); break;
                case 5: write_csv_row<LasPointFormat5>(writer, p, *view.header); break;
                case 6: write_csv_row<LasPointFormat6>(writer, p, *view.header); break;
                case 7: write_csv_row<LasPointFormat7>(writer, p, *view.header); break;
                case 8: write_csv_row<LasPointFormat8>(writer, p, *view.header); break;
                case 9: write_csv_row<LasPointFormat9>(writer, p, *view.header); break;
                case 10: write_csv_row<LasPointFormat10>(writer, p, *view.header); break;
            }
        }
        p += stride;
    }

    writer.close();
    return writer.get_io_seconds();
}

} // namespace laspar
