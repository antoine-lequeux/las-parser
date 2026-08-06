#pragma once

#include "las_data.hpp"
#include <chrono>
#include <format>
#include <print>
#include <vector>

namespace laspar
{

struct NumberFormat : std::numpunct<char>
{
    char do_thousands_sep() const override { return ','; }
    std::string do_grouping() const override { return "\3"; }
};

template <typename T>
inline std::string fmt_num(T value)
{
    static const std::locale comma_loc(std::locale::classic(), new NumberFormat);
    return std::format(comma_loc, "{:L}", value);
}

template <usize N>
inline std::string trim_char_array(const std::array<char, N>& arr)
{
    std::string_view sv(arr.data(), arr.size());
    usize null_pos = sv.find('\0');
    if (null_pos != std::string_view::npos) sv = sv.substr(0, null_pos);

    usize end = sv.find_last_not_of(' ');
    if (end == std::string_view::npos) return "";
    return std::string(sv.substr(0, end + 1));
}

inline bool is_leap_year(u16 year)
{
    if (year % 400 == 0) return true;
    if (year % 100 == 0) return false;
    return year % 4 == 0;
}

inline i32 get_days_in_month(u16 month, u16 year)
{
    i32 days_in_month[] = {31, is_leap_year(year) ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    return days_in_month[month];
}

inline std::string format_date(u16 day_of_year, u16 year)
{
    if (day_of_year == 0 || year == 0) return "Unknown";

    const char* month_names[] = {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"};

    u16 month = 0;
    i32 day = day_of_year;
    i32 days_in_month = get_days_in_month(month, year);
    while (month < 12 && day > days_in_month)
    {
        day -= days_in_month;
        month++;
    }

    if (month >= 12) return std::format("Day {}, Year {}", day_of_year, year);
    return std::format("{} {} {}", day, month_names[month], year);
}

inline void print_header(const LasHeader& header, std::string file_name)
{
    std::println("========================================================");
    std::println(" LAS HEADER INFORMATION ({})", file_name);
    std::println("========================================================");
    std::println(" Signature:          {:<4}", std::string_view(header.signature.data(), 4));
    std::println(" Version:            {}.{}", header.version_major, header.version_minor);
    std::println(" Source ID:          {}", header.file_source_id);
    std::println(" Global encoding:    {}", header.global_encoding);
    bool is_nil_guid = (header.guid_data_1 == 0 && header.guid_data_2 == 0 && header.guid_data_3 == 0);
    for (char c : header.guid_data_4)
        if (c != 0) is_nil_guid = false;

    if (is_nil_guid)
    {
        std::println(" Project GUID:       N/A");
    }
    else
    {
        std::string guid_str = std::format(
            "{:08x}-{:04x}-{:04x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}", header.guid_data_1,
            header.guid_data_2, header.guid_data_3, as<u8>(header.guid_data_4[0]), as<u8>(header.guid_data_4[1]),
            as<u8>(header.guid_data_4[2]), as<u8>(header.guid_data_4[3]), as<u8>(header.guid_data_4[4]),
            as<u8>(header.guid_data_4[5]), as<u8>(header.guid_data_4[6]), as<u8>(header.guid_data_4[7])
        );
        std::println(" Project GUID:       {}", guid_str);
    }

    std::println(" System identifier:  {}", trim_char_array(header.system_identifier));
    std::println(" Generating soft.:   {}", trim_char_array(header.generating_software));
    std::println(" Creation date:      {}", format_date(header.creation_day_of_year, header.creation_year));
    std::println("--------------------------------------------------------");
    std::println(" Header size:        {} bytes", header.header_size);
    std::println(" Point data offset:  {} bytes", header.offset_to_point_data);
    std::println(" Number of VLRs:     {}", header.number_of_vlr);
    std::println(" Point data format:  {}", header.point_data_record_format);
    std::println(" Point record len.:   {} bytes", header.point_data_record_length);
    std::println("--------------------------------------------------------");

    const u64 point_count = (header.version_major == 1 && header.version_minor >= 4)
                                ? header.number_of_point_records
                                : header.legacy_number_of_point_records;

    std::println(" Total points:       {}", fmt_num(point_count));

    std::println("--------------------------------------------------------");
    std::println(
        " Scale factors:      X: {:.4f}, Y: {:.4f}, Z: {:.4f}", header.x_scale_factor, header.y_scale_factor,
        header.z_scale_factor
    );
    std::println(
        " Offsets:            X: {:.4f}, Y: {:.4f}, Z: {:.4f}", header.x_offset, header.y_offset, header.z_offset
    );
    std::println("--------------------------------------------------------");
    std::println(" Bounding box:");
    std::println("   X:                [{:.2f}, {:.2f}]", header.min_x, header.max_x);
    std::println("   Y:                [{:.2f}, {:.2f}]", header.min_y, header.max_y);
    std::println("   Z:                [{:.2f}, {:.2f}]", header.min_z, header.max_z);
    std::println("========================================================");
    std::println("");
}

inline bool lint_header(const LasHeader& header, u64 file_size)
{
    std::vector<std::string> errors;
    std::vector<std::string> warnings;

    std::string_view sig(header.signature.data(), 4);
    if (sig != "LASF") errors.push_back(std::format("Invalid signature: expected 'LASF', got '{}'", sig));

    if (header.version_major != 1)
        errors.push_back(std::format("Unsupported LAS major version: {}", header.version_major));

    if (header.point_data_record_format > 10)
        errors.push_back(std::format("Unknown point data record format: {}", header.point_data_record_format));

    u16 min_record_length = 0;
    switch (header.point_data_record_format)
    {
        case 0: min_record_length = 20; break;
        case 1: min_record_length = 28; break;
        case 2: min_record_length = 26; break;
        case 3: min_record_length = 34; break;
        case 4: min_record_length = 57; break;
        case 5: min_record_length = 63; break;
        case 6: min_record_length = 30; break;
        case 7: min_record_length = 36; break;
        case 8: min_record_length = 38; break;
        case 9: min_record_length = 59; break;
        case 10: min_record_length = 67; break;
        default: break;
    }

    if (min_record_length > 0 && header.point_data_record_length < min_record_length)
    {
        errors.push_back(
            std::format(
                "Point record length too small. Format {} requires at least {} bytes, got {}",
                header.point_data_record_format, min_record_length, header.point_data_record_length
            )
        );
    }

    if (header.offset_to_point_data < header.header_size)
    {
        errors.push_back(
            std::format(
                "Offset to point data ({}) is smaller than header size ({})", header.offset_to_point_data,
                header.header_size
            )
        );
    }

    const u64 point_count = (header.version_major == 1 && header.version_minor >= 4)
                                ? header.number_of_point_records
                                : header.legacy_number_of_point_records;

    const u64 expected_min_file_size = header.offset_to_point_data + (point_count * header.point_data_record_length);
    if (file_size < expected_min_file_size)
    {
        errors.push_back(
            std::format(
                "File truncated or point count corrupt. File size is {} bytes, but points require at least {} bytes.",
                file_size, expected_min_file_size
            )
        );
    }

    if (header.min_x > header.max_x)
        warnings.push_back(std::format("Bounding Box X inverted: min_x ({}) > max_x ({})", header.min_x, header.max_x));
    if (header.min_y > header.max_y)
        warnings.push_back(std::format("Bounding Box Y inverted: min_y ({}) > max_y ({})", header.min_y, header.max_y));
    if (header.min_z > header.max_z)
        warnings.push_back(std::format("Bounding Box Z inverted: min_z ({}) > max_z ({})", header.min_z, header.max_z));

    if (errors.empty() && warnings.empty())
    {
        std::println(" [LINT] SUCCESS: No header corruption detected.\n");
        return true;
    }

    std::println("============================================================");
    std::println(" LINT REPORT");
    std::println("============================================================");

    if (!errors.empty())
    {
        std::println(" [!] ERRORS:");
        for (const auto& err : errors) std::println("     - {}", err);
    }

    if (!warnings.empty())
    {
        std::println(" [*] WARNINGS:");
        for (const auto& warn : warnings) std::println("     - {}", warn);
    }
    std::println("============================================================");
    std::println("");

    return errors.empty();
}

inline void update_header_for_write(LasHeader& header, u64 point_count, const BoundingBox& bbox)
{
    auto now = std::chrono::system_clock::now();
    auto today = std::chrono::time_point_cast<std::chrono::days>(now);
    std::chrono::year_month_day ymd {today};

    u16 year = as<u16>(i32 {ymd.year()});

    std::chrono::sys_days first_day_of_year =
        std::chrono::year_month_day {ymd.year(), std::chrono::January, std::chrono::day(1)};
    u16 day_of_year = as<u16>((today - first_day_of_year).count() + 1);

    header.creation_year = year;
    header.creation_day_of_year = day_of_year;

    header.number_of_point_records = point_count;
    if (point_count > std::numeric_limits<u32>::max())
        header.legacy_number_of_point_records = 0;
    else
        header.legacy_number_of_point_records = as<u32>(point_count);

    // Tracking them during AVX2 processing is too expensive, so we return empty arrays.
    header.legacy_number_of_points_by_return.fill(0);
    header.number_of_points_by_return.fill(0);

    if (point_count > 0)
    {
        header.min_x = bbox.min_x;
        header.max_x = bbox.max_x;
        header.min_y = bbox.min_y;
        header.max_y = bbox.max_y;
        header.min_z = bbox.min_z;
        header.max_z = bbox.max_z;
    }
    else
    {
        header.min_x = header.max_x = 0;
        header.min_y = header.max_y = 0;
        header.min_z = header.max_z = 0;
    }
}

} // namespace laspar
