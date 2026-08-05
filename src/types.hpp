#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <optional>
#include <string>
#include <string_view>

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN

namespace laspar
{

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using f32 = float;
using f64 = double;

using usize = std::size_t;

using String = std::string;
using StringView = std::string_view;

enum class Error : u8
{
    Success = 0,
    FileMissingOrUnavailable,
    FileEmpty,
    FailedToOpenFile,
    FailedToCreateFileMapping,
    FailedToMapViewOfFile,
    FileSmallerThanMinHeaderSize,
    InvalidSignature,
    UnsupportedLASVersion,
    CorruptHeader,
    TruncatedFile,
    UnsupportedPointFormat,
};

constexpr std::string_view to_string(Error err) noexcept
{
    switch (err)
    {
        case Error::Success: return "Success";
        case Error::FileMissingOrUnavailable: return "File missing or unavailable";
        case Error::FileEmpty: return "File is empty";
        case Error::FailedToOpenFile: return "Failed to open file";
        case Error::FailedToCreateFileMapping: return "Failed to create file mapping";
        case Error::FailedToMapViewOfFile: return "Failed to map view of file";
        case Error::FileSmallerThanMinHeaderSize: return "File smaller than minimum header size";
        case Error::InvalidSignature: return "Invalid LAS signature (expected 'LASF')";
        case Error::UnsupportedLASVersion: return "Unsupported LAS version";
        case Error::CorruptHeader: return "Corrupt header";
        case Error::TruncatedFile: return "Truncated file";
        case Error::UnsupportedPointFormat: return "Unsupported point format (0 to 10 are supported)";
    }
    return "Unknown error";
}

template <typename T>
using Result = std::expected<T, Error>;
template <typename E>
[[nodiscard]] constexpr std::unexpected<std::decay_t<E>> Fail(E&& err)
{
    return std::unexpected<std::decay_t<E>>(std::forward<E>(err));
}
template <typename T>
using Option = std::optional<T>;

template <typename To, typename From>
[[nodiscard]] constexpr To as(From value)
{
    return static_cast<To>(value);
}

} // namespace laspar

namespace std
{
template <>
struct formatter<laspar::Error>
{
    constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }

    auto format(laspar::Error err, auto& ctx) const { return std::format_to(ctx.out(), "{}", laspar::to_string(err)); }
};
} // namespace std