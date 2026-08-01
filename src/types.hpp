#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

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

template <typename T, typename E>
using Result = std::expected<T, E>;
template <typename E>
using Fail = std::unexpected<E>;
template <typename T>
using Option = std::optional<T>;

} // namespace laspar