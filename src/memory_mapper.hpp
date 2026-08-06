#pragma once

#include <filesystem>
#include <utility>

#include "platform.hpp"
#include "types.hpp"
#include <expected>

namespace laspar
{

class MemoryMappedFile
{
public:

    static std::expected<MemoryMappedFile, Error> open(const std::filesystem::path& filepath)
    {
        auto result = platform::map_file_read(filepath);
        if (!result) return std::unexpected(result.error());
        return MemoryMappedFile(std::move(*result));
    }

    [[nodiscard]] const u8* data() const { return m_mapped.data; }
    [[nodiscard]] usize size() const { return m_mapped.size; }

    MemoryMappedFile(const MemoryMappedFile&) = delete;
    MemoryMappedFile& operator=(const MemoryMappedFile&) = delete;

    MemoryMappedFile(MemoryMappedFile&&) = default;
    MemoryMappedFile& operator=(MemoryMappedFile&&) = default;

private:

    platform::MappedMemory m_mapped;

    explicit MemoryMappedFile(platform::MappedMemory&& mapped) noexcept : m_mapped(std::move(mapped)) {}
};

} // namespace laspar