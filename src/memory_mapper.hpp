#pragma once

#include <filesystem>
#include <utility>

#include "types.hpp"

namespace laspar
{

class MemoryMappedFile
{
public:

    static Result<MemoryMappedFile, String> open(const std::filesystem::path& filepath)
    {
        std::error_code ec;
        usize size = std::filesystem::file_size(filepath, ec);

        if (ec) return Fail("File does not exist or cannot be accessed.");
        if (size == 0) return Fail("File is empty.");

        HANDLE file_handle = CreateFileW(
            filepath.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
            nullptr
        );
        if (file_handle == INVALID_HANDLE_VALUE) return Fail("Failed to open file via CreateFileW.");

        HANDLE mapping_handle = CreateFileMappingW(file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mapping_handle == nullptr)
        {
            CloseHandle(file_handle);
            return Fail("Failed to create file mapping.");
        }

        const auto* mapped_data = static_cast<const u8*>(MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, 0));
        if (mapped_data == nullptr)
        {
            CloseHandle(mapping_handle);
            CloseHandle(file_handle);
            return Fail("Failed to map view of file.");
        }

        return MemoryMappedFile(mapped_data, size, file_handle, mapping_handle);
    }

    [[nodiscard]] const u8* data() const { return m_mapped_data; }
    [[nodiscard]] usize size() const { return m_file_size; }

    MemoryMappedFile(const MemoryMappedFile&) = delete;
    MemoryMappedFile& operator=(const MemoryMappedFile&) = delete;

    // Move constructor
    MemoryMappedFile(MemoryMappedFile&& other) noexcept
        : m_mapped_data(std::exchange(other.m_mapped_data, nullptr)), m_file_size(std::exchange(other.m_file_size, 0)),
          m_file_handle(std::exchange(other.m_file_handle, INVALID_HANDLE_VALUE)),
          m_mapping_handle(std::exchange(other.m_mapping_handle, nullptr))
    {
    }

    // Move assignment operator
    MemoryMappedFile& operator=(MemoryMappedFile&& other) noexcept
    {
        if (this != &other)
        {
            cleanup();

            m_mapped_data = std::exchange(other.m_mapped_data, nullptr);
            m_file_size = std::exchange(other.m_file_size, 0);
            m_file_handle = std::exchange(other.m_file_handle, INVALID_HANDLE_VALUE);
            m_mapping_handle = std::exchange(other.m_mapping_handle, nullptr);
        }
        return *this;
    }

    ~MemoryMappedFile() { cleanup(); }

private:

    const u8* m_mapped_data = nullptr;
    usize m_file_size = 0;
    HANDLE m_file_handle = INVALID_HANDLE_VALUE;
    HANDLE m_mapping_handle = nullptr;

    // Used by open() only.
    MemoryMappedFile(const u8* data, usize size, HANDLE file_handle, HANDLE mapping_handle)
        : m_mapped_data(data), m_file_size(size), m_file_handle(file_handle), m_mapping_handle(mapping_handle)
    {
    }

    void cleanup()
    {
        if (m_mapped_data) UnmapViewOfFile(m_mapped_data);
        if (m_mapping_handle) CloseHandle(m_mapping_handle);
        if (m_file_handle != INVALID_HANDLE_VALUE) CloseHandle(m_file_handle);

        m_mapped_data = nullptr;
        m_file_size = 0;
        m_mapping_handle = nullptr;
        m_file_handle = INVALID_HANDLE_VALUE;
    }
};

} // namespace laspar