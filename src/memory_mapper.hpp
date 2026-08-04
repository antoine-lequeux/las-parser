#pragma once

#include <filesystem>
#include <utility>

#include "types.hpp"

#ifdef _WIN32
    #include <windows.h>
#else
    #include <fcntl.h>
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <unistd.h>
#endif

namespace laspar
{

class MemoryMappedFile
{
public:

#ifdef _WIN32
    using FileHandle = HANDLE;
    using MapHandle = HANDLE;
    inline static const FileHandle INVALID_FILE_HANDLE = INVALID_HANDLE_VALUE;
#else
    using FileHandle = i32;
    using MapHandle = i32;
    static constexpr FileHandle INVALID_FILE_HANDLE = -1;
#endif

    static Result<MemoryMappedFile, String> open(const std::filesystem::path& filepath)
    {
        std::error_code ec;
        usize size = std::filesystem::file_size(filepath, ec);

        if (ec) return Fail("File does not exist or cannot be accessed.");
        if (size == 0) return Fail("File is empty.");

#ifdef _WIN32
        FileHandle file_handle = CreateFileW(
            filepath.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
            nullptr
        );
        if (file_handle == INVALID_FILE_HANDLE) return Fail("Failed to open file via CreateFileW.");

        MapHandle mapping_handle = CreateFileMappingW(file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mapping_handle == nullptr)
        {
            CloseHandle(file_handle);
            return Fail("Failed to create file mapping.");
        }

        const auto* mapped_data = as<const u8*>(MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, 0));
        if (mapped_data == nullptr)
        {
            CloseHandle(mapping_handle);
            CloseHandle(file_handle);
            return Fail("Failed to map view of file.");
        }
#else
        FileHandle file_handle = ::open(filepath.string().c_str(), O_RDONLY);
        if (file_handle == INVALID_FILE_HANDLE) return Fail("Failed to open file.");

        MapHandle mapping_handle = 0;

        const auto* mapped_data = as<const u8*>(::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, file_handle, 0));
        if (mapped_data == MAP_FAILED)
        {
            ::close(file_handle);
            return Fail("Failed to mmap file.");
        }
#endif

        return MemoryMappedFile(mapped_data, size, file_handle, mapping_handle);
    }

    [[nodiscard]] const u8* data() const { return m_mapped_data; }
    [[nodiscard]] usize size() const { return m_file_size; }

    MemoryMappedFile(const MemoryMappedFile&) = delete;
    MemoryMappedFile& operator=(const MemoryMappedFile&) = delete;

    // Move constructor
    MemoryMappedFile(MemoryMappedFile&& other) noexcept
        : m_mapped_data(std::exchange(other.m_mapped_data, nullptr)), m_file_size(std::exchange(other.m_file_size, 0)),
          m_file_handle(std::exchange(other.m_file_handle, INVALID_FILE_HANDLE)),
          m_mapping_handle(std::exchange(other.m_mapping_handle, {}))
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
            m_file_handle = std::exchange(other.m_file_handle, INVALID_FILE_HANDLE);
            m_mapping_handle = std::exchange(other.m_mapping_handle, {});
        }
        return *this;
    }

    ~MemoryMappedFile() { cleanup(); }

private:

    const u8* m_mapped_data = nullptr;
    usize m_file_size = 0;
    FileHandle m_file_handle = INVALID_FILE_HANDLE;
    MapHandle m_mapping_handle = {};

    // Used by open() only.
    MemoryMappedFile(const u8* data, usize size, FileHandle file_handle, MapHandle mapping_handle)
        : m_mapped_data(data), m_file_size(size), m_file_handle(file_handle), m_mapping_handle(mapping_handle)
    {
    }

    void cleanup()
    {
#ifdef _WIN32
        if (m_mapped_data) UnmapViewOfFile(m_mapped_data);
        if (m_mapping_handle) CloseHandle(m_mapping_handle);
        if (m_file_handle != INVALID_FILE_HANDLE) CloseHandle(m_file_handle);
#else
        if (m_mapped_data) ::munmap(const_cast<void*>(as<const void*>(m_mapped_data)), m_file_size);
        if (m_file_handle != INVALID_FILE_HANDLE) ::close(m_file_handle);
#endif

        m_mapped_data = nullptr;
        m_file_size = 0;
        m_mapping_handle = {};
        m_file_handle = INVALID_FILE_HANDLE;
    }
};

} // namespace laspar