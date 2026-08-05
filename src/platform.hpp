#pragma once

#include "types.hpp"
#include <algorithm>
#include <filesystem>
#include <utility>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <fcntl.h>
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <time.h>
    #include <unistd.h>
#endif

namespace laspar
{
namespace platform
{

#ifdef _WIN32
using FileHandle = HANDLE;
using MapHandle = HANDLE;
inline const FileHandle INVALID_FILE_HANDLE = INVALID_HANDLE_VALUE;
#else
using FileHandle = i32;
using MapHandle = i32;
inline const FileHandle INVALID_FILE_HANDLE = -1;
#endif

inline FileHandle open_file_write(const std::filesystem::path& filepath) noexcept
{
#ifdef _WIN32
    return CreateFileW(filepath.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
#else
    return ::open(filepath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
#endif
}

inline usize write_file(FileHandle file, const void* data, usize size)
{
    const u8* ptr = as<const u8*>(data);
    usize total_written = 0;

#ifdef _WIN32
    while (total_written < size)
    {
        DWORD to_write = as<DWORD>(std::min<usize>(size - total_written, 0xFFFFFFFF));
        DWORD written = 0;
        if (!WriteFile(file, ptr + total_written, to_write, &written, nullptr) || written == 0) break;
        total_written += written;
    }
#else
    while (total_written < size)
    {
        auto written = ::write(file, ptr + total_written, size - total_written);
        if (written <= 0)
        {
            if (errno == EINTR) continue; // Retry if the write was interrupted
            break;
        }
        total_written += as<usize>(written);
    }
#endif
    return total_written;
}

inline void seek_file(FileHandle file, u64 offset)
{
#ifdef _WIN32
    LARGE_INTEGER li;
    li.QuadPart = as<i64>(offset);
    SetFilePointerEx(file, li, nullptr, FILE_BEGIN);
#else
    ::lseek(file, as<off_t>(offset), SEEK_SET);
#endif
}

inline void close_file(FileHandle file)
{
#ifdef _WIN32
    if (file != INVALID_FILE_HANDLE) CloseHandle(file);
#else
    if (file != INVALID_FILE_HANDLE) ::close(file);
#endif
}

inline bool is_valid_file_handle(FileHandle file)
{
    return file != INVALID_FILE_HANDLE;
}

struct MappedMemory
{
    const u8* data = nullptr;
    usize size = 0;
    FileHandle file_handle = INVALID_FILE_HANDLE;
    MapHandle mapping_handle = {};

    MappedMemory() = default;

    MappedMemory(const u8* d, usize s, FileHandle f, MapHandle m) : data(d), size(s), file_handle(f), mapping_handle(m)
    {
    }

    ~MappedMemory() { unmap(); }

    MappedMemory(const MappedMemory&) = delete;
    MappedMemory& operator=(const MappedMemory&) = delete;

    MappedMemory(MappedMemory&& other) noexcept
        : data(std::exchange(other.data, nullptr)), size(std::exchange(other.size, 0)),
          file_handle(std::exchange(other.file_handle, INVALID_FILE_HANDLE)),
          mapping_handle(std::exchange(other.mapping_handle, {}))
    {
    }

    MappedMemory& operator=(MappedMemory&& other) noexcept
    {
        if (this != &other)
        {
            unmap();
            data = std::exchange(other.data, nullptr);
            size = std::exchange(other.size, 0);
            file_handle = std::exchange(other.file_handle, INVALID_FILE_HANDLE);
            mapping_handle = std::exchange(other.mapping_handle, {});
        }
        return *this;
    }

    void unmap()
    {
#ifdef _WIN32
        if (data) UnmapViewOfFile(data);
        if (mapping_handle) CloseHandle(mapping_handle);
        if (file_handle != INVALID_FILE_HANDLE) CloseHandle(file_handle);
#else
        if (data) ::munmap(const_cast<void*>(as<const void*>(data)), size);
        if (file_handle != INVALID_FILE_HANDLE) ::close(file_handle);
#endif
        data = nullptr;
        size = 0;
        mapping_handle = {};
        file_handle = INVALID_FILE_HANDLE;
    }
};

inline Result<MappedMemory> map_file_read(const std::filesystem::path& filepath)
{
    std::error_code ec;
    usize size = std::filesystem::file_size(filepath, ec);

    if (ec) return Fail(Error::FileMissingOrUnavailable);
    if (size == 0) return Fail(Error::FileEmpty);

#ifdef _WIN32
    FileHandle file_handle = CreateFileW(
        filepath.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
        nullptr
    );
    if (file_handle == INVALID_FILE_HANDLE) return Fail(Error::FailedToOpenFile);

    MapHandle mapping_handle = CreateFileMappingW(file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping_handle == nullptr)
    {
        CloseHandle(file_handle);
        return Fail(Error::FailedToCreateFileMapping);
    }

    const auto* mapped_data = as<const u8*>(MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, 0));
    if (mapped_data == nullptr)
    {
        CloseHandle(mapping_handle);
        CloseHandle(file_handle);
        return Fail(Error::FailedToMapViewOfFile);
    }
#else
    FileHandle file_handle = ::open(filepath.string().c_str(), O_RDONLY);
    if (file_handle == INVALID_FILE_HANDLE) return Fail(Error::FailedToOpenFile);

    MapHandle mapping_handle = 0;

    const auto* mapped_data = as<const u8*>(::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, file_handle, 0));
    if (mapped_data == MAP_FAILED)
    {
        ::close(file_handle);
        return Fail(Error::FailedToMapViewOfFile);
    }
#endif

    return MappedMemory(mapped_data, size, file_handle, mapping_handle);
}

inline void prefetch_memory(const void* ptr, usize size)
{
#ifdef _WIN32
    auto* p = as<const volatile u8*>(ptr);
    constexpr usize page_size = 4096;
    for (usize i = 0; i < size; i += page_size) (void)p[i];
#else
    ::madvise(const_cast<void*>(as<const void*>(ptr)), size, MADV_SEQUENTIAL | MADV_WILLNEED);
#endif
}

} // namespace platform
} // namespace laspar
