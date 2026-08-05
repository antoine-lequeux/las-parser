#pragma once

#include "platform.hpp"
#include "types.hpp"
#include <cstring>

namespace laspar
{

class BufferedFileWriter
{
public:

    BufferedFileWriter() = default;

    ~BufferedFileWriter() { close(); }

    bool open(const String& path)
    {
        m_file = platform::open_file_write(path);
        return platform::is_valid_file_handle(m_file);
    }

    inline void write(const void* data, usize size)
    {
        const u8* p = as<const u8*>(data);
        while (size > 0)
        {
            usize space = sizeof(m_buffer) - m_pos;
            usize to_copy = (size < space) ? size : space;
            std::memcpy(m_buffer + m_pos, p, to_copy);
            m_pos += to_copy;
            p += to_copy;
            size -= to_copy;

            if (m_pos == sizeof(m_buffer)) flush();
        }
    }

    void flush()
    {
        if (m_pos > 0 && is_open())
        {
            f64 start = std::chrono::duration<f64>(std::chrono::steady_clock::now().time_since_epoch()).count();
            platform::write_file(m_file, m_buffer, m_pos);
            m_io_seconds +=
                std::chrono::duration<f64>(std::chrono::steady_clock::now().time_since_epoch()).count() - start;
            m_pos = 0;
        }
    }

    void seek(u64 offset)
    {
        flush();
        platform::seek_file(m_file, offset);
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

    bool is_open() const { return platform::is_valid_file_handle(m_file); }
    f64 get_io_seconds() const { return m_io_seconds; }

private:

    platform::FileHandle m_file = platform::INVALID_FILE_HANDLE;
    u8 m_buffer[65536];
    usize m_pos = 0;
    f64 m_io_seconds = 0.0;
};

} // namespace laspar
