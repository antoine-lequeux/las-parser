#pragma once

#include "types.hpp"
#include <cstring>
#ifdef _WIN32
    #include <windows.h>
#else
    #include <fcntl.h>
    #include <time.h>
    #include <unistd.h>
#endif

namespace laspar
{

class BufferedFileWriter
{
public:

    BufferedFileWriter()
    {
#ifdef _WIN32
        QueryPerformanceFrequency(&m_qpf);
#endif
    }

    ~BufferedFileWriter() { close(); }

    bool open(const String& path)
    {
#ifdef _WIN32
        m_file = CreateFileA(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        return m_file != INVALID_HANDLE_VALUE;
#else
        m_fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
        return m_fd != -1;
#endif
    }

    inline void write(const void* data, usize size)
    {
        const u8* p = static_cast<const u8*>(data);
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
#ifdef _WIN32
            LARGE_INTEGER start, end;
            QueryPerformanceCounter(&start);

            DWORD written;
            WriteFile(m_file, m_buffer, static_cast<DWORD>(m_pos), &written, nullptr);

            QueryPerformanceCounter(&end);
            m_io_seconds += static_cast<f64>(end.QuadPart - start.QuadPart) / static_cast<f64>(m_qpf.QuadPart);
#else
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);

            ::write(m_fd, m_buffer, m_pos);

            clock_gettime(CLOCK_MONOTONIC, &end);
            m_io_seconds += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
#endif
            m_pos = 0;
        }
    }

    void seek(u64 offset)
    {
        flush();
#ifdef _WIN32
        LARGE_INTEGER li;
        li.QuadPart = static_cast<i64>(offset);
        SetFilePointerEx(m_file, li, nullptr, FILE_BEGIN);
#else
        ::lseek(m_fd, offset, SEEK_SET);
#endif
    }

    void close()
    {
        flush();
#ifdef _WIN32
        if (m_file != INVALID_HANDLE_VALUE)
        {
            CloseHandle(m_file);
            m_file = INVALID_HANDLE_VALUE;
        }
#else
        if (m_fd != -1)
        {
            ::close(m_fd);
            m_fd = -1;
        }
#endif
    }

    bool is_open() const
    {
#ifdef _WIN32
        return m_file != INVALID_HANDLE_VALUE;
#else
        return m_fd != -1;
#endif
    }
    f64 get_io_seconds() const { return m_io_seconds; }

private:

#ifdef _WIN32
    HANDLE m_file = INVALID_HANDLE_VALUE;
    LARGE_INTEGER m_qpf;
#else
    i32 m_fd = -1;
#endif
    u8 m_buffer[65536];
    usize m_pos = 0;
    f64 m_io_seconds = 0.0;
};

} // namespace laspar
