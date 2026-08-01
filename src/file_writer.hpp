#pragma once

#include "types.hpp"
#include <cstring>
#include <windows.h>

namespace laspar
{

class BufferedFileWriter
{
public:

    BufferedFileWriter() { QueryPerformanceFrequency(&m_qpf); }

    ~BufferedFileWriter() { close(); }

    bool open(const String& path)
    {
        m_file = CreateFileA(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        return m_file != INVALID_HANDLE_VALUE;
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
        if (m_pos > 0 && m_file != INVALID_HANDLE_VALUE)
        {
            LARGE_INTEGER start, end;
            QueryPerformanceCounter(&start);

            DWORD written;
            WriteFile(m_file, m_buffer, static_cast<DWORD>(m_pos), &written, nullptr);

            QueryPerformanceCounter(&end);
            m_io_seconds += static_cast<f64>(end.QuadPart - start.QuadPart) / m_qpf.QuadPart;

            m_pos = 0;
        }
    }

    void seek(u64 offset)
    {
        flush();
        LARGE_INTEGER li;
        li.QuadPart = offset;
        SetFilePointerEx(m_file, li, nullptr, FILE_BEGIN);
    }

    void close()
    {
        flush();
        if (m_file != INVALID_HANDLE_VALUE)
        {
            CloseHandle(m_file);
            m_file = INVALID_HANDLE_VALUE;
        }
    }

    bool is_open() const { return m_file != INVALID_HANDLE_VALUE; }
    f64 get_io_seconds() const { return m_io_seconds; }

private:

    HANDLE m_file = INVALID_HANDLE_VALUE;
    u8 m_buffer[65536];
    usize m_pos = 0;
    f64 m_io_seconds = 0.0;
    LARGE_INTEGER m_qpf;
};

} // namespace laspar
