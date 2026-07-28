#pragma once

#ifndef NDEBUG

    #include <format>
    #include <print>
    #include <source_location>
    #include <stacktrace>

namespace laspar::detail
{
template <typename... Args>
inline void log_impl(std::format_string<Args...> fmt, Args&&... args)
{
    std::println(fmt, std::forward<Args>(args)...);
}

inline void assert_impl(const char* condition_str, std::source_location loc = std::source_location::current())
{
    std::println(
        stderr, "\n[ASSERT FAILED: {}]\n  -> File: {}:{}\n\n{}", condition_str, loc.file_name(), loc.line(),
        std::stacktrace::current(1)
    );

    #if defined(__clang__) || defined(__GNUC__)
    __builtin_trap();
    #else
    std::_Exit(EXIT_FAILURE);
    #endif
}
} // namespace laspar::detail

    #define LASPAR_ASSERT(cond)                                                                                        \
        do                                                                                                             \
        {                                                                                                              \
            if (!(cond))                                                                                               \
            {                                                                                                          \
                ::laspar::detail::assert_impl(#cond);                                                                  \
            }                                                                                                          \
        }                                                                                                              \
        while (false)

    #define LASPAR_LOG(...) ::laspar::detail::log_impl(__VA_ARGS__)

#else

    #define LASPAR_ASSERT(cond)                                                                                        \
        do                                                                                                             \
        {                                                                                                              \
            (void)sizeof(cond);                                                                                        \
        }                                                                                                              \
        while (false)

    #define LASPAR_LOG(...)                                                                                            \
        do                                                                                                             \
        {                                                                                                              \
        }                                                                                                              \
        while (false)

#endif