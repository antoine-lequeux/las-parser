#include "benchmark.hpp"
#include "format_dispatcher.hpp"
#include "memory_mapper.hpp"
#include <chrono>
#include <print>
#include <vector>

using namespace laspar;

int main()
{
    auto file_result = MemoryMappedFile::open("test.las");

    if (!file_result)
    {
        std::println(stderr, "Fatal error: {}", file_result.error());
        return 1;
    }

    auto header_result = validate_las_header(file_result->data(), file_result->size());
    if (!header_result)
    {
        std::println(stderr, "Invalid LAS file: {}", header_result.error());
        return 1;
    }

    const auto& view = header_result.value();

    run_benchmark(view, file_result->data());

    std::println("File loaded. Point format: {}", static_cast<i32>(view.header->point_data_record_format & 0x3F));
    std::println("Scanning: {} points for classifications...\n", view.point_count);

    std::vector<u64> class_counts(256, 0);
    auto start = std::chrono::high_resolution_clock::now();

    auto result = dispatch_by_format(file_result->data(), view, [&class_counts](const auto& pt) {
        u8 point_class = 0;

        if constexpr (requires { pt.base; })
        {
            // Legacy formats: class is stored in the lower 5 bits.
            point_class = pt.base.classification & 0x1F;
        }
        else if constexpr (requires { pt.base_modern; })
        {
            // Modern formats: class is stored in the full byte.
            point_class = pt.base_modern.classification;
        }
        else
        {
            static_assert(false, "Unknown LAS point struct or incorrect required members.");
        }

        class_counts[point_class]++;
    });

    if (!result)
    {
        std::println(stderr, "Error: {}", result.error());
        return 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();

    std::println("--- Classification breakdown ---");
    for (usize i = 0; i < class_counts.size(); ++i)
        if (class_counts[i] > 0) std::println("Class {:<3}: {:<10} points", i, class_counts[i]);
    std::println("--------------------------------\nScan completed in {:.4f} seconds", seconds);

    return 0;
}