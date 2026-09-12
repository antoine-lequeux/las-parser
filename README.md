# LAS-Parser: High-Performance Cross-Platform Point Cloud Processing

This program is an ultra-fast, cross-platform C++23 parser and processor for LAS point cloud datasets. It is designed to read, filter, decimate, and export datasets containing millions of points in a fraction of a second.

This project was built from the ground up to push the boundaries of I/O throughput and CPU execution limits. By leveraging direct memory-mapped file access and vectorized mathematics (SIMD), this parser is capable of performing multi-constraint filtering and reductions **up to 15x faster** than traditional scalar implementations.

---

## Technical Highlights & Architecture

The core obstacles in processing multi-gigabyte point cloud datasets are usually memory limitations, I/O bottlenecks, and CPU execution time. LAS-Parser solves these challenges using modern C++ abstractions.

### 1. Vectorized SIMD Math
A significant performance hurdle in processing points is executing bounding box calculations, classification filters, and geometric masks sequentially. I replaced the initial scalar algorithms (and raw AVX2 intrinsics) with the [EVE library](https://github.com/jfalcou/eve) to produce expressive, hardware-agnostic SIMD pipelines.

The [SIMD processing loop](src/simd_processing.hpp#L150-L250) processes 4 coordinates at once (`eve::wide<f64, eve::fixed<4>>`). It uses vectorized `if_else` masks and Fused Multiply-Add (FMA) instructions to completely eliminate branching from the critical path. The SIMD pipeline is over 15x faster than scalar math on AVX2 architectures.

### 2. Zero-Copy I/O via Memory Mapping
Allocating gigabytes of RAM just to parse a file is incredibly inefficient. Instead, a [cross-platform memory mapper](src/memory_mapper.hpp#L36-L79)was implemented. Whether the parser is compiled on POSIX (`mmap`) or Windows (`MapViewOfFile`), it exposes a unified `std::span<const u8>` to the SIMD algorithms. The file is seamlessly paged into the L1/L2 cache by the operating system, creating near-instantaneous load times regardless of file size.

### 3. Extremely Fast ASCII Exporter
Exporting point clouds to `.csv` or `.xyz` format often bottlenecks on standard library float-to-string conversions (e.g., `std::to_chars` for `double`), which perform expensive rounding analysis to guarantee minimal round-trip decimals. 

To overcome this, I implemented a custom [fast float-to-ASCII stringifier](src/ascii_exporter.hpp#L42-L82). Since LAS files store precision factors as clean powers of ten (like `0.01`), it computes raw integer boundaries directly and emits ASCII digits, injecting the decimal point precisely where needed. This completely bypasses the arbitrary precision overhead of the standard `double` path.

### 4. Robust Ground-Truth Validation & CI/CD
Building complex SIMD logic introduces edge cases, especially when decimation, overlapping coordinate filters, and class exclusion limits collide. I wrote a [modern test suite using Boost.ext.ut](src/test_main.cpp#L309-L380) that dynamically compares the output of the high-speed vectorized path against the baseline scalar logic. 

I use **GitHub Actions CI/CD** to build and test the engine across `ubuntu-latest`, `macos-latest`, and `windows-latest`. This guarantees cross-platform stability (correctly mapping POSIX vs. Windows paradigms) while ensuring the SIMD bounding-boxes, headers, and point counts align perfectly across different CPU architectures.

---

## Performance Insights: Scalar vs. Vectorized

At the heart of the engine is the distinction between **scalar** and **vectorized (SIMD)** mathematics.

- **Scalar Processing** evaluates one single point at a time. It uses heavy branching (`if point is valid, do this`) which stalls modern CPU pipelines via branch mispredictions.
- **Vectorized Processing** evaluates blocks of data (for example, 4 points at once). Instead of traditional `if` statements, it evaluates conditions for all 4 points simultaneously and generates a binary mask (for example, `[True, False, False, True]`). It uses bitwise logic (`eve::if_else`) to apply transformations securely, making the CPU pipeline incredibly dense and predictable.

Because of this architectural shift, LAS-Parser minimizes cache misses, bypasses branch predictor stalls, and fully saturates CPU execution ports, leading to its massive 15x speedup over conventional methods.
