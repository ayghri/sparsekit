// loop_cycles_with_ghz.cpp
//
// Adds an empirical measurement of TSC frequency (GHz) so we can convert
// cycles/iter -> ns/iter directly in the same run.
//
// Build:
//   g++ -O3 -march=native -std=c++17 -fno-unroll-loops -fno-tree-vectorize loop_cycles_with_ghz.cpp -o loop_cycles
//
// Run:
//   ./loop_cycles 100000000

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
  #define NOINLINE __attribute__((noinline))
#else
  #define NOINLINE
#endif

static inline std::uint64_t rdtscp() {
#if defined(__x86_64__) || defined(_M_X64)
  unsigned aux;
  std::uint64_t rax, rdx;
  asm volatile("rdtscp" : "=a"(rax), "=d"(rdx), "=c"(aux) ::);
  // Serialize to avoid reordering around reads.
  asm volatile("lfence" ::: "memory");
  return (rdx << 32) | rax;
#else
  #error "RDTSCP benchmark requires x86_64"
#endif
}

static inline void keep_live(std::uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
  asm volatile("" :: "r"(x) : "memory");
#endif
}

volatile std::uint64_t g_sink = 0x123456789abcdefULL;

// Opaque function call via function pointer
NOINLINE std::uint64_t f_impl(std::uint64_t x) {
  // LCG-ish update, nontrivial but cheap
  return x * 6364136223846793005ULL + 1ULL;
}

// Measure TSC frequency by sleeping for a known wall time and measuring TSC delta.
// Returns GHz (cycles per nanosecond).
double measure_tsc_ghz(double sleep_seconds = 0.25, int trials = 7) {
  using clock = std::chrono::steady_clock;
  std::vector<double> ghz;
  ghz.reserve(trials);

  // Warm up a bit (helps stabilize turbo / scheduling jitter)
  for (volatile int i = 0; i < 1000000; ++i) {}

  for (int t = 0; t < trials; ++t) {
    auto w0 = clock::now();
    std::uint64_t c0 = rdtscp();

    std::this_thread::sleep_for(std::chrono::duration<double>(sleep_seconds));

    std::uint64_t c1 = rdtscp();
    auto w1 = clock::now();

    double wall_s = std::chrono::duration<double>(w1 - w0).count();
    double cycles = double(c1 - c0);

    // Hz = cycles / seconds; GHz = Hz / 1e9
    ghz.push_back((cycles / wall_s) / 1e9);
  }

  std::sort(ghz.begin(), ghz.end());
  return ghz[ghz.size() / 2]; // median
}

struct Sample {
  std::uint64_t min_cycles;
  std::uint64_t med_cycles;
  double cyc_per_iter;
  double ns_per_iter;
};

int main(int argc, char** argv) {
  std::uint64_t N = 100000000ULL;
  if (argc >= 2) N = std::stoull(argv[1]);
  std::cout << "N = " << N << " iterations\n";

  // Measure TSC GHz once up front
  double tsc_ghz = measure_tsc_ghz(0.25, 7);
  std::cout << "Estimated TSC frequency: " << tsc_ghz << " GHz\n\n";

  auto run = [&](const char* name, auto&& body) -> Sample {
    // Warmup
    body();

    std::vector<std::uint64_t> samples;
    samples.reserve(21);

    for (int t = 0; t < 21; ++t) {
      std::uint64_t t0 = rdtscp();
      body();
      std::uint64_t t1 = rdtscp();
      samples.push_back(t1 - t0);
    }

    std::sort(samples.begin(), samples.end());
    std::uint64_t mn  = samples.front();
    std::uint64_t med = samples[samples.size() / 2];

    double cpi = double(med) / double(N);
    // ns/iter = cycles/iter / (cycles/ns) = cycles/iter / GHz
    double ns_per_iter = cpi / tsc_ghz;

    std::cout << name
              << ": min " << mn << " cycles, median " << med << " cycles"
              << " => " << cpi << " cycles/iter"
              << " => " << ns_per_iter << " ns/iter\n";
    return {mn, med, cpi, ns_per_iter};
  };

  // 1) mul-add dependency loop
  auto mul_add_dep = [&]() {
    std::uint64_t x = g_sink;
    for (std::uint64_t i = 0; i < N; ++i) {
      x = x * 2862933555777941757ULL + 3037000493ULL;
      keep_live(x);
    }
    g_sink = x;
  };

  // 2) add-xor dependency loop
  auto add_xor_dep = [&]() {
    std::uint64_t x = g_sink;
    for (std::uint64_t i = 0; i < N; ++i) {
      x += (i ^ x);
      keep_live(x);
    }
    g_sink = x;
  };

  // 3) function pointer call per iter
  auto fnptr_call = [&]() {
    std::uint64_t x = g_sink;
    auto fn = &f_impl; // opaque-ish call target
    for (std::uint64_t i = 0; i < N; ++i) {
      x = fn(x);
      keep_live(x);
    }
    g_sink = x;
  };

  run("mul-add dep loop", mul_add_dep);
  run("add-xor dep loop", add_xor_dep);
  run("fnptr call loop ", fnptr_call);

  std::cout << "\n(sink=" << g_sink << ")\n";
  return 0;
}
