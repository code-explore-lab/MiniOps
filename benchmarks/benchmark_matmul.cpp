// benchmarks/benchmark_matmul.cpp
//
// MatMul 多实现对比计时（均为单线程）。
// 同一次运行内，各实现按轮转顺序交替测，尽量减小温度/频率漂移带来的偏差；
// 计时区间内只调用算子，输入输出内存在计时前一次性分配。
// 默认统计中位数（median），同时输出均值和标准差用于判断稳定性。
//
// 用法：
//   benchmark_matmul [--repeats N] [--warmup N] [--sizes 64,128,...]
//                    [--csv path.csv]
//
// 不指定 --repeats 时自动选择重复次数：以最慢实现约 500 ms 的采样总时长为
// 目标，最少 7 次、最多 2001 次。所有实现使用相同的重复次数。
//
// 必须使用 Release 构建，否则数据没有参考价值。

#include "miniops/operators.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kThreads = 1;
constexpr int kMinRepeats = 7;
constexpr int kMaxRepeats = 2001;
constexpr double kTargetSampleMs = 500.0;

using MatmulFn = void (*)(const float *, const float *, float *, int, int, int);

struct Impl {
  const char *name;
  MatmulFn fn;
};

// 第一项必须是 naive，它作为每个规模的加速比基线。
const Impl kImpls[] = {
    {"naive", &miniops::matmul},
    {"ikj", &miniops::matmul_ikj},
    {"blocking", &miniops::matmul_blocking},
};
constexpr int kImplCount = static_cast<int>(sizeof(kImpls) / sizeof(kImpls[0]));

struct CaseResult {
  const char *version;
  int m, n, k;
  int repeats;
  double median_ms;
  double mean_ms;
  double min_ms;
  double std_ms;
  double gflops;
  double speedup;
  double checksum;
};

double to_ms(Clock::duration d) {
  return std::chrono::duration<double, std::milli>(d).count();
}

double median_of(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  const std::size_t n = samples.size();
  if (n % 2 == 1) {
    return samples[n / 2];
  }
  return 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
}

// 确定性伪随机填充，取值约在 [-1, 1)，避免全零/非规格化数影响计时。
void fill_random(std::vector<float> &v, std::uint32_t seed) {
  std::uint32_t state = seed;
  for (float &x : v) {
    state = state * 1664525u + 1013904223u;
    // 取高 24 位映射到 [-1, 1)
    const std::uint32_t mantissa = (state >> 8) & 0xFFFFFFu;
    x = static_cast<float>(mantissa) * (2.0f / 16777216.0f) - 1.0f;
  }
}

CaseResult summarize(const Impl &impl, int size, int repeats,
                     const std::vector<double> &samples, double baseline_ms,
                     double checksum) {
  const double median_ms = median_of(samples);
  const double mean_ms =
      std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  const double min_ms = *std::min_element(samples.begin(), samples.end());

  double variance = 0.0;
  for (double t : samples) {
    variance += (t - mean_ms) * (t - mean_ms);
  }
  variance /= samples.size();
  const double std_ms = std::sqrt(variance);

  // 2 * M * N * K 次浮点运算；毫秒换算成秒。
  const double gflops =
      (2.0 * size * size * size) / (median_ms * 1e-3) / 1e9;
  const double speedup = baseline_ms / median_ms;

  return CaseResult{impl.name, size, size, size, repeats, median_ms,
                    mean_ms, min_ms, std_ms, gflops, speedup, checksum};
}

// M = N = K = size，所有实现交替测量。
std::vector<CaseResult> run_size(int size, int warmup, int repeats,
                                 bool auto_repeats) {
  const std::size_t elems = static_cast<std::size_t>(size) * size;
  std::vector<float> a(elems);
  std::vector<float> b(elems);
  fill_random(a, 12345u + static_cast<std::uint32_t>(size));
  fill_random(b, 67890u + static_cast<std::uint32_t>(size));

  // 每个实现独立输出，避免互相覆盖，也便于最后统一校验。
  std::vector<std::vector<float>> c(kImplCount);
  for (auto &buf : c) {
    buf.assign(elems, 0.0f);
  }

  // 预热：让缓存、频率和分支预测进入稳定状态，不计入结果。
  for (int w = 0; w < warmup; ++w) {
    for (int i = 0; i < kImplCount; ++i) {
      kImpls[i].fn(a.data(), b.data(), c[i].data(), size, size, size);
    }
  }

  if (auto_repeats) {
    // 用最慢实现的试跑时间决定重复次数，保证每个实现都有足够样本。
    double slowest_ms = 0.0;
    for (int i = 0; i < kImplCount; ++i) {
      const auto start = Clock::now();
      kImpls[i].fn(a.data(), b.data(), c[i].data(), size, size, size);
      const auto end = Clock::now();
      slowest_ms = std::max(slowest_ms, to_ms(end - start));
    }
    const double wanted = kTargetSampleMs / slowest_ms;
    if (wanted < kMinRepeats) {
      repeats = kMinRepeats;
    } else if (wanted > kMaxRepeats) {
      repeats = kMaxRepeats;
    } else {
      repeats = static_cast<int>(wanted + 0.5);
    }
  }

  std::vector<std::vector<double>> samples(kImplCount);
  for (auto &s : samples) {
    s.reserve(static_cast<std::size_t>(repeats));
  }

  // 轮转顺序：每轮把上一个实现排到最后，避免某个实现总是处于固定缓存状态。
  for (int r = 0; r < repeats; ++r) {
    for (int t = 0; t < kImplCount; ++t) {
      const int i = (r + t) % kImplCount;
      const auto start = Clock::now();
      kImpls[i].fn(a.data(), b.data(), c[i].data(), size, size, size);
      const auto end = Clock::now();
      samples[i].push_back(to_ms(end - start));
    }
  }

  // 在计时区间外读取结果，既做健全性检查，也防止结果被完全丢弃。
  std::vector<double> checksums(kImplCount, 0.0);
  for (int i = 0; i < kImplCount; ++i) {
    bool finite = true;
    for (float x : c[i]) {
      checksums[i] += x;
      finite = finite && std::isfinite(x);
    }
    if (!finite) {
      std::fprintf(stderr, "[ERROR] %s %dx%d result contains NaN/Inf\n",
                   kImpls[i].name, size, size);
      std::exit(1);
    }
  }

  // kImpls[0]（naive）是加速比基线。
  const double baseline_ms = median_of(samples[0]);
  std::vector<CaseResult> results;
  results.reserve(kImplCount);
  for (int i = 0; i < kImplCount; ++i) {
    results.push_back(
        summarize(kImpls[i], size, repeats, samples[i], baseline_ms, checksums[i]));
  }
  return results;
}

std::vector<int> parse_sizes(const char *text) {
  std::vector<int> sizes;
  const char *p = text;
  while (*p != '\0') {
    char *end = nullptr;
    const long value = std::strtol(p, &end, 10);
    if (end == p || value <= 0 || value > 65536) {
      std::fprintf(stderr, "[ERROR] invalid size list: %s\n", text);
      std::exit(2);
    }
    sizes.push_back(static_cast<int>(value));
    p = end;
    if (*p == ',') {
      ++p;
    } else if (*p != '\0') {
      std::fprintf(stderr, "[ERROR] invalid size list: %s\n", text);
      std::exit(2);
    }
  }
  if (sizes.empty()) {
    std::fprintf(stderr, "[ERROR] empty size list\n");
    std::exit(2);
  }
  return sizes;
}

void print_usage(const char *argv0) {
  std::printf(
      "Usage: %s [--repeats N] [--warmup N] [--sizes 64,128,...] [--csv FILE]\n"
      "  --repeats  fixed number of timed runs per size (default: auto)\n"
      "             auto targets ~500 ms of the slowest impl, 7 to 2001 runs\n"
      "  --warmup   number of untimed warm-up runs per size (default: 2)\n"
      "  --sizes    comma-separated matrix sizes, M=N=K (default: "
      "64,128,256,512,1024)\n"
      "  --csv      also write results to a CSV file\n",
      argv0);
}

bool write_csv(const char *path, const std::vector<CaseResult> &results,
               int warmup) {
  std::FILE *fp = std::fopen(path, "w");
  if (fp == nullptr) {
    std::fprintf(stderr, "[ERROR] cannot open CSV file: %s\n", path);
    return false;
  }
  std::fprintf(fp,
               "version,threads,size,median_ms,mean_ms,min_ms,std_ms,gflops,"
               "speedup,checksum,repeats,warmup\n");
  for (const CaseResult &r : results) {
    std::fprintf(fp, "%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%.4f,%.3f,%.6f,%d,%d\n",
                 r.version, kThreads, r.m, r.median_ms, r.mean_ms, r.min_ms,
                 r.std_ms, r.gflops, r.speedup, r.checksum, r.repeats, warmup);
  }
  std::fclose(fp);
  return true;
}

} // namespace

int main(int argc, char **argv) {
  int repeats = kMinRepeats;
  bool auto_repeats = true;
  int warmup = 2;
  std::vector<int> sizes = {64, 128, 256, 512, 1024};
  const char *csv_path = nullptr;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
      repeats = std::atoi(argv[++i]);
      auto_repeats = false;
    } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      warmup = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--sizes") == 0 && i + 1 < argc) {
      sizes = parse_sizes(argv[++i]);
    } else if (std::strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
      csv_path = argv[++i];
    } else if (std::strcmp(argv[i], "--help") == 0 ||
               std::strcmp(argv[i], "-h") == 0) {
      print_usage(argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "[ERROR] unknown argument: %s\n", argv[i]);
      print_usage(argv[0]);
      return 2;
    }
  }

  if ((!auto_repeats && repeats < 1) || warmup < 0) {
    std::fprintf(stderr, "[ERROR] repeats must be >= 1 and warmup >= 0\n");
    return 2;
  }

  std::printf("MiniOps MatMul Benchmark\n");
  std::printf("versions=");
  for (int i = 0; i < kImplCount; ++i) {
    std::printf("%s%s", i == 0 ? "" : ",", kImpls[i].name);
  }
  std::printf(" threads=%d layout=row-major dtype=float\n", kThreads);
  if (auto_repeats) {
    std::printf("warmup=%d repeats=auto(target %.0f ms, %d..%d) "
                "statistic=median\n",
                warmup, kTargetSampleMs, kMinRepeats, kMaxRepeats);
  } else {
    std::printf("warmup=%d repeats=%d statistic=median\n", warmup, repeats);
  }

#ifndef NDEBUG
  std::fprintf(stderr,
               "[WARN] NDEBUG is not defined: this is not a Release build, "
               "timings are not valid for reports.\n");
#endif

  std::printf("\n| version | size | threads | repeats | median_ms | mean_ms | "
              "min_ms | std_ms | GFLOP/s | speedup | checksum |\n");
  std::printf("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
              "| ---: | ---: |\n");

  std::vector<CaseResult> results;
  results.reserve(sizes.size() * kImplCount);
  for (int size : sizes) {
    std::fprintf(stderr, "[run] %dx%dx%d ...\n", size, size, size);
    std::vector<CaseResult> case_results =
        run_size(size, warmup, repeats, auto_repeats);
    for (const CaseResult &r : case_results) {
      results.push_back(r);
      std::printf("| %s | %d | %d | %d | %.3f | %.3f | %.3f | %.3f | %.3f | "
                  "%.3f | %.6f |\n",
                  r.version, r.m, kThreads, r.repeats, r.median_ms, r.mean_ms,
                  r.min_ms, r.std_ms, r.gflops, r.speedup, r.checksum);
    }
    std::fflush(stdout);
  }

  if (csv_path != nullptr && !write_csv(csv_path, results, warmup)) {
    return 1;
  }

  return 0;
}
