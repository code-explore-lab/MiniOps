// benchmarks/benchmark_matmul.cpp
//
// MatMul 多实现对比计时。串行实现默认单线程；omp 实现通过 --threads
// 指定 1,2,4,8,... 等线程数。
// 同一次运行内，各实现按轮转顺序交替测，尽量减小温度/频率漂移带来的偏差；
// 不同线程数之间同样轮转测量。计时区间内只调用算子，输入输出内存在计时前
// 一次性分配。默认统计中位数（median），同时输出均值和标准差。
//
// 用法：
//   benchmark_matmul [--repeats N] [--warmup N] [--sizes 64,128,...]
//                    [--impls name1,name2,...] [--threads 1,2,4,8]
//                    [--csv path.csv]
//
// 不指定 --repeats 时自动选择重复次数：以最慢实现约 500 ms 的采样总时长为
// 目标，最少 7 次、最多 2001 次。同一规模下所有实现和线程数使用相同的
// 重复次数。
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

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kMinRepeats = 7;
constexpr int kMaxRepeats = 2001;
constexpr double kTargetSampleMs = 500.0;

#ifdef _OPENMP
constexpr bool kOpenMpEnabled = true;
#else
constexpr bool kOpenMpEnabled = false;
#endif

using MatmulFn = void (*)(const float *, const float *, float *, int, int, int);

struct Impl {
  const char *name;
  MatmulFn fn;
  bool uses_omp;
};

// 若选中 naive，则它作为每个规模的加速比基线；否则以选中的第一个为基线。
const Impl kImpls[] = {
    {"naive", &miniops::matmul, false},
    {"ikj", &miniops::matmul_ikj, false},
    {"blocking", &miniops::matmul_blocking, false},
    {"blocking_ptr", &miniops::matmul_blocking_ptr, false},
    {"reg4", &miniops::matmul_reg4, false},
    {"reg8", &miniops::matmul_reg8, false},
    {"reg16", &miniops::matmul_reg16, false},
    {"rows2", &miniops::matmul_rows2, false},
    {"rows4", &miniops::matmul_rows4, false},
    {"rows8", &miniops::matmul_rows8, false},
    {"omp", &miniops::matmul_omp, true},
    {"omp_ikj", &miniops::matmul_omp_ikj, true},
};
constexpr int kImplCount = static_cast<int>(sizeof(kImpls) / sizeof(kImpls[0]));

struct CaseResult {
  const char *version;
  int threads;
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

CaseResult summarize(const Impl &impl, int threads, int size, int repeats,
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

  return CaseResult{impl.name, threads, size, size, size, repeats, median_ms,
                    mean_ms, min_ms, std_ms, gflops, speedup, checksum};
}

// M = N = K = size，选中的实现交替测量；omp 实现额外按 threads 列表测量。
std::vector<CaseResult> run_size(int size, int warmup, int repeats,
                                 bool auto_repeats,
                                 const std::vector<int> &impls,
                                 const std::vector<int> &threads) {
  const int n_impls = static_cast<int>(impls.size());
  const std::size_t elems = static_cast<std::size_t>(size) * size;
  std::vector<float> a(elems);
  std::vector<float> b(elems);
  fill_random(a, 12345u + static_cast<std::uint32_t>(size));
  fill_random(b, 67890u + static_cast<std::uint32_t>(size));

  // 每个实现独立输出，避免互相覆盖，也便于最后统一校验。
  std::vector<std::vector<float>> c(n_impls);
  for (auto &buf : c) {
    buf.assign(elems, 0.0f);
  }

#ifdef _OPENMP
  // 串行组固定为单线程，不受外部 OMP_NUM_THREADS 影响。
  omp_set_num_threads(1);
#endif

  // 预热：让缓存、频率和分支预测进入稳定状态，不计入结果。
  for (int w = 0; w < warmup; ++w) {
    for (int k = 0; k < n_impls; ++k) {
      kImpls[impls[k]].fn(a.data(), b.data(), c[k].data(), size, size, size);
    }
  }

  if (auto_repeats) {
    // 用最慢实现的试跑时间决定重复次数，保证每个实现都有足够样本。
    double slowest_ms = 0.0;
    for (int k = 0; k < n_impls; ++k) {
      const auto start = Clock::now();
      kImpls[impls[k]].fn(a.data(), b.data(), c[k].data(), size, size, size);
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

  std::vector<std::vector<double>> samples(n_impls);
  for (auto &s : samples) {
    s.reserve(static_cast<std::size_t>(repeats));
  }

  // 轮转顺序：每轮把上一个实现排到最后，避免某个实现总是处于固定缓存状态。
  for (int r = 0; r < repeats; ++r) {
    for (int t = 0; t < n_impls; ++t) {
      const int k = (r + t) % n_impls;
      const auto start = Clock::now();
      kImpls[impls[k]].fn(a.data(), b.data(), c[k].data(), size, size, size);
      const auto end = Clock::now();
      samples[k].push_back(to_ms(end - start));
    }
  }

  // 在计时区间外读取结果，既做健全性检查，也防止结果被完全丢弃。
  std::vector<double> checksums(n_impls, 0.0);
  for (int k = 0; k < n_impls; ++k) {
    bool finite = true;
    for (float x : c[k]) {
      checksums[k] += x;
      finite = finite && std::isfinite(x);
    }
    if (!finite) {
      std::fprintf(stderr, "[ERROR] %s %dx%d result contains NaN/Inf\n",
                   kImpls[impls[k]].name, size, size);
      std::exit(1);
    }
  }

  // 基线：优先使用选中的 naive，否则使用选中的第一个实现。
  int base = 0;
  for (int k = 0; k < n_impls; ++k) {
    if (std::strcmp(kImpls[impls[k]].name, "naive") == 0) {
      base = k;
      break;
    }
  }
  const double baseline_ms = median_of(samples[base]);

  // 线程扩展：对每个被选中的 omp 实现，分别测 threads 列表里 >1 的线程数。
  // t == 1 的结果已由上面的串行组覆盖。
  std::vector<int> extra_threads;
  for (int t : threads) {
    if (t != 1) {
      extra_threads.push_back(t);
    }
  }
  std::vector<int> omp_positions;
  for (int k = 0; k < n_impls; ++k) {
    if (kImpls[impls[k]].uses_omp) {
      omp_positions.push_back(k);
    }
  }

  std::vector<CaseResult> results;
  results.reserve(n_impls + omp_positions.size() * extra_threads.size());
  for (int k = 0; k < n_impls; ++k) {
    results.push_back(summarize(kImpls[impls[k]], 1, size, repeats, samples[k],
                                baseline_ms, checksums[k]));
  }
  if (omp_positions.empty() || extra_threads.empty()) {
    return results;
  }

#ifdef _OPENMP
  struct ThreadConfig {
    int impl_pos;
    int threads;
  };
  std::vector<ThreadConfig> configs;
  configs.reserve(omp_positions.size() * extra_threads.size());
  for (int pos : omp_positions) {
    for (int t : extra_threads) {
      configs.push_back(ThreadConfig{pos, t});
    }
  }
  const int n_configs = static_cast<int>(configs.size());

  // 每个配置先预热一次，避免线程池创建计入计时。
  for (const ThreadConfig &cfg : configs) {
    omp_set_num_threads(cfg.threads);
    const Impl &impl = kImpls[impls[cfg.impl_pos]];
    impl.fn(a.data(), b.data(), c[cfg.impl_pos].data(), size, size, size);
  }

  std::vector<std::vector<double>> tsamples(configs.size());
  for (auto &s : tsamples) {
    s.reserve(static_cast<std::size_t>(repeats));
  }

  // 实现与线程数之间轮转测量，减小频率/缓存状态漂移。
  for (int r = 0; r < repeats; ++r) {
    for (int q = 0; q < n_configs; ++q) {
      const int idx = (r + q) % n_configs;
      const ThreadConfig &cfg = configs[idx];
      const Impl &impl = kImpls[impls[cfg.impl_pos]];
      omp_set_num_threads(cfg.threads);
      const auto start = Clock::now();
      impl.fn(a.data(), b.data(), c[cfg.impl_pos].data(), size, size, size);
      const auto end = Clock::now();
      tsamples[idx].push_back(to_ms(end - start));
    }
  }
  omp_set_num_threads(1);

  // 每个配置再跑一次（计时外）校验结果，与同一实现单线程 checksum 对比。
  // 行划分不改变每个输出元素的累加顺序，正常情况下应完全一致。
  for (int q = 0; q < n_configs; ++q) {
    const ThreadConfig &cfg = configs[q];
    const Impl &impl = kImpls[impls[cfg.impl_pos]];
    omp_set_num_threads(cfg.threads);
    impl.fn(a.data(), b.data(), c[cfg.impl_pos].data(), size, size, size);

    const double ref_checksum = checksums[cfg.impl_pos];
    const double checksum_scale = std::max(1.0, std::fabs(ref_checksum));
    double sum = 0.0;
    bool finite = true;
    for (float x : c[cfg.impl_pos]) {
      sum += x;
      finite = finite && std::isfinite(x);
    }
    if (!finite || std::fabs(sum - ref_checksum) > 1e-4 * checksum_scale) {
      std::fprintf(stderr,
                   "[ERROR] %s %dx%d threads=%d result mismatch "
                   "(checksum %.6f vs %.6f)\n",
                   impl.name, size, size, cfg.threads, sum, ref_checksum);
      std::exit(1);
    }
    results.push_back(summarize(impl, cfg.threads, size, repeats, tsamples[q],
                                baseline_ms, sum));
  }
  omp_set_num_threads(1);
#else
  (void)omp_positions;
  std::fprintf(stderr,
               "[WARN] --threads > 1 ignored: built without OpenMP\n");
#endif
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

std::vector<int> parse_impls(const char *text) {
  std::vector<int> idxs;
  const char *p = text;
  while (true) {
    const char *comma = std::strchr(p, ',');
    const std::size_t len =
        comma != nullptr ? static_cast<std::size_t>(comma - p) : std::strlen(p);
    bool found = false;
    for (int i = 0; i < kImplCount; ++i) {
      if (std::strlen(kImpls[i].name) == len &&
          std::strncmp(kImpls[i].name, p, len) == 0) {
        idxs.push_back(i);
        found = true;
        break;
      }
    }
    if (len == 0 || !found) {
      std::fprintf(stderr, "[ERROR] unknown impl in list: %s\n", text);
      std::exit(2);
    }
    if (comma == nullptr) {
      break;
    }
    p = comma + 1;
  }
  return idxs;
}

std::vector<int> parse_threads(const char *text) {
  std::vector<int> threads;
  const char *p = text;
  while (*p != '\0') {
    char *end = nullptr;
    const long value = std::strtol(p, &end, 10);
    if (end == p || value <= 0 || value > 1024) {
      std::fprintf(stderr, "[ERROR] invalid thread list: %s\n", text);
      std::exit(2);
    }
    threads.push_back(static_cast<int>(value));
    p = end;
    if (*p == ',') {
      ++p;
    } else if (*p != '\0') {
      std::fprintf(stderr, "[ERROR] invalid thread list: %s\n", text);
      std::exit(2);
    }
  }
  if (threads.empty()) {
    std::fprintf(stderr, "[ERROR] empty thread list\n");
    std::exit(2);
  }
  return threads;
}

void print_usage(const char *argv0) {
  std::printf(
      "Usage: %s [--repeats N] [--warmup N] [--sizes 64,128,...]\n"
      "          [--impls name1,name2,...] [--threads 1,2,4,8] [--csv FILE]\n"
      "  --repeats  fixed number of timed runs per size (default: auto)\n"
      "             auto targets ~500 ms of the slowest impl, 7 to 2001 runs\n"
      "  --warmup   number of untimed warm-up runs per size (default: 2)\n"
      "  --sizes    comma-separated matrix sizes, M=N=K (default: "
      "64,128,256,512,1024)\n"
      "  --impls    subset of implementations to run (default: all)\n"
      "             naive,ikj,blocking,blocking_ptr,reg4,reg8,reg16,"
      "rows2,rows4,rows8,omp,omp_ikj\n"
      "  --threads  OpenMP thread counts for the omp impl (default: 1)\n"
      "             requires an OpenMP build; serial impls always use 1\n"
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
                 r.version, r.threads, r.m, r.median_ms, r.mean_ms, r.min_ms,
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
  std::vector<int> impls;
  for (int i = 0; i < kImplCount; ++i) {
    impls.push_back(i);
  }
  std::vector<int> threads = {1};
  const char *csv_path = nullptr;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
      repeats = std::atoi(argv[++i]);
      auto_repeats = false;
    } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      warmup = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--sizes") == 0 && i + 1 < argc) {
      sizes = parse_sizes(argv[++i]);
    } else if (std::strcmp(argv[i], "--impls") == 0 && i + 1 < argc) {
      impls = parse_impls(argv[++i]);
    } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      threads = parse_threads(argv[++i]);
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

  bool extra_threads = false;
  for (int t : threads) {
    extra_threads = extra_threads || (t != 1);
  }
  if (extra_threads && !kOpenMpEnabled) {
    std::fprintf(stderr,
                 "[ERROR] --threads > 1 requires an OpenMP build "
                 "(configure with -DMINIOPS_ENABLE_OPENMP=ON)\n");
    return 2;
  }
  bool omp_selected = false;
  for (int idx : impls) {
    omp_selected = omp_selected || kImpls[idx].uses_omp;
  }
  if (extra_threads && !omp_selected) {
    std::fprintf(stderr,
                 "[WARN] --threads ignored: 'omp' is not in --impls\n");
  }

  std::printf("MiniOps MatMul Benchmark\n");
  std::printf("versions=");
  for (std::size_t i = 0; i < impls.size(); ++i) {
    std::printf("%s%s", i == 0 ? "" : ",", kImpls[impls[i]].name);
  }
  std::printf(" threads=");
  for (std::size_t i = 0; i < threads.size(); ++i) {
    std::printf("%s%d", i == 0 ? "" : ",", threads[i]);
  }
  std::printf(" layout=row-major dtype=float openmp=%s",
              kOpenMpEnabled ? "on" : "off");
#ifdef _OPENMP
  std::printf(" omp_version=%d", _OPENMP);
#endif
  std::printf("\n");
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
  results.reserve(sizes.size() * (impls.size() + threads.size()));
  for (int size : sizes) {
    std::fprintf(stderr, "[run] %dx%dx%d ...\n", size, size, size);
    std::vector<CaseResult> case_results =
        run_size(size, warmup, repeats, auto_repeats, impls, threads);
    for (const CaseResult &r : case_results) {
      results.push_back(r);
      std::printf("| %s | %d | %d | %d | %.3f | %.3f | %.3f | %.3f | %.3f | "
                  "%.3f | %.6f |\n",
                  r.version, r.m, r.threads, r.repeats, r.median_ms, r.mean_ms,
                  r.min_ms, r.std_ms, r.gflops, r.speedup, r.checksum);
    }
    std::fflush(stdout);
  }

  if (csv_path != nullptr && !write_csv(csv_path, results, warmup)) {
    return 1;
  }

  return 0;
}
