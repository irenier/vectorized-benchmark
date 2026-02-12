#pragma once

#include <Eigen/Eigen>
#include <algorithm>
#include <benchmark/benchmark.h>
#include <cmath>
#include <execution>
#include <hwy/contrib/math/math-inl.h>
#include <hwy/highway.h>
#include <numeric>
#include <omp.h>
#include <random>
#include <tbb/tbb.h>
#include <vector>
#include <xsimd/xsimd.hpp>

namespace hn = hwy::HWY_NAMESPACE;

// 生成随机数据
inline std::vector<double> generate_data(size_t n) {
  std::vector<double> v(n);
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dis(-1.0, 1.0);
  for (size_t i = 0; i < n; ++i) {
    v[i] = dis(gen);
  }
  return v;
}

// 统一的 Benchmark 参数
#define BENCH_ARGS                                                             \
  ->RangeMultiplier(10)                                                        \
      ->Range(100, 1000000000)                                                 \
      ->Unit(benchmark::kNanosecond)                                           \
      ->Complexity()