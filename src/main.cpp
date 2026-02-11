// #include <armadillo>
#include <Eigen/Eigen>
#include <benchmark/benchmark.h>
#include <cmath>
#include <hwy/contrib/math/math-inl.h>
#include <hwy/highway.h>
#include <numeric>
#include <omp.h>
#include <random>
#include <tbb/tbb.h>
#include <vector>
#include <xsimd/xsimd.hpp>

namespace hn = hwy::HWY_NAMESPACE;

// 辅助函数：根据 N 生成随机数据
std::vector<double> generate_data(size_t n) {
  std::vector<double> v(n);
  std::mt19937 gen(42); // 使用固定种子保证可重复性
  std::uniform_real_distribution<double> dis(-1.0, 1.0);
  for (size_t i = 0; i < n; ++i)
    v[i] = dis(gen);
  return v;
}

static void vec_exp_reduce_eigen(benchmark::State &state) {
  const size_t n = state.range(0);
  auto data = generate_data(n);
  Eigen::Map<const Eigen::ArrayXd> v1(data.data(), data.size());

  for (auto _ : state) {
    double result = v1.exp().sum();
    benchmark::DoNotOptimize(result);
  }
  state.SetComplexityN(n);
}

// static void vec_exp_reduce_armadillo(benchmark::State &state) {
//   const size_t n = state.range(0);
//   auto data = generate_data(n);
//   arma::vec v1(data.data(), data.size());

//   for (auto _ : state) {
//     double result = arma::sum(arma::exp(v1));
//     benchmark::DoNotOptimize(result);
//   }
//   state.SetComplexityN(n);
// }

static void vec_exp_reduce_highway(benchmark::State &state) {
  const size_t n = state.range(0);
  auto data = generate_data(n);
  hn::ScalableTag<double> d;
  const size_t lanes = hn::Lanes(d);

  for (auto _ : state) {
    double result = 0;
    size_t i = 0;
    for (; i + lanes <= n; i += lanes) {
      result += hn::ReduceSum(d, hn::Exp(d, hn::Load(d, data.data() + i)));
    }
    // 处理剩余元素
    for (; i < n; ++i)
      result += std::exp(data[i]);
    benchmark::DoNotOptimize(result);
  }
  state.SetComplexityN(n);
}

static void vec_exp_reduce_xsimd(benchmark::State &state) {
  const size_t n = state.range(0);
  auto data = generate_data(n);

  using batch_type = xsimd::batch<double>;
  const size_t inc = batch_type::size;
  for (auto _ : state) {
    double result = 0;
    size_t i = 0;
    for (; i + inc <= n; i += inc) {
      auto va = xsimd::load_unaligned(data.data() + i);
      result += xsimd::reduce_add(xsimd::exp(xsimd::exp(va)));
    }
    for (; i < n; ++i) {
      result += std::exp(data[i]);
    }
    benchmark::DoNotOptimize(result);
  }
}

static void vec_exp_reduce_tbb(benchmark::State &state) {
  const size_t n = state.range(0);
  auto data = generate_data(n);

  for (auto _ : state) {
    double sum = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, n), 0.0,
        [&data](const tbb::blocked_range<size_t> &r, double local_sum) {
          for (size_t i = r.begin(); i < r.end(); ++i) {
            local_sum += std::exp(data[i]);
          }
          return local_sum;
        },
        std::plus<double>());
    benchmark::DoNotOptimize(sum);
  }
  state.SetComplexityN(n);
}

static void vec_exp_reduce_omp(benchmark::State &state) {
  const size_t n = state.range(0);
  auto data = generate_data(n);

  for (auto _ : state) {
    double sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
    for (size_t i = 0; i < n; ++i) {
      sum += std::exp(data[i]);
    }
  }
  state.SetComplexityN(n);
}

static void vec_exp_reduce_std(benchmark::State &state) {
  const size_t n = state.range(0);
  auto data = generate_data(n);

  for (auto _ : state) {
    double result =
        std::transform_reduce(data.begin(), data.end(), 0.0, std::plus<>(),
                              [](double x) { return std::exp(x); });
    benchmark::DoNotOptimize(result);
  }
  state.SetComplexityN(n);
}

#define ARGS                                                                   \
  ->RangeMultiplier(10)                                                        \
      ->Range(10, 1000000000)                                                  \
      ->Unit(benchmark::kNanosecond)                                           \
      ->Complexity()

BENCHMARK(vec_exp_reduce_eigen) ARGS;
// BENCHMARK(vec_exp_reduce_armadillo) ARGS;
BENCHMARK(vec_exp_reduce_highway) ARGS;
BENCHMARK(vec_exp_reduce_xsimd) ARGS;
BENCHMARK(vec_exp_reduce_tbb) ARGS;
BENCHMARK(vec_exp_reduce_omp) ARGS;
BENCHMARK(vec_exp_reduce_std) ARGS;

BENCHMARK_MAIN();