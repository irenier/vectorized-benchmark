#include "common.hpp"

static void expsum_eigen(benchmark::State &state) {
  const size_t n = state.range(0);
  auto data = generate_data(n);
  Eigen::Map<const Eigen::ArrayXd> v(data.data(), n);

  for (auto _ : state) {
    double res = v.exp().sum();
    benchmark::DoNotOptimize(res);
  }
  state.SetComplexityN(n);
}

static void expsum_highway(benchmark::State &state) {
  const size_t n = state.range(0);
  auto data = generate_data(n);
  hn::ScalableTag<double> d;
  const size_t lanes = hn::Lanes(d);

  for (auto _ : state) {
    auto acc = hn::Zero(d);
    size_t i = 0;
    for (; i + lanes <= n; i += lanes) {
      auto v = hn::LoadU(d, data.data() + i);
      acc = hn::Add(acc, hn::Exp(d, v));
    }
    double res = hn::ReduceSum(d, acc);
    // 剩余处理
    for (; i < n; ++i)
      res += std::exp(data[i]);
    benchmark::DoNotOptimize(res);
  }
  state.SetComplexityN(n);
}

static void expsum_xsimd(benchmark::State &state) {
  const size_t n = state.range(0);
  auto data = generate_data(n);
  using batch_type = xsimd::batch<double>;
  const size_t inc = batch_type::size;

  for (auto _ : state) {
    batch_type acc(0.0);
    size_t i = 0;
    for (; i + inc <= n; i += inc) {
      auto v = xsimd::load_unaligned(data.data() + i);
      acc += xsimd::exp(v);
    }
    double res = xsimd::reduce_add(acc);
    for (; i < n; ++i)
      res += std::exp(data[i]);
    benchmark::DoNotOptimize(res);
  }
  state.SetComplexityN(n);
}

static void expsum_omp(benchmark::State &state) {
  const size_t n = state.range(0);
  auto data = generate_data(n);

  for (auto _ : state) {
    double res = 0.0;
#pragma omp parallel for reduction(+ : res)
    for (size_t i = 0; i < n; ++i) {
      res += std::exp(data[i]);
    }
    benchmark::DoNotOptimize(res);
  }
  state.SetComplexityN(n);
}

static void expsum_tbb(benchmark::State &state) {
  const size_t n = state.range(0);
  auto data = generate_data(n);

  for (auto _ : state) {
    double res = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, n), 0.0,
        [&data](const tbb::blocked_range<size_t> &r, double init) {
          for (size_t i = r.begin(); i < r.end(); ++i) {
            init += std::exp(data[i]);
          }
          return init;
        },
        std::plus<double>());
    benchmark::DoNotOptimize(res);
  }
  state.SetComplexityN(n);
}

static void expsum_std(benchmark::State &state) {
  const size_t n = state.range(0);
  auto data = generate_data(n);

  for (auto _ : state) {
    double res = std::transform_reduce(data.begin(), data.end(), 0.0,
                                       std::plus<double>(),
                                       [](double x) { return std::exp(x); });
    benchmark::DoNotOptimize(res);
  }
  state.SetComplexityN(n);
}

BENCHMARK(expsum_eigen) BENCH_ARGS;
BENCHMARK(expsum_highway) BENCH_ARGS;
BENCHMARK(expsum_xsimd) BENCH_ARGS;
BENCHMARK(expsum_omp) BENCH_ARGS;
BENCHMARK(expsum_tbb) BENCH_ARGS;
BENCHMARK(expsum_std) BENCH_ARGS;

BENCHMARK_MAIN();