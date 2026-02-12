#include "common.hpp"

// DOT: result = sum(x[i] * y[i])

static void dot_eigen(benchmark::State &state) {
  const size_t n = state.range(0);
  auto x = generate_data(n);
  auto y = generate_data(n);

  Eigen::Map<const Eigen::VectorXd> vx(x.data(), n);
  Eigen::Map<const Eigen::VectorXd> vy(y.data(), n);

  for (auto _ : state) {
    double res = vx.dot(vy);
    benchmark::DoNotOptimize(res);
  }
  state.SetComplexityN(n);
}

static void dot_highway(benchmark::State &state) {
  const size_t n = state.range(0);
  auto x = generate_data(n);
  auto y = generate_data(n);

  hn::ScalableTag<double> d;
  const size_t lanes = hn::Lanes(d);

  for (auto _ : state) {
    auto sum_vec = hn::Zero(d);
    size_t i = 0;
    for (; i + lanes <= n; i += lanes) {
      auto vx = hn::LoadU(d, x.data() + i);
      auto vy = hn::LoadU(d, y.data() + i);
      sum_vec = hn::MulAdd(vx, vy, sum_vec);
    }
    double res = hn::ReduceSum(d, sum_vec);
    for (; i < n; ++i) {
      res += x[i] * y[i];
    }
    benchmark::DoNotOptimize(res);
  }
  state.SetComplexityN(n);
}

static void dot_xsimd(benchmark::State &state) {
  const size_t n = state.range(0);
  auto x = generate_data(n);
  auto y = generate_data(n);

  using batch_type = xsimd::batch<double>;
  const size_t inc = batch_type::size;

  for (auto _ : state) {
    auto vres = batch_type(0.0);
    size_t i = 0;
    for (; i + inc <= n; i += inc) {
      auto vx = xsimd::load_unaligned(x.data() + i);
      auto vy = xsimd::load_unaligned(y.data() + i);
      vres = xsimd::fma(vx, vy, vres); // vres += vx * vy
    }
    double res = xsimd::reduce_add(vres);
    for (; i < n; ++i) {
      res += x[i] * y[i];
    }
    benchmark::DoNotOptimize(res);
  }
  state.SetComplexityN(n);
}

static void dot_tbb(benchmark::State &state) {
  const size_t n = state.range(0);
  auto x = generate_data(n);
  auto y = generate_data(n);

  for (auto _ : state) {
    double res = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, n), 0.0,
        [&](const tbb::blocked_range<size_t> &r, double local_sum) {
          for (size_t i = r.begin(); i != r.end(); ++i) {
            local_sum += x[i] * y[i];
          }
          return local_sum;
        },
        std::plus<double>());
    benchmark::DoNotOptimize(res);
  }
  state.SetComplexityN(n);
}

static void dot_omp(benchmark::State &state) {
  const size_t n = state.range(0);
  auto x = generate_data(n);
  auto y = generate_data(n);

  for (auto _ : state) {
    double res = 0.0;
#pragma omp parallel for reduction(+ : res)
    for (size_t i = 0; i < n; ++i) {
      res += x[i] * y[i];
    }
    benchmark::DoNotOptimize(res);
  }
  state.SetComplexityN(n);
}

static void dot_std(benchmark::State &state) {
  const size_t n = state.range(0);
  auto x = generate_data(n);
  auto y = generate_data(n);

  for (auto _ : state) {
    double res =
        std::transform_reduce(x.begin(), x.end(), y.begin(), 0.0, std::plus<>(),
                              [](double a, double b) { return a * b; });
    benchmark::DoNotOptimize(res);
  }
  state.SetComplexityN(n);
}

BENCHMARK(dot_eigen) BENCH_ARGS;
BENCHMARK(dot_highway) BENCH_ARGS;
BENCHMARK(dot_xsimd) BENCH_ARGS;
BENCHMARK(dot_omp) BENCH_ARGS;
BENCHMARK(dot_tbb) BENCH_ARGS;
BENCHMARK(dot_std) BENCH_ARGS;

BENCHMARK_MAIN();