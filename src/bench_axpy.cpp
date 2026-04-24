#include "common.hpp"

// AXPY: Y = alpha * X + Y

const double alpha = 2.5;

static void axpy_eigen(benchmark::State &state) {
  const size_t n = state.range(0);
  auto x = generate_data(n);
  auto y = generate_data(n);

  // Eigen Map 避免数据拷贝
  Eigen::Map<const Eigen::VectorXd> vx(x.data(), n);
  Eigen::Map<Eigen::VectorXd> vy(y.data(), n);

  for (auto _ : state) {
    vy += alpha * vx;
    benchmark::DoNotOptimize(y.data());
  }
  state.SetComplexityN(n);
}

static void axpy_highway(benchmark::State &state) {
  const size_t n = state.range(0);
  auto x = generate_data(n);
  auto y = generate_data(n);

  hn::ScalableTag<double> d;
  const size_t lanes = hn::Lanes(d);

  for (auto _ : state) {
    const auto alpha_val = hn::Set(d, alpha);

    size_t i = 0;
    for (; i + lanes <= n; i += lanes) {
      auto vx = hn::LoadU(d, x.data() + i);
      auto vy = hn::LoadU(d, y.data() + i);
      auto res = hn::MulAdd(alpha_val, vx, vy);
      hn::StoreU(res, d, y.data() + i);
    }
    for (; i < n; ++i) {
      y[i] += alpha * x[i];
    }
    benchmark::DoNotOptimize(y.data());
  }
  state.SetComplexityN(n);
}

static void axpy_xsimd(benchmark::State &state) {
  const size_t n = state.range(0);
  auto x = generate_data(n);
  auto y = generate_data(n);

  using batch_type = xsimd::batch<double>;
  const size_t inc = batch_type::size;

  for (auto _ : state) {
    const auto alpha_val = batch_type(alpha);

    size_t i = 0;
    for (; i + inc <= n; i += inc) {
      auto vx = xsimd::load_unaligned(x.data() + i);
      auto vy = xsimd::load_unaligned(y.data() + i);
      auto res = xsimd::fma(alpha_val, vx, vy);
      res.store_unaligned(y.data() + i);
    }
    for (; i < n; ++i) {
      y[i] += alpha * x[i];
    }
    benchmark::DoNotOptimize(y.data());
  }
  state.SetComplexityN(n);
}

static void axpy_omp(benchmark::State &state) {
  const size_t n = state.range(0);
  auto x = generate_data(n);
  auto y = generate_data(n);

  for (auto _ : state) {
#pragma omp parallel for simd
    for (size_t i = 0; i < n; ++i) {
      y[i] += alpha * x[i];
    }
    benchmark::DoNotOptimize(y.data());
  }
  state.SetComplexityN(n);
}

static void axpy_tbb(benchmark::State &state) {
  const size_t n = state.range(0);
  auto x = generate_data(n);
  auto y = generate_data(n);

  for (auto _ : state) {
    tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
                      [&x, &y](const tbb::blocked_range<size_t> &r) {
                        for (size_t i = r.begin(); i < r.end(); ++i) {
                          y[i] += alpha * x[i];
                        }
                      });
    benchmark::DoNotOptimize(y.data());
  }
  state.SetComplexityN(n);
}

static void axpy_std(benchmark::State &state) {
  const size_t n = state.range(0);
  auto x = generate_data(n);
  auto y = generate_data(n);

  for (auto _ : state) {
    std::transform(std::execution::par_unseq, y.begin(), y.end(), x.begin(),
                   y.begin(),
                   [](double yi, double xi) { return yi + alpha * xi; });
    benchmark::DoNotOptimize(y.data());
  }
  state.SetComplexityN(n);
}

BENCHMARK(axpy_eigen) BENCH_ARGS;
BENCHMARK(axpy_highway) BENCH_ARGS;
BENCHMARK(axpy_xsimd) BENCH_ARGS;
BENCHMARK(axpy_omp) BENCH_ARGS;
BENCHMARK(axpy_tbb) BENCH_ARGS;
BENCHMARK(axpy_std) BENCH_ARGS;

BENCHMARK_MAIN();