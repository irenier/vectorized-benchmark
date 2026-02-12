#include "common.hpp"

// NRM2: result = sqrt(sum(x[i]^2))

static void nrm2_eigen(benchmark::State &state) {
  const size_t n = state.range(0);
  auto x = generate_data(n);
  Eigen::Map<const Eigen::VectorXd> vx(x.data(), n);

  for (auto _ : state) {
    double res = vx.norm();
    benchmark::DoNotOptimize(res);
  }
  state.SetComplexityN(n);
}

static void nrm2_highway(benchmark::State &state) {
  const size_t n = state.range(0);
  auto x = generate_data(n);

  hn::ScalableTag<double> d;
  const size_t lanes = hn::Lanes(d);

  for (auto _ : state) {
    auto sum_vec = hn::Zero(d);
    size_t i = 0;
    for (; i + lanes <= n; i += lanes) {
      auto vx = hn::LoadU(d, x.data() + i);
      sum_vec = hn::MulAdd(vx, vx, sum_vec);
    }
    double sum = hn::ReduceSum(d, sum_vec);
    for (; i < n; ++i) {
      sum += x[i] * x[i];
    }
    double res = std::sqrt(sum);
    benchmark::DoNotOptimize(res);
  }
  state.SetComplexityN(n);
}

static void nrm2_xsimd(benchmark::State &state) {
  const size_t n = state.range(0);
  auto x = generate_data(n);

  using batch_type = xsimd::batch<double>;
  const size_t inc = batch_type::size;

  for (auto _ : state) {
    auto vres = batch_type(0.0);
    size_t i = 0;
    for (; i + inc <= n; i += inc) {
      auto vx = xsimd::load_unaligned(x.data() + i);
      vres = xsimd::fma(vx, vx, vres); // vres += vx * vx
    }
    double sum = xsimd::reduce_add(vres);
    for (; i < n; ++i) {
      sum += x[i] * x[i];
    }
    double res = std::sqrt(sum);
    benchmark::DoNotOptimize(res);
  }
  state.SetComplexityN(n);
}

static void nrm2_omp(benchmark::State &state) {
  const size_t n = state.range(0);
  auto x = generate_data(n);

  for (auto _ : state) {
    double sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
    for (size_t i = 0; i < n; ++i) {
      sum += x[i] * x[i];
    }
    double res = std::sqrt(sum);
    benchmark::DoNotOptimize(res);
  }
  state.SetComplexityN(n);
}

static void nrm2_tbb(benchmark::State &state) {
  const size_t n = state.range(0);
  auto x = generate_data(n);

  for (auto _ : state) {
    double sum = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, n), 0.0,
        [&x](const tbb::blocked_range<size_t> &r, double local_sum) {
          for (size_t i = r.begin(); i < r.end(); ++i) {
            local_sum += x[i] * x[i];
          }
          return local_sum;
        },
        std::plus<double>());
    double res = std::sqrt(sum);
    benchmark::DoNotOptimize(res);
  }
  state.SetComplexityN(n);
}

static void nrm2_std(benchmark::State &state) {
  const size_t n = state.range(0);
  auto x = generate_data(n);

  for (auto _ : state) {

    double sum =
        std::transform_reduce(x.begin(), x.end(), 0.0, std::plus<double>(),
                              [](double v) { return v * v; });
    double res = std::sqrt(sum);
    benchmark::DoNotOptimize(res);
  }
  state.SetComplexityN(n);
}

BENCHMARK(nrm2_eigen) BENCH_ARGS;
BENCHMARK(nrm2_highway) BENCH_ARGS;
BENCHMARK(nrm2_xsimd) BENCH_ARGS;
BENCHMARK(nrm2_omp) BENCH_ARGS;
BENCHMARK(nrm2_tbb) BENCH_ARGS;
BENCHMARK(nrm2_std) BENCH_ARGS;

BENCHMARK_MAIN();