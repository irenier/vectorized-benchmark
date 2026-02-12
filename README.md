# 向量化计算库性能测试报告

## 实验环境与配置

- 测试平台：Intel(R) Core(TM) i9-10900X CPU @ 3.70GHz (Cascade Lake)
- 硬件架构：10 Cores / 20 Threads，支持 AVX-512 (F, DQ, BW, VL, VNNI)
- 编译器：g++ 14.2.0
- 编译参数：见 [CMakeFile](CMakeLists.txt)
- 软件依赖：Highway (1.3.0), xsimd (14.0.0), Eigen (3.4.0), TBB, OpenMP

## 结果展示

###

AXPY ($Y = aX + Y$): 经典的向量加权累加，测试内存访问吞吐量与 FMA 指令流。

![axpy](./results/benchmark_axpy.svg)

DOT (点积): 测试跨通道（Across-lane）规约效率。

![dot](./results/benchmark_dot.svg)

NRM2 (L2 范数): 测试单向量平方累加的稳定性与速度。

![nrm2](./results/benchmark_nrm2.svg)

SumExp ($\sum e^{x_i}$): 测试 SIMD 数学函数逼近（Polynomial Approximation）的性能，属于重度计算任务。

![sumexp](./results/benchmark_sumexp.svg)

## 核心性能指标汇总 (Big-O Analysis)

下表展示了各库实现在大规模数据下的渐进时间复杂度系数。该系数越小，代表单核/多核吞吐量越高。

| 实现方式 | AXPY (ns/N) | DOT (ns/N) | NRM2 (ns/N) | SumExp (ns/N) |
| --- | --- | --- | --- | --- |
| Eigen | 1.22 | 1.12 | 0.62 | 1.45 |
| Highway (SIMD) | 1.24 | 1.11 | 0.62 | 1.29 |
| xsimd (SIMD) | 1.22 | 1.15 | 0.62 | 1.16 |
| OpenMP (Multi-core) | 0.35 | 0.22 | 0.10 | 0.57 |
| TBB (Multi-core) | 0.35 | 0.22 | 0.10 | 0.61 |
| std (Scalar/PSTL) | 1.27 | 1.35 | 0.72 | 4.80 |

## 算子深度对比分析

### 访存密集型：AXPY, DOT, NRM2

在单线程模式下，Eigen, Highway 和 xsimd 的表现几乎一致（约 1.1-1.2 ns/N）。

- 内存瓶颈：对于 AXPY 这种需要三路流（读 X, 读 Y, 写 Y）的操作，性能主要受限于单核内存带宽。
- 规约优化：NRM2 (0.62 ns/N) 的速度几乎是 DOT (1.11 ns/N) 的两倍。这是因为 NRM2 只需读取一个向量，减少了 50% 的内存总线压力。

### 计算密集型：SumExp

在指数运算测试中，SIMD 库展现了压倒性优势：

- SIMD 加速比：xsimd (1.16 ns/N) 相比 `std` 标量循环 (4.80 ns/N) 提升了约 4.1 倍。
- 库对比：在处理 `exp` 函数时，xsimd 略微优于 Highway 和 Eigen。这可能源于 xsimd 在 Cascade Lake 架构上对 AVX-512 FMA 指令流的更优排布。

### 并行加速：OpenMP 与 TBB

- 扩展性：在 $N=10^9$ 的大规模数据下，OpenMP 相比单线程 SIMD 实现了约 3.5 - 6 倍 的实际加速（例如 NRM2 从 0.62 降至 0.10）。
- 并行开销：在数据量小于 $10^4$ 时，OpenMP/TBB 的 real_time 远高于单线程（约 3000-7000ns），说明多线程调度开销在此规模下不可忽略。

## 结论

1. 策略选择：对于计算受限任务（如 SumExp），SIMD 向量化是性能提升的首要手段；对于访存受限任务（如 AXPY），多核并行是突破单核带宽瓶颈的唯一途径。
2. 库推荐：
    - xsimd：在处理复杂数学函数（Exp）时展现了极高的能效比。
    - Eigen：在标准 BLAS 操作中保持了极其稳健的底层逻辑，且易于集成。
    - OpenMP：在处理超大规模 $N>10^7$ 数组时，作为顶层调度框架的表现最为出色。
