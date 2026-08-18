# `cuflash-attn` 基准证据重建设计

## 1. 目标

建立一套“从当前提交可以重新运行、从每个数字可以追溯到原始样本”的 benchmark。该任务不承诺实现超过 PyTorch SDPA，也不以修改数字表为完成标志。

## 2. 前置门槛

在同一 commit 和同一 GPU 上先通过：

- FP32/FP16/BF16 forward 数值。
- causal/non-causal。
- 支持的 head dimensions。
- backward 与外部 autograd reference。
- compute-sanitizer。

任何 benchmark case 若 correctness 不通过，结果标为 invalid，不得进入 speedup 表。

## 3. 基线定义

### 3.1 当前 row-wise naive

将当前 baseline 更名为 `RowwiseNaiveForward`，准确说明：

- 每个 query row 一个 block。
- score 只在 block shared memory 中保存一行。
- 不物化全局 N×N matrix。
- 受动态 shared memory 和单 CTA 并行结构限制。

删除“会因 N² score matrix OOM”的描述。其 HBM GB/s 若无法从真实访问可靠推导，则暂时不报告，只报告 latency 和明确的 algorithmic FLOPs。

### 3.2 PyTorch SDPA

使用独立 Python comparison driver：

- 同一 GPU、dtype、shape、causal。
- 预先分配 tensor。
- 明确选择 PyTorch SDPA backend 或记录自动选择结果。
- 分别计时，不把 ctypes 转换或 host copy 放进 kernel-only 数字。

它是外部现实基线，不要求塞入同一个 Google Benchmark binary。

### 3.3 不在本轮实现的基线

真正 materialized attention 需要 QK GEMM、mask/softmax 和 PV GEMM 的完整 pipeline。除非准备准确实现和计时，否则不要为了名字保留一个假的 materialized baseline。

## 4. 计时实现

Google Benchmark case 使用 manual time：

1. 分配/初始化、stream/event 创建在 benchmark loop 外。
2. 执行固定 warmup 并同步。
3. 每次 iteration：
   - start event 记录在目标 stream。
   - 调用被测 operator。
   - stop event 记录在同一 stream。
   - 同步 stop event。
   - `cudaEventElapsedTime`。
   - `state.SetIterationTime(seconds)`。
4. 注册 benchmark 时调用 `UseManualTime()`。
5. 检查 API error 和 CUDA event/launch error。

如果需要测 host/operator latency，另建命名明确的 benchmark，不与 kernel-only 表混用。

## 5. 矩阵

### 5.1 Required correctness matrix

```text
dtype: FP32, FP16, BF16
causal: false, true
head_dim: 32, 64, 128
seq_len: 1, 7, 31, 32, 33, 127, 128, 129, 512
batch/heads: (1,1), (1,8), (2,3)
```

允许根据明确 API 边界剔除不支持组合，但必须在 manifest 中记录为 unsupported，而不是遗漏。

### 5.2 Required performance matrix

为了控制时间，主表固定：

```text
dtype: FP16
causal: true
batch: 1
heads: 8
head_dim: 64, 128
seq_len: 128, 512, 2048, 4096
```

FP32/BF16、多 batch 和 non-causal 放补充表。大于 4096 的序列只有在资源和正确性稳定后加入，不预先声称 32K/128K。

## 6. 结果格式

每次运行产生一个目录：

```text
results/<date>-<gpu>-<commit>/
  environment.json
  correctness.json
  google-benchmark.json
  pytorch-sdpa.json
  summary.md
  profiler/
```

`environment.json` 至少记录：

- commit、dirty。
- GPU、compute capability、driver、CUDA。
- host compiler、CMake、build type、arch flags。
- clock/power/MIG（能读取时）。
- PyTorch 版本和 SDPA backend。
- warmup、iterations、repetitions。

原始 JSON 是权威数据，Markdown 表由脚本生成，禁止手工抄写。

## 7. 指标

- median latency。
- p10/p90 或 p95。
- 样本数。
- algorithmic TFLOP/s，公式公开。
- speedup = baseline median / candidate median。

HBM GB/s 只有在 bytes 模型与真实实现一致时报告；否则宁可删除。Profiler 的 DRAM throughput 与自行计算的 effective bandwidth 也要分开命名。

## 8. Profiler 证据

从矩阵选 2–3 个代表 case：

- 短序列 latency-bound。
- 中序列。
- 长序列 bandwidth/compute-bound。

记录：

- kernel duration。
- achieved occupancy。
- DRAM throughput。
- tensor/core utilization（适用时）。
- warp stall top reasons。
- shared memory/register 使用。

每次优化只改变一个中心机制，并保留 before/after commit 和同环境结果。

## 9. 文档修订

所有命令必须使用实际变量：

```text
BUILD_BENCHMARKS=ON
CMAKE_CUDA_ARCHITECTURES=<arch>
```

从干净 clone 自动验证：configure、build benchmark target、列出 benchmark、运行一个最小 filter、生成 JSON。

旧 v0.4 表格若保留，应移到 `historical/`，注明不可作为当前版本结果。当前 README 只引用最新可复现 summary。

## 10. 任务拆分

| 任务 | 产物 |
|---|---|
| CUFA-101 | CUDA Event manual timing + timing unit test/smoke |
| CUFA-103 | row-wise baseline 更名、指标修正 |
| CUFA-102 | 干净 clone 文档 smoke |
| CUFA-RESULT | environment/result schema 与生成脚本 |
| CUFA-GPU | 固定硬件完整运行和 raw artifacts |
| CUFA-PROFILE | 代表 case profiler 报告 |

前四项可以在无 GPU 环境编写和编译，但只有 CUFA-GPU 实际完成后才能发布新性能数字。

## 11. 禁止事项

- 根据旧表格反推或手填新数字。
- 在不同 GPU、dtype、mask 或 shape 间直接算 speedup。
- 把 host synchronize 时间称为 CUDA Event kernel-only。
- correctness 失败后仍保留性能结果。
- 只提交 summary，不提交 raw JSON 和 environment。
- 为追求好看数字隐藏慢 case 或改变 baseline backend。

