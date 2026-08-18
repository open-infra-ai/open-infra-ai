# `cuflash-attn` 仓库审计

## 1. 定位判断

`cuflash-attn` 是五仓中工程外壳最成熟的项目，适合继续作为 CUDA kernel 深度旗舰。它的中心价值应是：

- 能解释 online softmax 和 tiled FlashAttention。
- 能展示 FP32/FP16/BF16 前向、反向和 causal 路径。
- 能用独立数值测试、compute-sanitizer 和 profiler 证明优化过程。

当前不应扩展成通用 attention framework。先把现有实现的数值证据与 benchmark 证据做实，比增加 varlen/GQA/FA3 名词更有价值。

## 2. 当前架构

```text
Public C++/C API
  -> parameter validation / dispatch
  -> typed forward kernels
       tiled Q/K/V
       online softmax
       causal masking
  -> typed backward kernels
       dQ pass
       dK/dV pass
  -> tests / PyTorch comparison
  -> Google Benchmark driver
```

项目同时具备 CMake presets、install/export、package smoke、examples、文档站点、CodeQL、host compile matrix 和自托管 GPU workflow。

## 3. 值得保留的资产

### 3.1 API 与实现边界较清楚

公开头文件与内部 kernel 分离，错误码覆盖非法维度、空指针、CUDA error、OOM 和不支持 head dimension。它比 Academy 的 header-only 教学表面更接近可集成 library。

### 3.2 前后向都有真实实现

仓库不是只包装 PyTorch 或只实现 naive attention。online softmax、causal mask、typed dispatch 和反向分解构成了有实质的专项作品。

### 3.3 工程验证结构较好

- hosted CI 明确只做编译和 host smoke。
- GPU workflow 明确要求真实 GPU，包含 numerical suite、RapidCheck 和 compute-sanitizer：[`gpu.yml`](../../../../cuflash-attn/.github/workflows/gpu.yml#L1)。
- CI 覆盖多架构编译、Debug sanitizer 配置、package smoke、docs 和 CodeQL。
- release 方向比其他 C++ 仓库更完整。

### 3.4 当前快照可编译

本次在 CUDA 12.0、Release、sm_80、关闭 tests/examples/benchmarks/shared library 后成功生成 `libcuflash_attn.a`。

构建产生大量 NVCC 生成代码与 GCC pedantic 的 line-directive warning，但没有编译失败。它属于工具链噪声，后续可调整 warning 作用范围，不是 correctness 问题。

## 4. 主要问题

### CUFA-101 / P1：性能计时文档与实现不一致

性能文档声明：

- Google Benchmark `UseManualTime()`。
- CUDA Event kernel-only 计时。
- 10 次预热、30 次中位数。

见 [`docs/performance/benchmarks.md`](../../../../cuflash-attn/docs/performance/benchmarks.md#L30)。实际 benchmark loop 只是调用 API 后 `cudaStreamSynchronize(stream)`，没有 CUDA Event，也没有 `state.SetIterationTime`：[`benchmarks/bench_flash_attention.cu`](../../../../cuflash-attn/benchmarks/bench_flash_attention.cu#L211)。

Google Benchmark 记录的是 host wall time，包含 wrapper/launch/synchronize，但文档把它描述为 kernel-only。这会使结果口径不可信。

### CUFA-102 / P1：复现命令无效

文档使用：

```text
CUFASH_ATTN_BENCHMARKS
CUFASH_ATTN_ARCHS
```

实际 CMake 使用 `BUILD_BENCHMARKS` 和标准 `CMAKE_CUDA_ARCHITECTURES`：[`CMakeLists.txt`](../../../../cuflash-attn/CMakeLists.txt#L17)、[`CMakeLists.txt`](../../../../cuflash-attn/CMakeLists.txt#L211)。文档还保留 `your-org` clone 地址和与 preset 输出不一致的可执行文件路径。

复现文档必须由干净 clone smoke 自动执行，避免再次漂移。

### CUFA-103 / P1：naive baseline 定义失真

注释称它物化完整 N×N score matrix，但 kernel 实际只为当前 row 使用 `seq_len` 个 shared scores：[`benchmarks/bench_flash_attention.cu`](../../../../cuflash-attn/benchmarks/bench_flash_attention.cu#L56)。

后续指标又把 N² score matrix 的全局写回和读取计入 HBM bytes：[`benchmarks/bench_flash_attention.cu`](../../../../cuflash-attn/benchmarks/bench_flash_attention.cu#L157)。这与真实内存访问不符。

影响：

- baseline 名称和 OOM 叙述错误。
- HBM GB/s 计算没有物理意义。
- 与标准 materialized attention 的比较不成立。
- shared memory 为 `seq_len * sizeof(float)`，大序列还会受单 block shared memory 限制。

应选择一种明确方案：实现真正的多 kernel materialized baseline，或把当前实现改称 row-wise naive fused attention 并重写流量公式。

### CUFA-104 / P1：历史性能表缺少可追溯 artifact

ROADMAP 承认当前文档是旧版本快照，但文档包含超出当前 tracked benchmark 固定矩阵的硬件、序列和结论。没有发现与表格逐项对应的原始 JSON、环境 manifest 和 profiler artifact。

旧数字可以保留为明确的历史记录，但不能作为当前 v0.5.0 的验证结果或优化基线。

### CUFA-105 / P1：GPU 证据依赖外部 runner 状态

workflow 注释说明无自托管 runner 时会排队。仅存在 YAML 不能证明最近执行成功。每次 release 应附最近一次真实 GPU 验证日期、设备、commit 和 artifact。

## 5. 能力边界

当前实现主要是同形状 dense self-attention：

- Q/K/V 采用相同 `num_heads`。
- 不支持 GQA/MQA。
- 不支持 varlen/unpadded。
- 不支持 cross-attention 的不同 Q/K 长度。
- 没有 decode `query_len=1` 专用 kernel 或 Split-KV。
- 不包含 dropout 和更完整训练框架语义。

这些边界不是当前 bug，但 README/API reference 必须明确。因为 `tiny-llm` 的目标 Qwen 使用 GQA，现版本不能直接作为它的 attention backend。

## 6. 测试缺口

- 当前实现所有支持 dtype/head_dim/causal 的系统矩阵。
- 更长 sequence 下的数值稳定性。
- backward 对 PyTorch autograd 的完整 shape/dtype 矩阵。
- 动态 shared memory 边界和资源不足错误。
- GPU workflow 最近成功的可见证据。
- benchmark driver 自身的计时口径测试。
- 文档命令从干净 clone 的可执行 smoke。

## 7. 推荐顺序

1. 在真实 GPU 上跑当前 numerical、property、sanitizer，先确认 correctness。
2. 删除或降级无法追溯的当前性能声明。
3. 决定 naive baseline 的真实定义。
4. 统一 CUDA Event/manual timing 与文档。
5. 固定最小 benchmark matrix 和 JSON schema。
6. 做一轮 profiler 驱动、带 before/after 的优化。
7. 只有需要服务真实 runtime 时，再设计 GQA/decode；不要作为低成本“顺手功能”。

详细方案见 [基准证据重建设计](../designs/cuflash-attn-benchmark.md)。

## 8. 成熟度判断

| 维度 | 判断 |
|---|---|
| Kernel 实质 | 较强 |
| API/packaging | 五仓最佳 |
| Hosted CI | 编译与 host smoke 较强 |
| GPU 数值证据 | 当前快照未现场验证 |
| 性能证据 | 需要重建 |
| Runtime 可复用性 | 受等头/dense 边界限制 |
| 生产可用性 | 不适用 |

