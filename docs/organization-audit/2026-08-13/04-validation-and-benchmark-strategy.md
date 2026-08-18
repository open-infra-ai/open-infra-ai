# 验证与基准体系

## 1. 核心原则

正确性验证与性能测量是两条不同的证据链：

- correctness 回答“结果是否符合外部定义”。
- benchmark 回答“在已正确的实现和明确边界下有多快”。

任何 benchmark 结果都必须链接到同一提交上的 correctness 结果。

## 2. 六级验证阶梯

### L1：单算子

覆盖：RMSNorm、RoPE、GEMM、W8A16、attention、sampling。

要求：

- 外部 PyTorch/NumPy 公式不能调用被测项目实现。
- 非对称 shape，避免转置错误被方阵掩盖。
- GQA 至少覆盖 `Hq != Hkv`。
- 固定 golden fixture，加少量 property test。
- 对 NaN/Inf/empty/非法 shape 的行为有明确契约。

### L2：Transformer 层

使用小配置，例如：

```text
C=64, Hq=4, Hkv=2, D=16, intermediate=128, S=7
```

保存单层权重和以下中间值：

- attention RMSNorm 输出。
- Q/K/V 投影。
- RoPE 后 Q/K。
- attention 输出。
- residual 后状态。
- MLP 输出和最终层输出。

逐中间值比较比只比层最终输出更容易定位错误。

### L3：模型 logits

选择一个固定、体积较小的目标模型。建议继续使用项目已有叙述中的 Qwen2.5-0.5B-Instruct GGUF，但必须记录文件 hash。

对固定 token IDs：

- 比较 prefill 最后位置 logits。
- 比较连续 3–5 个 decode step logits。
- 记录 top-N token id、logit 和最大绝对/相对误差。
- 先采用 greedy，避免 RNG 干扰。

### L4：token 序列

- 同模型、同 prompt、同 chat template、同 BOS/EOS 配置。
- temperature=0。
- 与 llama.cpp 或另一个权威 runtime 逐 token 比较。
- 发生首个分歧时输出该位置两边 top-N logits。

### L5：Serving

覆盖：

- prefill/decode 混合批次。
- 多请求不同长度。
- 请求取消、客户端断开、backend 失败。
- block 用尽和资源回收。
- streaming 拼接等于一次性 decode。
- finish reason 区分 EOS、length、cancel、error。

关键不变量：

```text
allocated_blocks + free_blocks == total_blocks
每个物理块最多属于一个 live sequence
完成/取消/失败后资源最终全部归还
sequence visible length 单调且每 step 只提交一次
```

### L6：性能

只有 L1–L4 对相同实现成立后开始。

## 3. 测试环境矩阵

| Lane | 环境 | 必须执行 |
|---|---|---|
| Format/static | 普通 CPU runner | format、lint、type check、文档链接 |
| Host correctness | CPU runner | parser、tokenizer、reference、状态机、package smoke |
| CUDA compile | CPU runner + toolkit | 目标架构编译，不宣称数值通过 |
| GPU numerical | 真实 GPU | L1/L2、compute-sanitizer、真实模型 smoke |
| GPU integration | 真实 GPU | L3/L4/L5 |
| Benchmark | 固定专用 GPU | L6、环境 manifest、原始数据 artifact |

每个 lane 使用独立 check 名称。GPU 不可用不应让“GPU numerical”显示为成功；可以是未运行或由定期 runner 单独报告。

## 4. Fixture 设计

推荐目录语义：

```text
fixtures/
  manifest.json
  rope-half-split-v1.npz
  attention-gqa-v1.npz
  transformer-layer-qwen-v1.npz
  logits-qwen2.5-0.5b-v1.json
```

`manifest.json` 至少包含：

```json
{
  "schema_version": 1,
  "generator": "external-reference-script",
  "generator_commit": "...",
  "seed": 20260813,
  "dtype": "float32",
  "layout": "B,H,S,D",
  "model_sha256": "...",
  "tolerances": {
    "atol": 0.001,
    "rtol": 0.001
  }
}
```

fixture 生成器应与被测仓库隔离，避免 import 被测 kernel 或 helper。

## 5. Benchmark 规范

### 5.1 必记环境

- GPU 名称、compute capability、显存、driver。
- CUDA toolkit、NVCC、host compiler。
- OS、CPU、PyTorch/Triton/CMake 版本。
- power limit、application clocks、MIG 状态（如适用）。
- 仓库 commit、dirty 状态和构建参数。

### 5.2 计时边界

必须明确选择：

- `kernel-only`：已分配输入，CUDA Event 记录同一 stream 上的 kernel 区间。
- `operator`：包含必要布局转换和 workspace，但不含模型加载。
- `end-to-end`：包含 tokenizer/调度/传输等完整边界。

三种数字不得混在同一 speedup 表中。

### 5.3 统计方法

- 至少一次显式 warmup，直到 lazy initialization 完成。
- 保存每次原始样本，不只保存平均值。
- 报告 median、p10、p90 或 p95、样本数。
- 对短 kernel 使用批量重复，减少 event 精度影响。
- baseline 与被测实现使用同 dtype、shape、mask、layout 和输出语义。

### 5.4 性能指标

Kernel：

- latency、有效 TFLOP/s、有效 GB/s。
- FLOP/byte 公式必须与实现真实读写一致。
- profiler 记录 occupancy、memory throughput、warp stall 原因。

Runtime/Serving：

- TTFT、TPOT、inter-token latency。
- request/token throughput。
- p50/p95/p99 latency。
- 峰值显存、KV 利用率、调度等待时间。

## 6. 防止共模错误

- kernel 和 reference 不共享 index helper。
- 至少一个 golden vector 来自第三方实现。
- 测试使用非方形、非 2 的幂、`Hq != Hkv` 的 shape。
- 对 layout conversion 做单独 round-trip 和索引测试。
- 真实模型差分保留首个分歧的中间值。

## 7. 当前仓库落地顺序

1. `triton-fused-ops`：先生成标准 RoPE golden fixture。
2. `tiny-llm`：复用数学定义但独立实现 loader，完成 L1/L2。
3. `tiny-llm`：完成 L3/L4。
4. `paged-infer`：完成 CPU incremental oracle 和 L5。
5. `cuflash-attn`：刷新 GPU numerical 和 L6。
6. Academy：把相同 fixture 用作教学反例和回归测试。

