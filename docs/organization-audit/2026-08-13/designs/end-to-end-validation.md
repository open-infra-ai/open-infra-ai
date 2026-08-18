# 跨仓真实模型验证设计

## 1. 目标

建立组织级的外部证据链，使算子、runtime 和 Serving 对“同一个模型计算”使用一致的数学定义和测试向量，同时避免五个仓库产生源码耦合。

## 2. 权威来源分层

| 层级 | 权威来源 | 用途 |
|---|---|---|
| 数学算子 | 独立 PyTorch/NumPy 明确公式 | RoPE、RMSNorm、GQA attention、MLP |
| 单层 | PyTorch 小模型/显式层实现 | 中间张量差分 |
| GGUF runtime | 固定提交的 llama.cpp harness | 同一 GGUF 的 logits/token |
| Tokenizer | Hugging Face tokenizers + GGUF vocab/merges | token ID 与 decode |
| Serving | 组织自定义状态机 oracle | 调度、取消、资源、streaming |

Reference generator 不能 import 被测仓库的 helper 或 kernel。

## 3. 模型与数据策略

### 3.1 小型合成 fixture

用于 L1/L2，必须提交到仓库或组织 fixture 包：

- 固定 seed `20260813`。
- 小而非对称的 shape。
- Hq != Hkv。
- 位置包含 0 和非零。
- 保存 FP32 expected 和测试 dtype input。

建议配置：

```text
vocab=97
C=64
layers=2
Hq=4
Hkv=2
D=16
intermediate=96
S=7
```

### 3.2 真实模型

继续使用项目已有经验的 Qwen2.5-0.5B-Instruct，但模型身份由 SHA-256 而不是文件名确定。

优先级：

1. F16/F32 GGUF，用于减少 reference 与 W8A16 之间额外量化差异。
2. 已有 Q4_K_M GGUF，用于验证真实 loader 和再量化路径。

大模型文件不提交 Git；通过环境变量/本地 cache 提供，manifest 只保存来源说明、文件大小和 SHA-256。

## 4. Artifact schema

建议组织级 fixture 目录结构：

```text
ai-infer-fixtures-v1/
  manifest.json
  tensors/
    rope-v1.bin
    gqa-attention-v1.bin
    transformer-layer-v1.bin
  expected/
    layer-trace-v1.json
    logits-prefill-v1.json
    logits-decode-v1.json
    tokens-greedy-v1.json
  generators/
    generate_math_fixtures.py
    llama_cpp_logits_harness.cpp
```

每个 tensor 记录：

```text
name
dtype
shape
logical layout
byte order
offset/length
generation formula/version
```

不要使用无 manifest 的裸 binary。

## 5. L1：算子验证

### RoPE

- half-split concat cache。
- Q heads 和 KV heads 分别覆盖。
- position 0、1、较大值。

### GQA attention

Reference 中可显式 `repeat_interleave(K/V, group_size, head axis)` 后调用 PyTorch SDPA。被测实现不得真的复制 cache 作为最终方案。

### RMSNorm/MLP/量化

- FP32 accumulate reference。
- 非全 1 norm weights。
- gate/up 不使用相同数据。
- 量化覆盖 block 边界和尾部。

消费方：`tiny-llm`、`triton-fused-ops`；适用 shape 下也可供 Academy/CuFlash 使用。

## 6. L2：单层 trace

Reference generator 输出以下检查点：

```text
input hidden
attn norm
q/k/v projection
q/k after RoPE
attention probabilities 或 row summary
attention output
post-attention residual
ffn norm
gate/up/activated product
post-ffn residual
```

`tiny-llm` 测试通过 test-only trace hook 或分步调用获得对应值。不要把 debug trace 做成稳定 public API。

定位规则：只处理第一个不一致检查点，后续差异视为级联结果。

## 7. L3：真实 GGUF logits

### 7.1 Reference harness

固定一个 llama.cpp commit，构建小型 harness：

- 加载同一 GGUF。
- 接受固定 token IDs，避免先混入 tokenizer 差异。
- 执行 prefill。
- 导出指定 position 的完整 logits 或 top-N。
- 逐 token decode 并导出每步 logits。
- 记录 llama.cpp commit、build flags 和 backend。

不要只解析 CLI 最终文本；首个 token 分歧必须能回到 logits。

### 7.2 比较指标

对 F16/synthetic：

- max/mean absolute error。
- max relative error（避开近零项时需说明）。
- cosine similarity。
- top-1/top-10 token overlap。

对 Q4→W8A16 再量化路径，bitwise 或非常紧的 logits tolerance 不现实。仍必须记录误差分布，并以 greedy top-1/token 序列作为高层验收。阈值只能由固定 baseline 实测确定，不能为了通过测试临时放宽。

## 8. L4：token 与文本

建立两组测试：

### Raw prompt

- 明确 token IDs。
- 不使用 chat template。
- `do_sample=false`。
- 固定 max tokens 和 EOS。

### Chat prompt

- 固定模型 chat template 版本。
- 先比较 template 渲染后的 token IDs。
- 再比较生成 token。

至少包含：

- 英文短句。
- 中文短句。
- 数字/代码片段。
- 能触发多个 decode step 的普通问答。

最终同时比较 token IDs 和 decode 文本；只比较字符串会掩盖 tokenizer 差异。

## 9. L5：Serving 验证

在 `paged-infer` 真实 backend 尚未接入前，先用 deterministic backend fixture 验证：

- 两请求 prefill/decode 交错。
- 不同 max tokens。
- EOS/length。
- cancel 和客户端断开。
- backend 在指定 step 失败。
- block 紧张和复用。
- SSE chunk 拼接等于完整 decode。

接入 `tiny-llm` 后，同样 case 再运行一次，但不要把模型数值问题与调度状态问题混在同一失败中。

## 10. 跨仓消费方式

共享的是 artifact schema 和数据，不共享被测源码：

- Triton 读取 NumPy/Torch fixture。
- C++ 仓库用简单 binary loader 读取同一 tensor。
- Rust 仓库读取 JSON/token fixture 或生成对应小型数据。
- 每个仓库在测试中声明自己消费的 schema version。

fixture breaking change 提升 schema version，旧消费者不能静默解释新 layout。

## 11. 执行环境

每次 L3/L4 记录：

- 模型 SHA-256。
- 五仓相关 commit 和 dirty 状态。
- reference commit。
- GPU/driver/CUDA。
- quantization/re-quantization 路径。
- prompt/token IDs。
- tolerance 和首个分歧位置。

结果写入 timestamped artifact，不覆盖旧结果。

## 12. 阶段门槛

```text
Gate A: RoPE + GQA L1 通过
Gate B: 单层全部检查点 L2 通过
Gate C: 固定 token prefill/decode logits L3 通过
Gate D: greedy token L4 通过
Gate E: serving 状态与 streaming L5 通过
Gate F: 才允许发布性能结果
```

## 13. 失败诊断顺序

1. Token IDs/position 是否相同。
2. 模型 config、tensor shape、bias/tied weights 是否相同。
3. Q/K/V projection。
4. RoPE。
5. attention 与 KV visible length。
6. residual/norm/MLP。
7. final norm/lm head。
8. sampling 和 tokenizer decode。

不要从最终文本猜测 kernel 问题。

## 14. 完成定义

- fixture generator 与被测源码隔离。
- synthetic L1/L2 全部通过。
- 同一 GGUF 至少有 prefill + 3 decode logits 报告。
- 至少三组 greedy token 序列可复现。
- Serving deterministic scenario 的资源不变量通过。
- 所有 artifact 有 manifest、commit 和环境。
- 只有达到 Gate F 的提交可以进入正式 benchmark 文档。

