# `tiny-llm` 仓库审计

## 1. 定位判断

`tiny-llm` 应继续作为组织旗舰仓库，因为它唯一覆盖了从模型文件、tokenizer 到 Transformer runtime 和 sampling 的完整方向。GGUF parser、量化类型处理和 tokenizer 是当前最有价值的资产。

但“组件存在”还没有形成“真实模型计算正确”。当前主推理链存在 layout、GQA 和 RoPE 三个相互关联的 P0，必须先于性能和 Serving 集成修复。

## 2. 当前数据流

```text
GGUF
 -> metadata/config + tokenizer data
 -> tensor decode/dequantize
 -> W8A16 runtime weights
 -> embedding
 -> N x TransformerLayer
      RMSNorm
      Q/K/V W8A16 projection
      KV append
      prefill/decode attention
      output projection + residual
      RMSNorm + SwiGLU + residual
 -> final RMSNorm + lm_head
 -> host-side sampling
 -> token IDs
```

主要入口：

- 模型构造：[`src/inference_engine.cpp`](../../../../tiny-llm/src/inference_engine.cpp#L17)
- 生成循环：[`src/inference_engine.cpp`](../../../../tiny-llm/src/inference_engine.cpp#L121)
- Transformer attention：[`src/transformer.cpp`](../../../../tiny-llm/src/transformer.cpp#L191)
- KV Cache：[`src/kv_cache.cpp`](../../../../tiny-llm/src/kv_cache.cpp#L8)
- GGUF 配置提取：[`src/gguf_parser.cpp`](../../../../tiny-llm/src/gguf_parser.cpp#L449)
- 权重加载：[`src/model_loader.cpp`](../../../../tiny-llm/src/model_loader.cpp#L76)

## 3. 值得保留的资产

### 3.1 GGUF 路径不是空壳

实现包含：

- architecture prefix 感知的 metadata 读取。
- 从 tokenizer token 数组派生 vocab size。
- 多种 GGUF 量化格式反量化。
- runtime W8A16 重量化和权重转置。
- tensor 缺失检查和真实 Qwen GGUF 的已有验证记录。

这一部分适合继续作为模型 IO 与量化学习主线。

### 3.2 Tokenizer 已有外部差分思路

README/ROADMAP 记录了与 Hugging Face tokenizer 的逐 ID 对齐。相比只写一个字符级 tokenizer，这个资产更接近真实推理工程。

### 3.3 Runtime 组件边界基本可用

ModelLoader、TransformerLayer、KVCacheManager、InferenceEngine 和 sampling 已经分层。修复时应收紧契约，不需要推倒重写。

### 3.4 构建可闭合

本次在 CUDA 12.0、sm_80、Release、`BUILD_TESTS=OFF` 下成功编译：

- `libtiny_llm.a`
- `tiny_llm_demo`

编译器报告 attention 的 `position` 参数未使用，这与 RoPE 缺失的静态结论一致。

## 4. P0 正确性问题

### TLLM-001：QKV 与 attention 布局不一致

W8A16 GEMM 写 `output[row * N + col]`，因此投影输出是 token-major `[num_tokens, heads * D]`：[`kernels/w8a16_matmul.cu`](../../../../tiny-llm/kernels/w8a16_matmul.cu#L70)。

Transformer 直接把该 buffer 传给 attention：[`src/transformer.cpp`](../../../../tiny-llm/src/transformer.cpp#L199)。Prefill kernel 却用：

```text
((batch * num_heads + head) * seq_len + position) * head_dim
```

即 head-major `[B,H,S,D]`：[`kernels/attention.cu`](../../../../tiny-llm/kernels/attention.cu#L130)。

当 `S>1` 且 `H>1` 时，两种索引不等价。当前测试若只验证独立 attention 的 head-major 输入，无法证明 Transformer 投影到 attention 的连接正确。

### TLLM-002：GQA 不成立

模型配置明确包含 `num_kv_heads`：[`include/tiny_llm/types.h`](../../../../tiny-llm/include/tiny_llm/types.h#L12)。K/V buffer 也只按 `Hkv * D` 分配，但调用 prefill/decode attention 时传入 `Hq`，kernel 用 query head 直接索引 K/V。

对于 README 使用的 Qwen2.5 `Hq=14,Hkv=2`：

- 绝大多数 query head 会越过 K/V 的逻辑范围。
- KV Cache 物理容量是 Hkv，但 decode kernel 按 Hq 布局读取。
- 即使没有触发可见 CUDA error，结果也不可能符合 GQA。

模型配置校验还缺少：

```text
hidden_dim == num_heads * head_dim
num_heads % num_kv_heads == 0
```

### TLLM-003：RoPE 未实现到运行时

仓库只解析 `rope_theta`：[`src/gguf_parser.cpp`](../../../../tiny-llm/src/gguf_parser.cpp#L467)。Q/K 投影后直接 append/attention，没有任何 RoPE 调用：[`src/transformer.cpp`](../../../../tiny-llm/src/transformer.cpp#L199)。传入 attention 的 `position` 未使用。

这会使同一 token 在不同位置拥有相同的 Q/K 位置表示，真实自回归模型 logits 必然错误。

### TLLM-004：目标模型 tensor 契约不完整

`TransformerWeights` 只有 Q/K/V/O weights 和 norm/MLP weights，没有 attention bias：[`include/tiny_llm/types.h`](../../../../tiny-llm/include/tiny_llm/types.h#L61)。Loader 也只要求和加载 `.weight`：[`src/model_loader.cpp`](../../../../tiny-llm/src/model_loader.cpp#L98)。

是否必需不能靠记忆决定，必须对固定目标 GGUF 的 tensor 清单核对。目标设计需要支持：

- architecture-specific 的可选/必需 Q/K/V bias。
- `output.weight` 缺失时可能使用 tied token embedding。
- 每个 tensor 的期望 shape、磁盘 layout 与 runtime layout。

当前 loader 强制要求独立 output/lm head：[`src/model_loader.cpp`](../../../../tiny-llm/src/model_loader.cpp#L109)。

## 5. P1 工程问题

### 5.1 关键错误被忽略

- `appendKV` 返回 `Result`，Transformer 不检查：[`src/transformer.cpp`](../../../../tiny-llm/src/transformer.cpp#L213)。
- prefill 的 `advanceSeqLen` 失败只记录日志并继续。
- decode 的 `advanceSeqLen` 返回值完全忽略：[`src/inference_engine.cpp`](../../../../tiny-llm/src/inference_engine.cpp#L286)。
- prefill/decode 多个内部函数以 `void/int` 返回，无法把错误传回 `generate`。

同时 CUDA 宏使用异常，其他路径使用 `Result<T>`。应选择清楚的边界：内部可用异常或 Result，但 public `load/generate` 必须稳定收敛为一种失败语义。

### 5.2 `repetition_penalty` 只验证、不使用

配置中包含该字段，sampling 只看到当前 logits，未传入历史 token，无法实施 repetition penalty。应实现历史感知处理或移除/拒绝非默认值。

### 5.3 CLI 与路线图不一致

ROADMAP 以 `tiny_llm_demo model.gguf --prompt "..."` 作为阶段完成命令，但 CLI 没有 `--prompt`，并明确说 demo 不执行 generation：[`src/main.cpp`](../../../../tiny-llm/src/main.cpp#L279)。其中“tokenizer pending”又与 README 的 tokenizer 已完成矛盾。

### 5.4 GGUF 关键 metadata 不应静默默认

配置提取发现关键字段非法时，会回退到 4096/32/32000 等默认值：[`src/gguf_parser.cpp`](../../../../tiny-llm/src/gguf_parser.cpp#L493)。对真实模型，这可能把“不支持或损坏的文件”转成巨大错误分配或错误 shape。

建议：只有 optional 字段有有依据的默认；hidden/layers/heads/vocab 等关键字段缺失必须失败。

### 5.5 高价值 property tests 被禁用

Incremental decoding equivalence 等测试位于 `#if 0`：[`tests/test_transformer.cu`](../../../../tiny-llm/tests/test_transformer.cu#L196)。这正是 layout/KV 问题最需要的性质。

不要简单恢复一组依赖冲突的 RapidCheck 测试；先写固定小 fixture 和普通 GTest，再考虑 property framework。

### 5.6 CI 绿灯含义有限

CI 在无 GPU 的 GitHub-hosted runner 上编译 CUDA，然后大量测试通过 `GTEST_SKIP()` 跳过。应把 CUDA compile 与 GPU numerical 拆成不同 check，不能用一个笼统绿灯表达二者。

### 5.7 Release artifact 不可作为 SDK

release workflow 复制 demo、include 和顶层 CMakeLists，没有复制静态库、源码、依赖或 package config。消费者无法按该 CMakeLists 重新构建或链接完整 library。

## 6. 有意边界

- 单请求、`max_batch_size=1`。
- W8A16 是学习实现，不追求成熟框架的 kernel 覆盖。
- 当前没有 paged KV、continuous batching、多 GPU。
- sampling 在 host 侧，性能不会优秀。

这些不是当前 P0。先证明单请求真实模型正确，再扩展并发或性能。

## 7. 测试缺口

- 非对称 QKV layout adapter 测试。
- GQA 14→2 数值和 sanitizer。
- RoPE 外部 golden vector。
- 单层逐中间值差分。
- prefill 与逐 token decode 等价。
- 固定 GGUF logits 和 llama.cpp token 对齐。
- malformed GGUF/metadata 的 fail-fast。
- tied embedding 和 architecture-specific bias。
- package smoke。

## 8. 推荐顺序

严格按以下顺序，避免错误相互掩盖：

1. 固定逻辑布局和明确转换边界。
2. 实现 GQA 映射。
3. 实现 RoPE。
4. 核对目标模型 tensor/bias/tied weights。
5. 建立单层 oracle。
6. 建立真实模型 logits oracle。
7. 完成 CLI 和逐 token 对齐。
8. 最后处理性能、release 和 Serving 集成。

详细设计见 [`tiny-llm` 正确性修复](../designs/tiny-llm-correctness.md)。

## 9. 成熟度判断

| 维度 | 判断 |
|---|---|
| GGUF/parser/tokenizer | 有实质资产 |
| Transformer 数值 | 当前不成立 |
| GQA/RoPE | 当前不成立 |
| 测试 | 组件测试较多，关键集成 oracle 缺失 |
| 构建 | 可编译 |
| 真实模型运行 | 尚未闭环 |
| 生产可用性 | 不适用 |

