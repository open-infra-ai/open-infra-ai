# `tiny-llm` 推理正确性修复设计

## 1. 目标

让 `tiny-llm` 在固定单请求场景中，能够对一个真实 Qwen/Llama 风格 decoder-only 模型完成数值可解释的 prefill 和 decode，并建立外部 oracle。

本设计优先正确性，不同时做性能优化、continuous batching、paged KV、多 GPU 或 `cuflash-attn` 集成。

## 2. 设计决策

### 2.1 Runtime 统一采用 token-major 物理布局

这是对现有 GEMM 输出和 KV append 改动最小的选择：

| 张量 | 逻辑/物理连续布局 |
|---|---|
| hidden | `[S,C]` |
| Q | `[S,Hq,D]` |
| K/V projection | `[S,Hkv,D]` |
| K/V cache（每 sequence/layer） | `[Tcapacity,Hkv,D]` |
| attention output | `[S,Hq,D]`，展平为 `[S,C]` |

索引必须写成显式公式：

```text
q(s,h,d)     = ((s * Hq  + h)  * D + d)
kv(s,kh,d)   = ((s * Hkv + kh) * D + d)
cache(t,kh,d)= ((t * Hkv + kh) * D + d)
```

不得继续在 kernel 注释中写 `[B,H,S,D]` 而接收 token-major pointer。未来若支持 batch，先扩为 `[B,S,H,D]`，不要把 batch 与 head 展平后改变顺序。

### 2.2 GQA 映射

前置校验：

```text
C == Hq * D
Hq > 0
Hkv > 0
Hq % Hkv == 0
```

连续分组映射：

```text
group_size = Hq / Hkv
kv_head = query_head / group_size
```

所有 attention API 同时接收 `num_q_heads` 和 `num_kv_heads`；禁止继续用含义不明的单一 `num_heads` 读取 K/V。

### 2.3 RoPE 使用 half-split cache

为减少 cache 内存，内部显式采用 half cache：

- `rope_cos`: `[max_seq_len,D/2]`, FP32。
- `rope_sin`: `[max_seq_len,D/2]`, FP32。
- `inv_freq[i] = theta^(-2i/D)`。

对每个 head：

```text
x1 = x[d]
x2 = x[d + D/2]
out[d]       = x1*cos[pos,d] - x2*sin[pos,d]
out[d+D/2]   = x1*sin[pos,d] + x2*cos[pos,d]
```

Q 对 Hq 个 head 应用，K 对 Hkv 个 head应用，V 不处理。输入和输出可为 FP16，但乘加用 FP32。

### 2.4 KV Cache 长度事务

- `appendKV` 写入 `current_len` 起始的若干 position，不自行增加长度。
- 每层写自己的 cache slice。
- 当前 layer attention 可见 `current_len + num_tokens`。
- 所有 layer 成功后，InferenceEngine 只调用一次 `advanceSeqLen`。
- 任一层失败时，当前逻辑长度不提交；下一次请求不能继续使用部分状态。

第一阶段仍保持单 sequence slot，不设计复杂 rollback；发生失败后直接释放该 sequence。

### 2.5 模型权重按 architecture contract 加载

先对固定目标 GGUF 输出完整 tensor manifest，再决定每种 tensor 是否必需。至少处理：

- Q/K/V projection bias（若目标架构存在）。
- `output.weight` 与 tied token embedding 二选一。
- norm、embedding、Q/K/V/O、gate/up/down 的期望 shape。
- GGUF 磁盘 tensor 维度到 runtime `[K,N]` 的转置规则。

不允许对未知 architecture 静默套用 Qwen/Llama 默认值。

## 3. 目标接口

以下是语义草图，不要求逐字采用名称：

```cpp
void attention_prefill(
    const half* q,          // [S,Hq,D]
    const half* k,          // [S,Hkv,D]
    const half* v,          // [S,Hkv,D]
    half* output,           // [S,Hq,D]
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim,
    float scale,
    cudaStream_t stream);

void attention_decode(
    const half* q,          // [Hq,D]
    const half* k_cache,    // [T,Hkv,D]
    const half* v_cache,    // [T,Hkv,D]
    half* output,           // [Hq,D]
    int num_q_heads,
    int num_kv_heads,
    int visible_len,
    int head_dim,
    float scale,
    cudaStream_t stream);
```

`KVCacheConfig.num_heads` 应改成语义准确的 `num_kv_heads`。这是内部/早期项目的合理 breaking change，优于长期保留歧义字段。

RoPE API 应明确起始绝对位置：

```cpp
apply_rope_inplace(q, k, cos, sin,
                   num_tokens, start_position,
                   num_q_heads, num_kv_heads, head_dim, stream);
```

## 4. 实施任务

### TLLM-001：统一 layout

允许修改：

- `kernels/attention.cu/.cuh`
- `src/transformer.cpp`
- `src/kv_cache.cpp`
- 对应 headers 和 tests

步骤：

1. 先新增 token-major prefill/decode fixture，使用 `S=3,H=2,D=4` 等非对称小 shape。
2. 修改 attention 索引为 `[S,H,D]`/`[T,H,D]`。
3. 把现有独立 attention tests 的输入明确转换到新 layout。
4. KV append 保持按 position 连续 copy，但注释和 config 字段改为 Hkv。
5. 添加 cache offset 测试，直接验证两个 position、两个 head 的地址。

验收：

- `S>1,H>1` prefill 与外部 CPU/PyTorch reference 对齐。
- decode 与同一 prefix 的 reference 对齐。
- 旧 head-major 输入不能在测试中被误当成新 layout。

### TLLM-002：实现 GQA

步骤：

1. 增加模型配置一致性校验。
2. attention kernel 同时接收 Hq/Hkv。
3. 使用明确 group mapping。
4. 增加 `Hq=4,Hkv=2` 和目标 `14→2` fixture。
5. compute-sanitizer 运行非 MHA case。

验收：

- MHA `Hq==Hkv` 不回归。
- GQA 与把每个 KV head 逻辑 repeat 到相应 query group 的 PyTorch reference 等价。
- 非整除配置在任何 CUDA allocation/launch 前失败。

### TLLM-003：实现 RoPE

建议新增独立 `rope.cu/.cuh`，避免把公式散落进 attention。

步骤：

1. 使用外部脚本生成 half-split golden fixture。
2. Engine 初始化一次 FP32 cos/sin half cache，并负责生命周期。
3. Transformer 在 Q/K projection 和 bias 后、KV append 前应用 RoPE。
4. Prefill 使用 `start_position=current_len`；decode 使用显式 `position`。
5. D 必须为偶数，position 必须小于 cache capacity。

验收：

- position 0、1、较大 position 的 Q/K 与外部 reference 对齐。
- Hq 与 Hkv 不同时均正确。
- incremental 和 full-prefix 使用相同绝对位置。

### TLLM-004：模型 tensor contract

步骤：

1. 添加只读 manifest 命令/测试辅助，输出 tensor name/type/shape。
2. 为目标 architecture 建立明确映射表。
3. 给权重结构添加必要的 optional bias，并完善释放/错误清理。
4. bias 在投影后、RoPE 前添加。
5. 若 `output.weight` 缺失且 architecture 允许 tying，将 embedding 转成 runtime lm-head 表示；否则失败。
6. 删除关键 metadata 的“reasonable defaults”。

验收：

- 固定目标 GGUF 的 tensor 清单全部匹配。
- 缺一个必需 tensor 时在 GPU allocation 前返回含 tensor 名的错误。
- 有/无 bias、有独立/tied output 的小 fixture 均覆盖。

### TLLM-005：错误传播

步骤：

- `TransformerLayer::forward/forwardPrefill/runLayer/attention` 返回 `Result<void>`。
- `InferenceEngine::prefill` 返回 `Result<void>`，`decodeStep` 返回 `Result<int>`。
- 检查 `appendKV` 和 `advanceSeqLen`。
- 发生任何错误时释放 sequence，并让 `generate` 返回错误。
- CUDA async error 至少在 phase/step 同步点收敛。

非目标：不要在这个任务里把整个仓库改成异常或 `expected` 风格。

### TLLM-006：单层与真实模型 oracle

步骤见 [跨仓真实模型验证设计](end-to-end-validation.md)。必须先完成 L2，再启用 CLI generation。

## 5. 测试设计

### 5.1 小型外部 fixture

推荐参数：

```text
S=7
C=64
Hq=4
Hkv=2
D=16
intermediate=96 或 128
dtype=FP16 inputs/weights, FP32 reference accumulation
```

保存 Q/K/V、RoPE 后 Q/K、attention output、layer output。Generator 不 import `tiny-llm` 源码。

### 5.2 必测性质

- token-major layout 与显式 transpose 后的 head-major reference 等价。
- GQA 等价于 reference 中逻辑 repeat KV heads。
- prefill 最后 token 的结果与逐 token incremental 结果接近。
- cache layer/position/head 相互隔离。
- 改变 position 会改变 RoPE 后 Q/K。
- 错误 append 不增加 visible length。

## 6. 不接受的实现方式

- 只在调用 attention 前盲目 transpose，却不定义 cache layout。
- 为了让 Qwen 不越界，临时把 K/V 分配成 Hq 份而仍不实现 GQA。
- 复制项目自己的 RoPE helper 生成 expected output。
- 只比较最终生成字符串，不比较中间 logits。
- 放宽 tolerance 到足以掩盖 layout 错误。
- 通过 `cudaDeviceSynchronize()` 到处同步来隐藏状态问题。
- 在同一 PR 中加入 batching、paged cache 或性能优化。

## 7. 完成定义

本设计完成必须同时满足：

- TLLM-001 至 TLLM-005 对应测试通过。
- 单层外部 fixture 的所有中间值通过。
- 固定真实模型至少完成 prefill + 3 decode step logits 比较。
- greedy token 序列有可重复记录。
- GPU 型号、CUDA、模型 hash、reference commit 被记录。
- 所有无法执行的项明确标成未验证，而不是通过。

