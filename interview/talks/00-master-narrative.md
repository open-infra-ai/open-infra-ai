# 总叙事（10 分钟版的骨架 + 30 秒电梯）

证据与数字以 [`../EVIDENCE_MATRIX.md`](../EVIDENCE_MATRIX.md)、[`../NUMBERS_CARD.md`](../NUMBERS_CARD.md) 为准。朗读时不要改数字。

## 1. 30 秒电梯版

**中文（约 110 字）**

我做的是一条从 CUDA 算子到推理引擎的可验证学习链：五个仓库，四层能力。旗舰是 tiny-llm，Qwen2.5-0.5B 在 RTX 3060 上 TPOT 做到 6.09 ms/token；paged-infer 用 Rust 控制面走真实分页 KV，三并发和 llama.cpp greedy 对齐，量化分歧如实记录。不是迷你 vLLM，是能讲清瓶颈、契约和验证的作品集。

**English (~95 words)**

I built a four-layer AI infra portfolio across five repos: CUDA GEMM teaching, Triton fused ops, a from-scratch FlashAttention, a native C++ runtime, and a Rust serving control plane. The flagship is tiny-llm: Qwen2.5-0.5B on an RTX 3060 Laptop at 6.09 ms/token TPOT. paged-infer drives real paged KV over a C ABI; three concurrent requests match llama.cpp greedy on prompt 1, with an honest W8A16 vs Q4_K_M token split on prompt 2. This is a verifiable learning system, not a production vLLM clone.

## 2. 四层能力 + 五仓分工

```
L1 Kernel 基础     cuda-foundations     CUDA C++ GEMM 阶梯与测量
                   triton-fused-ops     同一批算子的 Triton 表达 + torch.library
L2 Kernel 深度     cuflash-attn         FlashAttention 前后向（WMMA / FlashDecoding）
L3 Runtime         tiny-llm             GGUF → W8A16 → token；分页 KV 策略 1
L4 Serving         paged-infer          调度 / BlockPool / OpenAI API；经 C ABI 调 tiny-llm
Meta               aicl-lab             landing + 本面试证据包
```

Landing：<https://github.com/aicl-lab/aicl-lab>。五仓 `phase-2-e` 钉在「link portfolio」提交；其后有 ROADMAP 对齐的 docs commit。

**切仓原则（会被追问）**：一个算法一个 owner。FlashAttention → cuflash-attn；量化 GEMM → tiny-llm；分页控制面 → paged-infer。跨仓只走窄 ABI（`tiny-llm/include/tiny_llm/ffi.h` ↔ `paged-infer/src/tiny_llm_ffi.rs`），不互相 include。教学仓不得被 runtime 依赖。

## 3. 同一 prompt 在五仓间怎么走

以 `"Hello!"` / 聊天模板后的真实请求为例。五仓**不是**一个进程里的五段流水线；是一条学习链，真正跑 token 的是 L3+L4。

| 步 | 发生什么 | 指向 |
|----|----------|------|
| 1 | 调度器准入、分配 KV 块、组 batch | `paged-infer/src/scheduler.rs` 状态机 Pending→Prefill→Decode |
| 2 | C ABI `tinyllm_step`：扁平 `input_tokens` / `block_tables` / `num_blocks` | `ffi.h` ABI v2，9 个 int 的 `TinyLlmConfig` |
| 3 | GGUF 已在 `tinyllm_load` 反量化并重量化为 W8A16 | `tiny-llm` GGUF parser + `W8A16MatMulTest` |
| 4 | Prefill：QKV → RoPE `apply_rope_inplace` → 分页 scatter 写 KV → attention | `transformer.cpp`；GQA 14→2 |
| 5 | Decode：M==1 转置 GEMM + CUDA Graphs 重放 | `w8a16_matmul_m1_transposed_kernel`；`TLLM_CUDA_GRAPHS` 默认开 |
| 6 | 采样 greedy token，调度器把新 token 编回连续批 | 请求 1 的 24 个 id 与 llama.cpp 全等 |
| 7 | 旁路学习：同一 GEMM 在 cuda-foundations / Triton；同一 FA 在 cuflash vs Triton 参考 | LEARNING_PATH；`triton_ops::sgemm` |

**不要说**「我的 serving 把 FlashAttention 训练核接到了 vLLM」。tiny-llm 用自己的 attention kernel；cuflash 是专项深挖，未链进这条 generate 路径。

## 4. 三个最有说服力的故事（按这个顺序讲）

### ① Decode：相对 llama.cpp 从约 6.6× 收到 1.65×

- **Before**：C1 前 TPOT 24.348 ms（graphs ON、转置前）；相对 llama.cpp 3.7 ms 约 6.6×。
- **证据**：`tiny_llm_kernel_bench` 显示瓶颈在 GEMM，不是 attention。lm_head `[1,896]@[896,151936]` 10.0002 ms；N=4864 的 W8A16 0.1631 ms。根因：M==1 时 lane 读 `weight[k*N+col]`，stride = N，完全不 coalesced。
- **改动**：权重转置为 `[N,K]`，`w8a16_matmul_m1_transposed_kernel` 沿 K 连续读（`tiny-llm@db5451b`）；再默认开 CUDA Graphs（`f897084`）。
- **After**：microbench lm_head 0.9794 ms（~10.2×）；端到端 TPOT 6.560（C1）→ **6.087 ms**（C2）；比值 **1.65**。峰值显存 2494 → **3368 MB**（转置副本的代价）。
- **诚实**：不是同量化；llama.cpp 原生 Q4_K_M。ncu 不可用，用 microbench 替代。

### ② 分页 KV：策略 2 连续 KV → ABI v2 策略 1

- **Before**：FFI 只能连续 KV；paged-infer 的 block table 传了也不生效。
- **改动**：`TinyLlmConfig` 第 9 个 int `max_num_blocks`；`tinyllm_step` 增加 `block_tables`/`num_blocks`（`be8984e` / `050c80a`）。Rust `sizeof == 9*4` 布局守卫。默认策略 1，`PAGED_INFER_TINY_LLM_STRATEGY=2` 回退。
- **After**：`FFITest.PagedKVStrategyMatchesContiguous` 真模型逐 token 一致；3 并发 e2e 请求 1 全序列 24 id 对齐 llama.cpp（`9c3700b`）。
- **代价**：scatter/gather 多一次显存往返；分页路径未接 CUDA Graphs。不要背 3030 vs 5118 MiB（无归档）。

### ③ 差分测试抓住 `append_kv_at` 只写 1 token

- CUDA Graphs 要求 `write_pos` 从 device int 读，否则图会把地址固化、每步写回同一 slot（`cuda-graphs.md`）。
- 为此加了 `append_kv_at` 小 kernel。Decode 永远 `num_tokens=1`，容易把 grid 按「一行 KV」来写。
- Prefill 一次要写 S 行。注释与实现：`elementwise.cu`「D2e 修复：prefill 多 token 时逐行写入」，`total = num_tokens * per_token`，`offset = (write_pos + t) * per_token + j`。
- **怎么发现**：策略 1 vs 策略 2 的真模型差分（`FFITest.PagedKVStrategyMatchesContiguous`，`7b456cd`）。连续 KV 路径是对的，分页 prefill 若只落 1 个 token，后续 decode 可见 KV 残缺，token 立刻分叉。
- **教训**：为 decode/graph 特化的 kernel 必须用 S>1 的差分锁住 prefill；「能跑」不等于「能写满 cache」。
