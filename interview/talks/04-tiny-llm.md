# tiny-llm · 10 分钟讲述稿

优化故事必须用 microbench 与 TPOT **24.348 → 6.560 → 6.087 ms**（[`../NUMBERS_CARD.md`](../NUMBERS_CARD.md) §1–2）。

## 0. 一句话定位

CUDA 原生最小运行时：GGUF 到 token。

## 1. 2 分钟：做什么、边界、为什么这样切

IN：GGUF 加载与反量化、W8A16 推理、KV、tokenizer、采样、分页 KV 策略 1、C ABI、bench、CUDA Graphs。

OUT：调度/CB → paged-infer；FA 深挖 → cuflash；Triton → triton-fused-ops。

不是低配 llama.cpp：目标是 **能精确回答瓶颈在反量化、访存还是 launch**。模型：Qwen2.5-0.5B Instruct，GGUF Q4_K_M 重量化 W8A16。GQA 14→2、RoPE 在 `transformer.cpp` 真实调用。tokenizer 30 例 417 token 与 HF 逐 id 对齐。

本次 freeze：`tiny_llm_tests` **174 passed / 1 skipped**（第二 GGUF）。

## 2. 3 分钟：最难的实现细节

**M==1 decode GEMM 的访存形状。**

Decode 每层都是 `[1,K] @ [K,N]`。旧 `w8a16_matmul_m1_kernel`：一个 warp 一列，lane `k, k+32, ...` 归约 K，但权重是 `weight[k*N + col]`。同一 warp 32 个 lane 地址步长 **N×1B（int8）**，完全不 coalesced。lm_head N=151936 时这就是主瓶颈。

C1：转置为 `[N,K]`，`weight_t[col*K + k]`，lane 沿 K 连续，stride=1（`db5451b`，`w8a16_matmul.cu` 注释写明对比）。

第二细节（1 分钟）：CUDA Graphs 把 `visible_len` / `write_pos` / RoPE pos 改成 device int，否则图固化参数（`cuda-graphs.md`）。`append_kv_at` 必须写 `num_tokens` 行，否则 prefill 只落 1 token（见总叙事故事 ③）。

## 3. 2 分钟：优化故事（规定数字）

| 阶段 | TPOT ms | tok/s | 备注 |
|------|--------:|------:|------|
| C1 前 | **24.348** | 41.072 | graphs ON，未转置 |
| C1 后 | **6.560** | 152.442 | 转置快路径 |
| C2 后 | **6.087** | 164.283 | graphs 默认 ON |

llama.cpp 同卡 `tg64`：**3.7 ms / 272.2 t/s**。比值 约 6.6× → **1.65×**。

Microbench 同文档：

- lm_head **10.0002 → 0.9794 ms**（~10.2×）
- W8A16 N=4864 **0.1631 → 0.0486 ms**

代价：峰值显存 **2494 → 3368 MB**（转置副本）。ncu 不可用（`ERR_NVGPUCTRPERM`），证据是 `tiny_llm_kernel_bench`。graphs on/off greedy 一致：`CudaGraphsGenerateMatchesNonGraph`。

## 4. 2 分钟：验证方法

- L1 kernel：GQA/RoPE/W8A16 vs CPU。
- L4 生成：与 llama.cpp greedy；前缀一致后因 W8A16 vs Q4_K_M 可分叉（「is/equals」）。
- 分页 vs 连续：`FFITest.PagedKVStrategyMatchesContiguous`。
- Tokenizer：`TokenizerRealModel.DifferentialAgainstHuggingFace`。
- 方法论：`benchmark-methodology.md`（TTFT 口径不同，禁止乱除）。

## 5. 1 分钟：短板与下一步

短板：无 Tensor Core decode GEMM；分页路径无 Graphs；第二真实模型未跑；无长上下文显存曲线。下一步（冻结外）：paged decode 直接读 pool、或上游小 PR。默认冻结。

## 6. 追问清单

1. 为什么不用 llama.cpp？ → 学习可控性；产品会选它。见 cross-cutting。
2. W8A16 是什么？ → INT8 权重 + FP16 scale，group 128；激活 FP16。
3. 为什么转置能快？ → coalesced K 读。
4. 为什么显存涨了？ → 3368 含转置副本。
5. Graphs 捕获了什么？ → decode device path；变化值在 device int。
6. GQA 14→2 怎么映射？ → `kv_head = q_head / group_size`。
7. RoPE 进没进前向？ → `apply_rope_inplace` 在 attention 前。
8. 和 llama.cpp 还差多少？ → TPOT 1.65×，量化不同。
9. FFI 几个 int？ → 9，含 `max_num_blocks`。
10. append_kv_at 的 bug 是什么？ → prefill 多 token 曾只写 1 行；差分抓住。
