# Numbers Card

> Phase 3 T3。数字从仓库文档/本次 freeze 测试输出**原样复制**。硬件未另写时均为 **RTX 3060 Laptop 6GB，驱动 591.44，CUDA 12.0，WSL2**。
> 冻结核验：[`FREEZE_AUDIT.md`](FREEZE_AUDIT.md)。证据指针：[`EVIDENCE_MATRIX.md`](EVIDENCE_MATRIX.md)。

## 1. tiny-llm TPOT / 吞吐

| 数字 | 来源 | commit | 复现 | 口径 |
|------|------|--------|------|------|
| TPOT mean **6.087 ms/token**（表内亦写作 6.09 / README 6.1） | `tiny-llm/docs/performance/results/2026-08-18-decode-optimization.md:61`；README 基准快照 6.1 | C2 `f897084` | `./build/tiny_llm_bench ../models/qwen2.5-0.5b-instruct-q4_k_m.gguf --prompt "你好" --max-tokens 64 --warmup 3 --iters 5` | greedy；graphs 默认 ON；W8A16 |
| decode **164.283 tok/s**（README 164.3） | 同上 `:62` | `f897084` | 同上 | 1/TPOT |
| TTFT mean **10.567 ms**（README 10.6） | 同上 `:60`；README | `f897084` | 同上 | 含 prefill+首 token |
| C1 前 TPOT **24.348 ms** / 41.072 tok/s / TTFT 29.584 | 同上 C1 前列表 | C1 前基线 | 同上加历史 `--graphs` | graphs ON、转置前 |
| C1 后 TPOT **6.560 ms** / 152.442 tok/s | 同上 | `db5451b` | 同上 | 转置后、C2 前 |
| llama.cpp TPOT **3.7 ms**（`tg64` 272.2 t/s） | 同上 `:103-104`；`2026-08-18-rtx3060.md:39-40` | llama.cpp `885c5bb`；tiny 对比档 `d234157`（rtx3060 文）/ C2 `f897084` | `llama-bench -m ...q4_k_m.gguf -ngl 99 -n 64 -t 1 -r 3` | **原生 Q4_K_M**，不是 W8A16 |
| 比值 tiny/llama TPOT **1.65** | decode-optimization `:103` | `f897084` | 两命令同机 | 非同量化；C1 前约 6.6 |

README 快照 commit 写成 `f897084`，命令 `--iters 10`；decode-optimization 表用 `--iters 5`。面试时报数时跟表走并报对应命令。

## 2. microbench（`tiny_llm_kernel_bench`）

来源：`2026-08-18-decode-optimization.md:79-84`。commit：C1 `db5451b` 前后。命令：`./build/tiny_llm_kernel_bench`。

| 项 | shape | 前 (ms) | 后 (ms) |
|----|-------|--------:|--------:|
| w8a16_matmul | M=1,K=896,N=128 | 0.0622 | 0.0157 |
| w8a16_matmul | M=1,K=896,N=896 | 0.0387 | 0.0164 |
| w8a16_matmul | M=1,K=896,N=4864 | 0.1631 | 0.0486 |
| w8a16_matmul | M=1,K=4864,N=896 | 0.1568 | 0.0506 |
| fp16_matmul lm_head | M=1,K=896,N=151936 | 10.0002 | 0.9794 |

小 kernel（attention_decode / rmsnorm / rope）在 ~10–50 µs，文档标明噪声 ±50%，不当瓶颈。

## 3. 显存

| 数字 | 来源 | commit | 复现 | 口径 |
|------|------|--------|------|------|
| 峰值增量 **3368 MB**（含 M==1 转置副本） | decode-optimization `:63`；README | `f897084` | `tiny_llm_bench` 同上 | `cudaMemGetInfo` 加载前 vs generate 后 |
| C1 前峰值 **2494 MB** | 同上 | C1 前 | 同上 | 无转置副本 |
| rtx3060 文峰值 **2490 MB** | `2026-08-18-rtx3060.md:41` | `d234157` | `--iters 10` 那次 | 优化前对比档 |
| 3030 MiB vs 5118 MiB | **六仓无归档** | — | — | 不要当作实测。策略 1 vs 2 正确性见 FFI 差分测试，不是这对数字 |

## 4. cuflash

硬件/commit：`docs/performance/benchmarks.md` §1.5，commit `6860cbc`，batch=1 heads=8，CUDA Event 中位数。

复现：`cmake --preset release && cmake --build --preset release && ./build/release/cuflash_bench`

**Forward 非 causal（ms）** — benchmarks.md 本机快照表：

| seq_len | hd | FP16 | BF16 | FP32 |
|--------:|---:|-----:|-----:|-----:|
| 256 | 64 | 0.158 | 0.162 | 0.841 |
| 512 | 64 | 0.496 | 0.502 | 2.86 |
| 1024 | 64 | 1.76 | 1.79 | 12.0 |
| 2048 | 64 | 6.33 | 6.45 | 47.8 |
| 4096 | 64 | 23.5 | 23.7 | 177 |
| 4096 | 128 | 84.1 | 82.6 | 379 |

**Forward causal FP32 hd=64**：1024 → 4.55 ms；4096 → 56.7 ms（同节）。

**Causal skip E2b**（`causal-boundary-skip.md`，before `d144765` / after `e1735b3`）：

| seq_len | before | after | 变化 |
|--------:|-------:|------:|-----:|
| 256 | 0.518 | 0.524 | +1.2% |
| 512 | 1.53 | 1.52 | −0.7% |
| 1024 | 4.63 | 4.59 | −0.9% |
| 2048 | 15.9 | 15.6 | −1.9% |
| 4096 | 58.4 | 57.7 | −1.2% |

grid overflow smoke：测试名 `ForwardTest.GridYOverflowSmoke`（B*H=65536）。本次 freeze 含在 70 个执行通过的 ctest 里。

## 5. triton-fused-ops

来源：README「性能基准」，commit `ebf6c32+`，torch 2.5.1 / triton 3.1.0（README 原文；本机 venv 可能更新，数字仍以该表为准）。

| 算子 | 配置 | 延迟 |
|------|------|------|
| fused_gated_mlp silu | (1,128,4096,11264) | **3.45 ms** |
| fused_gated_mlp gelu | 同上 | 3.50 ms |
| fused_rmsnorm_rope | (1,128,4096) | **0.104 ms** |

SGEMM 差分：`tests/test_sgemm.py` 本次 freeze **24 passed**（4 shapes × fp16/fp32 等参数化 + 边界/失败路径）。rtol/atol=1e-2。

## 6. 本次 freeze 测试规模（2026-08-18 实测）

| 仓 | 结果 | 命令 |
|----|------|------|
| cuda-foundations | 0 failed / **209 collected**；**78 skipped**；131 执行 | `ctest --preset default` @ `44ac954` 源码 |
| triton-fused-ops | **116 passed, 1 skipped** | `.venv/bin/python -m pytest -q` |
| cuflash | **71 collected, 0 failed, 1 skipped**（pytorch comparison） | `ctest --preset release` |
| tiny-llm | **174 passed, 1 skipped** / 175；skip=`SecondModelTest.*` | `tiny_llm_tests` + `TLLM_GGUF_TEST_MODEL` |
| paged-infer | **218 passed**（无 `tiny-llm` feature）；e2e 用例 0 运行 | `cargo fmt/clippy/test` |

## 7. llama.cpp 对齐（token）

来源：`paged-infer/tests/tiny_llm_text_e2e.rs`。commit：`9c3700b` + 诚实分歧 `9c974d3`。

| 请求 | 断言 | 序列 |
|------|------|------|
| 1 | 全序列相等 | 24 个 id，末位 EOS **151645**；文本 `Hello! I'm just a computer program...` |
| 2 | 前缀 3 + EOS | llama.cpp `[17,10,17,16819,…,151645]`（equals）；tiny-llm `[17,10,17,374,…,151645]`（is） |

本次 T1 未重跑 `--features tiny-llm`。

## 8. 数字的边界

1. **量化**：tiny-llm 走 W8A16；llama.cpp 走 Q4_K_M。token 会在 argmax 边界分叉（请求 2）。延迟比不是「同一 kernel 谁更快」。
2. **TTFT**：tiny-llm 含首 token 采样；llama-bench `pp1` 不含。禁止把 README 早期 ~4.7× TTFT 当公平比。
3. **单一硬件**：所有表都是 RTX 3060 Laptop 6GB / WSL2。换卡数字作废。
4. **ncu/nsys 不可用**：decode 故事靠 `tiny_llm_kernel_bench`，没有 occupancy 计数器。
5. **causal skip ±2%**：负结果。不要说「我们做了因果优化所以更快」。
6. **cuda-foundations 209**：CTest 0 failed ≠ 209 执行。本机 78 skip。
7. **3030 vs 5118**：提纲里有、仓库里没有。分页 vs 连续的可讲证据是逐 token 差分，不是这对 MiB。
8. **3368 MB**：转置副本把显存抬上去换来 TPOT；不是 KV 分页节省。
9. **cuflash vs SDPA 0.42×–0.67×**：教学实现预期差距，ROADMAP 未把 causal skip 写成追上 FA2。
10. **paged-infer 218**：不含真实 GPU e2e。mini-vLLM 故事的 token 对齐测试要显式开 feature。

## 9. 如果面试官只让我报 5 个数

1. **TPOT 6.09 ms/token**（C2，graphs ON，W8A16，Qwen2.5-0.5B）——运行时主数字。
2. **lm_head 10.0 → 0.98 ms**（转置 M==1）——能讲清瓶颈不在 attention。
3. **tiny/llama TPOT 1.65×**——带「不是同量化」一句。
4. **请求 1：24 token 与 llama.cpp greedy 全等；请求 2：is/equals 量化分叉**——正确性诚实。
5. **causal skip ±2%**——证明我会把负结果写进文档。
