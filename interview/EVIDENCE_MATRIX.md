# Evidence Matrix — 30 条核心声明

> Phase 3 T2。每条必须能指向仓库/文件/commit/测试名/复现命令之一。
> 性能数字从文档原样复制，不改写。冻结核验见 [`FREEZE_AUDIT.md`](FREEZE_AUDIT.md)。

### E1. SGEMM 优化阶梯从 naive 到 Tensor Core，每步有实测 TFLOPS
- 证据类型：benchmark 文档
- 位置：`cuda-foundations/docs/en/benchmarks/index.md:31-38`（1024³ FP32）
- 关键 commit：测量页 `cuda-foundations` 文档站 RTX 3060 快照（见同目录 `rtx3060-laptop-2026-08-17`）
- 复现命令：
```bash
cd cuda-foundations && cmake --preset default && cmake --build --preset default
# 01 模块 SGEMM bench；数字以 docs/en/benchmarks/ 归档为准
```
- 口径/限制：Naive 0.58 / Tiled 0.92 / Bank-conflict-free 0.66 / Double-buffer 0.68 / WMMA 1.09 / cuBLAS 5.58 TFLOPS。教学阶梯，不是打赢 cuBLAS。Bank-conflict-free 与 double-buffer 相对 tiled **更慢**，文档未隐瞒。

### E2. CUDA 与 Triton 的 SGEMM「同题异构」对比存在
- 证据类型：代码路径 + 文档
- 位置：CUDA 阶梯 `cuda-foundations/01-sgemm-tutorial/`；Triton `triton-fused-ops/triton_ops/kernels/sgemm.py`；导航 `cuda-foundations/LEARNING_PATH.md:9-11`；triton README 指向 cuda-foundations
- 关键 commit：`triton-fused-ops@e85d824` `feat(triton): SGEMM kernel with differential tests`
- 复现命令：
```bash
cd triton-fused-ops && .venv/bin/python -m pytest -q tests/test_sgemm.py
```
- 口径/限制：两边都是教学/练习实现；没有同一脚本、同一时刻的 head-to-head TFLOPS 表。对比点是「同题两套实现 + 各自差分测试」，不是生产选型跑分。

### E3. Triton 三个融合算子有独立参考实现与差分测试
- 证据类型：测试
- 位置：`triton-fused-ops/tests/test_rmsnorm_rope.py`、`tests/test_gated_mlp.py`、`tests/test_flash_attention.py`；参考 `triton_ops/reference/`
- 关键 commit：`triton-fused-ops@0c5b1ed` 收敛到可验证 Transformer kernels
- 复现命令：
```bash
cd triton-fused-ops && .venv/bin/python -m pytest -q tests/test_rmsnorm_rope.py tests/test_gated_mlp.py tests/test_flash_attention.py
```
- 口径/限制：FlashAttention 前向是 cuflash-attn 的参考实现，不是本仓优化旗舰。无 GPU 时 kernel 差分 skip。

### E4. TRIT-001 RoPE half-split 约定 bug 已修复并有测试
- 证据类型：代码路径 + 测试
- 位置：`triton-fused-ops/triton_ops/reference/rmsnorm_rope.py:320-324`（concat 而非 `repeat_interleave`）；`examples/rmsnorm_rope_example.py:33-36`
- 关键 commit：`triton-fused-ops@b1bcdcb` `fix(triton): TRIT-001 half-split RoPE convention and Triton 3.x compatibility`
- 复现命令：
```bash
cd triton-fused-ops && .venv/bin/python -m pytest -q tests/test_compute_rope.py tests/test_rmsnorm_rope.py
```
- 口径/限制：契约是 Llama/Qwen `rotate_half`（half-split），不是 interleaved pair。审计 TRIT-001 关闭依赖这条，而不是「随机 cos/sin 自比通过」。

### E5. torch.library 注册的三个自定义 op 可直接调用
- 证据类型：代码路径 + 测试
- 位置：`triton-fused-ops/triton_ops/ops.py:6-8,120-127`；schema：`triton_ops::sgemm`、`triton_ops::fused_rmsnorm_rope`、`triton_ops::fused_gated_mlp`
- 关键 commit：`triton-fused-ops@1bbf5c8` `feat(torch): register custom ops via torch.library`
- 复现命令：
```bash
cd triton-fused-ops
.venv/bin/python -c "import torch, triton_ops; print(torch.ops.triton_ops.sgemm)"
.venv/bin/python -m pytest -q tests/test_torch_library.py
```
- 口径/限制：本次 freeze `test_torch_compile_smoke` skip（compile 失败不伪造通过）。op 只接受 CUDA 张量。

### E6. FlashAttention 前向+反向多精度通过差分测试
- 证据类型：测试
- 位置：`cuflash-attn/tests/unit/`（forward/backward/dtype）；本次 freeze `ctest --preset release` 71 collected / 0 failed / 1 skip（pytorch comparison）
- 关键 commit：`cuflash-attn` v0.5.0 前后向 WMMA 路径（CHANGELOG `[0.5.0]`）
- 复现命令：
```bash
cd cuflash-attn && cmake --preset release && cmake --build --preset release && ctest --preset release --output-on-failure
```
- 口径/限制：skip 的 pytorch 对比本次未跑。head_dim 支持 32/64/128。不是 FA2/FA3 吞吐竞品。

### E7. grid.y > 65535 的 launch bug 已修复并有回归测试
- 证据类型：测试 + 代码路径
- 位置：`cuflash-attn/tests/unit/test_forward.cu:397` `ForwardTest.GridYOverflowSmoke`（B=512,H=128 → 65536）；kernel `src/forward/flash_attention_forward_typed.cu:22`
- 关键 commit：`cuflash-attn@d144765` `fix(forward): flatten grid.y batch*heads for >65535 launches`
- 复现命令：
```bash
cd cuflash-attn && ctest --preset release -R GridYOverflowSmoke --output-on-failure
```
- 口径/限制：该回归走 FP32 scalar、seq_len=1；不覆盖 WMMA tile 约束。

### E8. causal 边界块跳过优化实测 ±2%，诚实记录为负结果
- 证据类型：benchmark 文档
- 位置：`cuflash-attn/docs/performance/causal-boundary-skip.md:32-41,65-72`
- 关键 commit：`cuflash-attn@e1735b3` `perf(forward): skip fully-future KV blocks in causal path`
- 复现命令：
```bash
./build/release/cuflash_attn_bench --benchmark_filter='Forward_Causal'
```
- 口径/限制：FP32 causal 256–4096 变化约 +1.2% 到 −1.9%，低于 10% 阈值。文档写明「增益低于噪声」。保留改动是因为语义自文档化与无效访存，**不是因为加速了**。

### E9. FlashDecoding / Split-KV 前向已实现并有 benchmark
- 证据类型：代码路径 + 测试
- 位置：`cuflash-attn/src/forward/flash_decoding.cu:1`；`tests/unit/test_flash_decoding.cu`（`MatchesCpuReferenceF32` / `ChunkCountInvariant`）；`benchmarks/bench_flash_attention.cu` Split-KV 段
- 关键 commit：`cuflash-attn@9f65df7` `feat: E3 FlashDecoding (Split-KV) decode 前向`
- 复现命令：
```bash
cd cuflash-attn && ctest --preset release -R FlashDecoding --output-on-failure
```
- 口径/限制：query_len=1 的 decode 路径；不是把整个 FA 训练前向换成 Split-KV。

### E10. tiny-llm 从 GGUF 到文本一条命令跑通真实 Qwen2.5-0.5B
- 证据类型：代码路径 + 测试 + README
- 位置：`tiny-llm/src/main.cpp:198-248`；README 状态表「真实模型端到端生成」
- 关键 commit：阶段 1 端到端路径（权重契约 `tiny-llm@d234157`）
- 复现命令：
```bash
cd tiny-llm
./build/tiny_llm_demo /home/shane/github/aicl/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --prompt "你好" --max-tokens 32 --show-tokens
```
- 口径/限制：GGUF Q4_K_M 加载后重量化为 W8A16 再推理。本次 freeze 用同一模型跑了 `tiny_llm_tests`（含 graphs on/off 生成对齐）。

### E11. tokenizer 与 HF tokenizers 差分逐 id 对齐（30 例 417 token）
- 证据类型：测试 + fixture
- 位置：`tiny-llm/tests/test_tokenizer.cpp:105` `TokenizerRealModel.DifferentialAgainstHuggingFace`；`tests/tokenizer_fixture_cases.h`（`kCaseCount`，本文件点算 **30 例、417 token**）
- 关键 commit：tokenizer 差分合入（见 tiny-llm CHANGELOG Unreleased tokenizer 段）
- 复现命令：
```bash
TLLM_GGUF_TEST_MODEL=/home/shane/github/aicl/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  ./build/tiny_llm_tests --gtest_filter='TokenizerRealModel.*'
```
- 口径/限制：门控于真实 GGUF。本次 freeze 该套件通过。

### E12. GGUF 反量化（Q4_0/Q5_0/Q8_0/Q4_K/Q6_K）与 Python gguf 参考一致
- 证据类型：测试
- 位置：`tiny-llm/tests/test_quantization.cpp`（`DequantizeQ5_0` / `DequantizeQ4_K` / `DequantizeQ6_K` + `GGUFRealModelTest`）
- 关键 commit：量化路径合入（CHANGELOG Unreleased）
- 复现命令：
```bash
TLLM_GGUF_TEST_MODEL=/home/shane/github/aicl/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  ./build/tiny_llm_tests --gtest_filter='Dequantize*:*GGUFRealModelTest*'
```
- 口径/限制：合成块期望值来自 Python `gguf.quants`；真实模型首块门控于 Qwen2.5-0.5B Q4_K_M。

### E13. W8A16 量化推理路径有 CPU/参考差分测试
- 证据类型：测试
- 位置：`tiny-llm/tests/test_w8a16_matmul.cu`（`W8A16MatMulTest.*`，含 `TransposedFastPathMatchesCpuReference`）；`tests/test_quantization.cpp` `WeightW8A16RoundTripPreservesValues`
- 关键 commit：`tiny-llm@db5451b` 转置 M==1 快路径
- 复现命令：
```bash
./build/tiny_llm_tests --gtest_filter='W8A16MatMulTest.*'
```
- 口径/限制：差分是 kernel vs CPU 参考，不是 vs PyTorch 完整模型。本次 freeze 该套件通过。

### E14. GQA（14→2）与 RoPE 进入真实计算路径并有精确规格测试
- 证据类型：测试 + 代码路径
- 位置：kernel `tiny-llm/tests/test_kernels.cu:889` `GQAMappingDecodeMatchesCpuReference`；RoPE `test_kernels.cu:1279` `RoPETest.ApplyInplaceMatchesReference`；前向调用 `src/transformer.cpp:282,360` `apply_rope_inplace`；真实模型 GQA 14→2 见 README
- 关键 commit：`tiny-llm@fdbabcc`（GQA）、`tiny-llm@1038639`（RoPE）
- 复现命令：
```bash
./build/tiny_llm_tests --gtest_filter='*GQA*:*RoPE*'
```
- 口径/限制：Qwen2.5-0.5B 端到端已验 14→2。第二真实模型（`TLLM_GGUF_TEST_MODEL_2`）本次 skip。

### E15. 转置权重 M==1 GEMM 优化：microbench 与端到端 before/after
- 证据类型：benchmark 文档
- 位置：`tiny-llm/docs/performance/results/2026-08-18-decode-optimization.md:56-84`
- 关键 commit：C0 `ca70de2` → C1 `db5451b` → C2 `f897084`
- 复现命令：
```bash
./build/tiny_llm_kernel_bench
./build/tiny_llm_bench ../models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --prompt "你好" --max-tokens 64 --warmup 3 --iters 5
```
- 口径/限制：lm_head FP16 `10.0002 → 0.9794 ms`（~10.2×）；端到端 TPOT `24.348 → 6.087 ms`。N=4864：`0.1631 → 0.0486 ms`。ncu 不可用，microbench 是替代证据。

### E16. CUDA Graphs 默认开启，graphs on/off greedy 输出逐 token 一致
- 证据类型：测试 + 文档
- 位置：`tiny-llm/tests/` `InferenceEngineTest.CudaGraphsGenerateMatchesNonGraph`（本次 freeze 通过）；文档 `2026-08-18-decode-optimization.md:110-127`
- 关键 commit：`tiny-llm@f897084` `perf(runtime): enable CUDA Graphs decode by default with opt-out`
- 复现命令：
```bash
./build/tiny_llm_tests --gtest_filter='InferenceEngineTest.CudaGraphsGenerateMatchesNonGraph'
# 或
./build/tiny_llm_demo model.gguf --prompt "你好" --max-tokens 32 --show-tokens
TLLM_CUDA_GRAPHS=0 ./build/tiny_llm_demo model.gguf --prompt "你好" --max-tokens 32 --show-tokens
```
- 口径/限制：默认开启，`TLLM_CUDA_GRAPHS=0` opt-out。捕获失败回退常规路径。

### E17. 与 llama.cpp 的对比方法论文档 + 公平性声明
- 证据类型：文档
- 位置：`tiny-llm/docs/performance/benchmark-methodology.md:31-54`；实测归档 `docs/performance/results/2026-08-18-rtx3060.md` 与 `2026-08-18-decode-optimization.md:99-108`
- 关键 commit：`tiny-llm@753d913` `docs: B2 llama.cpp 对比实测`
- 复现命令：见 methodology 第 4 节（`tiny_llm_bench` + `llama-bench -ngl 99`）
- 口径/限制：llama.cpp 原生 Q4_K_M；tiny-llm 重量化 W8A16。TTFT 口径不同，文档禁止直接相除当公平比。C2 后 TPOT 比值 1.65（6.09 / 3.7 ms）。

### E18. 「2+2 is/equals」量化分歧被诚实记录为前缀一致 + EOS
- 证据类型：测试
- 位置：`paged-infer/tests/tiny_llm_text_e2e.rs:159-166,272-280`
- 关键 commit：`paged-infer@9c974d3` `test(e2e): honest llama.cpp divergence fixture for 2+2 prompt`
- 复现命令：
```bash
cd paged-infer
TINY_LLM_DIR=/home/shane/github/aicl/tiny-llm \
TINY_LLM_MODEL=/home/shane/github/aicl/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
PINF_TOKENIZER_JSON=/home/shane/github/aicl/models/tokenizer.json \
  cargo test --features tiny-llm --test tiny_llm_text_e2e -- --nocapture
```
- 口径/限制：llama.cpp `[17,10,17,16819,…,151645]`（equals）；tiny-llm `[17,10,17,374,…,151645]`（is）。第 4 个 token 是 W8A16 vs Q4_K_M argmax 边界翻转。本次 T1 **未**重跑该 feature 测试。

### E19. FFI C ABI v2：9-int 布局 + num_blocks，Rust 布局守卫测试
- 证据类型：代码路径 + 测试
- 位置：`tiny-llm/include/tiny_llm/ffi.h:4,28-38`；`paged-infer/src/tiny_llm_ffi.rs:143-144` `tiny_llm_config_layout_is_stable`
- 关键 commit：`tiny-llm@be8984e`；`paged-infer@050c80a`
- 复现命令：
```bash
cd paged-infer && cargo test tiny_llm_config_layout_is_stable
```
- 口径/限制：`sizeof(TinyLlmConfig) == 9*4`。策略 1 用 `max_num_blocks > 0` 与 `block_tables`/`num_blocks`。

### E20. paged KV 策略 1 与连续 KV 策略 2 真模型逐 token 差分一致
- 证据类型：测试
- 位置：`tiny-llm/tests/test_ffi.cpp:199` `FFITest.PagedKVStrategyMatchesContiguous`
- 关键 commit：`tiny-llm@7b456cd` `test(ffi): paged strategy differential vs contiguous strategy`
- 复现命令：
```bash
TLLM_GGUF_TEST_MODEL=/home/shane/github/aicl/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  ./build/tiny_llm_tests --gtest_filter='FFITest.PagedKVStrategyMatchesContiguous'
```
- 口径/限制：本次 freeze FFI 套件包含在 174 passed 内。策略 1 有 gather/scatter 往返，正确性优先于延迟。

### E21. paged-infer 3 并发 e2e 与 llama.cpp 参考对齐（请求 1 含 EOS 共 24 token）
- 证据类型：测试
- 位置：`paged-infer/tests/tiny_llm_text_e2e.rs:166,257-266`；参考序列 24 个 id，末位 `151645`
- 关键 commit：`paged-infer@9c3700b` `test(e2e): 3-way paged concurrency with llama.cpp token parity`
- 复现命令：同 E18
- 口径/限制：请求 1 全序列严格相等。请求 2 见 E18。本次 T1 默认 `cargo test` 因未开 feature 跑了 0 个 e2e 用例。

### E22. 分页 KV 与连续 KV 的显存对比
- 证据类型：文档（**计划中的 3030 vs 5118 MiB 对在六仓内没有归档文件**）
- 位置：已归档的峰值显存是 W8A16 端到端增量：`tiny-llm/docs/performance/results/2026-08-18-decode-optimization.md:63`（C1 前 2494 MB → C1/C2 后 **3368 MB**，含转置副本）
- 关键 commit：`tiny-llm@f897084` / 文档 `6d0471e`
- 复现命令：
```bash
./build/tiny_llm_bench ../models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --prompt "你好" --max-tokens 64 --warmup 3 --iters 5
```
- 口径/限制：**不要背 3030/5118**——PHASE3 提纲写过这对数字，但仓库 README/results 里找不到来源。策略 1 vs 2 的正确性证据是 E20，不是显存表。3368 MB 是转置副本把峰值抬上去，不是分页节省。

### E23. paged-infer 调度器资源守恒不变量（used+free==total）有属性测试
- 证据类型：测试
- 位置：`paged-infer/src/kv_cache.rs:416` `prop_block_count_invariant`；`src/scheduler.rs:1512` `prop_resources_reclaimed_after_cancel_and_failure`
- 关键 commit：v0.2.0 属性测试合入
- 复现命令：
```bash
cd paged-infer && cargo test prop_block_count_invariant prop_resources_reclaimed
```
- 口径/限制：本次 freeze lib 144 tests 通过（含这些属性测试）。

### E24. paged-infer 内存水位线 / HOL / 优先级 / NaN / Unicode 修复各有回归测试
- 证据类型：测试
- 位置：HOL `tests/integration_tests.rs:699` `test_small_pending_request_not_blocked_by_large_one`；NaN `src/types/request.rs:267` `test_nan_parameters_rejected`；Unicode `src/engine.rs:1645` `test_stop_sequence_unicode_byte_offset_end_to_end`；优先级 `src/scheduler.rs:839` `test_priority_higher_prefill_starts_first`；水位线 `src/scheduler.rs:519` + 相关单测
- 关键 commit：paged-infer T0–T8 / T12 与优先级调度
- 复现命令：
```bash
cd paged-infer && cargo test test_small_pending_request_not_blocked_by_large_one \
  test_nan_parameters_rejected test_stop_sequence_unicode_byte_offset_end_to_end \
  test_priority_higher_prefill_starts_first
```
- 口径/限制：本次 freeze integration 15 + lib 144 通过。

### E25. paged-infer OpenAI 兼容 API + SSE + metrics 有 server integration 测试
- 证据类型：测试
- 位置：`paged-infer/tests/server_integration.rs`（`/v1/completions`、SSE `data: [DONE]`、`/metrics` Prometheus 名 `paged_*`）
- 关键 commit：v0.2.0 server 测试
- 复现命令：
```bash
cd paged-infer && cargo test --test server_integration
```
- 口径/限制：本次 freeze 37 passed。测的是 CPU 参考后端的 HTTP 契约，不是 tiny-llm GPU serving 压测。

### E26. 五仓 IN/OUT 边界声明齐全；04-inference-engine 已降级教学预览
- 证据类型：文档
- 位置：各仓 README「项目边界 / Scope」；`cuda-foundations/04-inference-engine/README.md:3-5`；`cuda-foundations/LEARNING_PATH.md:17-19`
- 关键 commit：tiny-llm `ef56907` 等 D2 边界提交；04 降级 `cuda-foundations@8483ed3`（MASTER_PLAN 记录）
- 复现命令：读五个 README 的 IN/OUT 节
- 口径/限制：边界是作品集纪律，不是运行时依赖图。tiny-llm 禁止 include cuda-foundations 头文件。

### E27. 组织级改名 cuda-kernel-academy → cuda-foundations，五仓源码旧 slug 0 命中
- 证据类型：文档 + grep
- 位置：仓库名 `aicl-lab/cuda-foundations`；namespace `cuda_foundations`（`45854fb`）
- 关键 commit：`cuda-foundations@45854fb` mechanical rename；`37bc9de` 文档站
- 复现命令：
```bash
grep -rn "cuda-kernel-academy" --exclude-dir=.git --exclude-dir=build \
  cuda-foundations triton-fused-ops cuflash-attn tiny-llm paged-infer || true
```
- 口径/限制：本 freeze 对上述五仓 **0 命中**。`aicl-lab/docs/organization-audit/` 与根目录 PHASE 计划仍保留旧名作为历史。教学品牌仍可用 “CUDA Kernel Academy”。

### E28. ncu/nsys 在本机不可用的替代证据链
- 证据类型：文档 + microbench 工具
- 位置：`tiny-llm/docs/performance/results/2026-08-18-decode-optimization.md:34-52`；`tiny-llm/src/kernel_bench.cpp`；`cuda-foundations/docs/en/guides/profiling.md`（WSL2 GPUCTRPERM runbook）
- 关键 commit：`tiny-llm@ca70de2` `perf(bench): add kernel microbenchmark for decode-path evidence`
- 复现命令：
```bash
./build/tiny_llm_kernel_bench
```
- 口径/限制：`ncu` → `ERR_NVGPUCTRPERM`；`nsys stats` importer 缺失。替代链能回答「哪个 kernel 占 decode」，不能代替 occupancy/roofline 计数器。

### E29. 性能数字诚信：benchmark 附硬件 / commit / 复现命令
- 证据类型：文档
- 位置：tiny-llm README 基准快照；`2026-08-18-decode-optimization.md` 元信息表；cuflash `docs/performance/benchmarks.md` §1.5；triton README「性能基准」；cuda-foundations `docs/en/benchmarks/index.md`
- 关键 commit：各仓实测回填提交（如 `tiny-llm@6d0471e`、`cuflash-attn` benchmarks 刷新 `6860cbc`）
- 复现命令：各归档文件顶部的复现命令
- 口径/限制：单一硬件 RTX 3060 Laptop。causal skip 负结果也归档（E8）。禁止把计划数字（E22 的 3030/5118）当实测。

### E30. 六仓 GitHub 可见 + phase tag + landing repo
- 证据类型：git tag + GitHub API
- 位置：landing `https://github.com/aicl-lab/aicl-lab`；五仓 `phase-2-e`（在「link portfolio」提交上，不是本轮 ROADMAP HEAD）
- 关键 commit：五仓 `docs: link portfolio landing repo`；aicl-lab `1ab3e66` / `42fad33`
- 复现命令：
```bash
gh api repos/aicl-lab/aicl-lab --jq .full_name
# 本 freeze：六个 full_name 均返回 aicl-lab/<name>
git -C tiny-llm tag --list 'phase-*'
```
- 口径/限制：K 阶段后六仓 ahead 0；五仓 tag = `phase-3-docs`，meta tag = `phase-3-interview`（`9e0b4f7`）。tag 链：`phase-2-e` → `phase-3-docs`（五仓）→ `phase-3-interview`（meta）。
