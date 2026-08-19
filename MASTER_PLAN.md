# AICL-Lab 五仓 Master Development Plan

> **版本**：2026-08-18
> **目标**：将五个仓库从"能跑"推进到"面试可讲"，每个项目有清晰的深度锚点和完成证据。
> **执行方**：低成本 AI 编程模型（如 Claude Haiku、GPT-4o-mini、DeepSeek-Coder 等）。
> **原则**：一次只做一个任务，验收通过再做下一个。不凭直觉优化，不跳过验证命令。
> **Phase 1 状态**：A-E 阶段已由低成本模型执行完毕（完成报告见本文件第 7 节勾选状态）。
> **下一步**：Phase 2 计划见 [`PHASE2_PLAN.md`](PHASE2_PLAN.md)（重命名 cuda-foundations + 边界收口 + decode 性能攻坚 + 分页 KV 端到端）。
> 后续任务请以 PHASE2_PLAN.md 为准，本文件仅保留 Phase 1 历史任务记录。
> **Phase 3 状态**：✅ T1–T10 完成（aicl-lab@phase-3-interview）；五仓 `phase-3-docs` tag 已推送。
> **当前唯一执行入口**：[`PLAN_v3.md`](PLAN_v3.md)（PLAN v3：仓库完成状况分析 + 阶段 K 收尾 / 阶段 I 面试执行 / 阶段 D 可选深度增量）。
> **阶段 K 状态**：✅ K1–K4 完成，六仓 ahead 0，tag 链 `phase-2-e → phase-3-docs → phase-3-interview` 完整。
> **阶段 I 状态**：✅ I0–I7 完成（aicl-lab@phase-i-ready）；见 [`PLAN_I.md`](PLAN_I.md)。
> **Phase 2 状态**：A0–B5 ✅ / C0–C3 ✅ / D0–D5 ✅ / **E0–E4 ✅（2026-08-18 作品集收尾完成）**；
> 五仓已打 `phase-2-e` tag 并推送，进入"面试就绪冻结"态；landing 页见 <https://github.com/aicl-lab/aicl-lab>。

---

## 0. 执行协议（所有任务通用）

### 0.1 每条任务必须遵守

1. **单任务单提交**：一个任务只改该任务列出的文件，完成后 `git commit`。
2. **验收命令必须全绿**：任务结尾的 build + test 命令必须通过，失败就继续修。
3. **禁止改测试让错误实现通过**。只有产品行为有意变更时才允许同步修改测试。
4. **不要顺手重构**：发现计划外问题，记录为 NOTE 但不扩大修改范围。
5. **性能数字只写实测**：任何 benchmark 数字必须先由脚本在真实 GPU 上测出来。
6. **提交前删除临时文件**：不要提交 `.bak`、`*.tmp`、core dump。

### 0.2 各仓库基线命令

| 仓库 | 构建命令 | 测试命令 |
|------|----------|----------|
| cuda-foundations | `cmake --preset default && cmake --build --preset default` | `ctest --preset default` |
| triton-fused-ops | `pip install -e '.[dev]'` | `pytest -q` |
| cuflash-attn | `cmake --preset release && cmake --build --preset release` | `ctest --preset release --output-on-failure` |
| tiny-llm | `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON && cmake --build build -j` | `./build/tiny_llm_tests` |
| paged-infer | `cargo build` | `cargo test` |

### 0.3 任务规模标签

| 标签 | 含义 | 预估时间 |
|------|------|----------|
| 🔴 P0 | 正确性修复，必须完成 | 1-4 小时/任务 |
| 🟠 P1 | 面试证据，强烈建议 | 2-8 小时/任务 |
| 🟡 P2 | 加分项，有余力再做 | 2-4 小时/任务 |
| 🟢 P3 | 文档/整理，低优先级 | 0.5-2 小时/任务 |

---

## 1. 全局优先级总览

```
阶段 A（P0，先做）：正确性修复 —— 修完才能谈性能
  A1. tiny-llm: QKV layout 统一
  A2. tiny-llm: GQA 映射实现
  A3. tiny-llm: RoPE 进入计算路径
  A4. tiny-llm: 模型权重契约完整
  A5. paged-infer: fmt/clippy 修复 + 执行输出校验 + 内存水位线 + 队头阻塞 + 生命周期钩子 + NaN 校验 + Unicode 修复 + 文档对齐

阶段 B（P0）：基准与对比基线 —— 建立可信数字
  B1. tiny-llm: benchmark 驱动
  B2. tiny-llm: llama.cpp 对比方法论文档
  B3. cuflash-attn: benchmark 刷新 + 文档更新
  B4. triton-fused-ops: GPU benchmark 跑一次

阶段 C（P1）：深度锚点 —— 每个项目的核心叙事
  C1. tiny-llm: CUDA Graphs 或 Speculative Decoding
  C2. cuflash-attn: 一轮优化迭代（双缓冲/异步拷贝 或 warp 级 softmax 归约）
  C3. triton-fused-ops: Triton SGEMM + torch.library 注册

阶段 D（P1）：跨仓对接与边界文档
  D1. tiny-llm + paged-infer: 分页 KV + continuous batching 端到端
  D2. 各项目 README 补完 IN/OUT 边界
  D3. 04-inference-engine 降级为"教学预览"
  D4. triton-fused-ops FlashAttention 降级为"参考实现"

阶段 E（P2/P3）：收尾
  E1. cuda-foundations 剩余问题修复 ✅（2026-08-18 复核 d880996：build+ctest 209/209 0 失败；04 模块 GPU 测试在本机 skip 属既有环境异常，见执行报告）
  E2. paged-infer: chunked prefill 或优先级调度（选一）✅（2026-08-18, paged-infer@a69b146 选优先级调度）
  E3. cuflash-attn: FlashDecoding/Split-KV ✅（2026-08-18, cuflash-attn@9f65df7）
  E4. cuda-foundations 改名 cuda-foundations（可选）✅（2026-08-18：`cuda-kernel-academy` → `cuda-foundations` 全链路改名完成，见 PHASE2_PLAN 任务 B0）
```

---

## 2. 阶段 A：正确性修复（P0，必须完成）

### 任务 A1：tiny-llm QKV layout 统一

**背景**：QKV 投影输出可能是 token-major `[B*S, H*D]`，但 attention kernel 可能按 head-major `[B, H, S, D]` 解读。两者不一致会导致 prefill/decode 数值错误。

**改动文件**：
- `kernels/attention.cu` / `kernels/attention.cuh`
- `src/transformer.cpp`
- `tests/test_kernels.cu`

**实施步骤**：

1. 在 `kernels/attention.cuh` 的文档注释中**明确写出**每个 kernel 期望的输入 layout：
   ```
   // Q: [num_tokens, num_q_heads, head_dim]  (token-major)
   // K_cache: [num_kv_heads, max_seq_len, head_dim]  (head-major)
   // V_cache: [num_kv_heads, max_seq_len, head_dim]  (head-major)
   ```
2. 在 `src/transformer.cpp` 中找到调用 attention kernel 的位置，确认传入的 Q/K/V 指针对应的实际 layout 与 kernel 声明一致。
3. 如果发现不一致，在 `transformer.cpp` 中插入一个**显式 layout 转换函数**（如 `reorder_qkv_token_to_head_major`），并添加单元测试。
4. 在 `tests/test_kernels.cu` 中增加一个测试：构造 `S>1, H>1` 的 Q/K/V，调用 attention kernel，与 CPU 参考实现逐元素比较，容差 `1e-2f`。

**验收命令**：
```bash
cmake --build build -j$(nproc)
./build/tiny_llm_tests --gtest_filter='*Attention*'
./build/tiny_llm_tests
# 期望：全部通过，新增测试通过
```

**完成标准**：Attention kernel 的文档注释写明了输入 layout；测试覆盖了 `S>1, H>1` 的 prefill 场景。

---

### 任务 A2：tiny-llm GQA 映射实现

**背景**：当前代码可能没有实现 `kv_head = q_head / group_size` 的映射，导致 GQA 模型（如 Qwen2.5 的 14→2）的 KV head 索引错误。

**改动文件**：
- `kernels/attention.cu`
- `src/transformer.cpp`
- `tests/test_kernels.cu`

**实施步骤**：

1. 在 `kernels/attention.cu` 的 attention kernel 中，找到读取 K/V cache 的索引计算，确认是否包含 `kv_head = q_head / (num_q_heads / num_kv_heads)` 的映射。
2. 如果缺失，添加：
   ```cuda
   int group_size = num_q_heads / num_kv_heads;
   int kv_head = q_head / group_size;
   ```
3. 在 `tests/test_kernels.cu` 中增加一个 GQA 测试：
   - `num_q_heads = 4, num_kv_heads = 2, seq_len = 8, head_dim = 64`
   - 随机初始化 Q/K/V，与 CPU 参考（手动实现 GQA 映射）逐元素比较
   - 容差 `1e-2f`

**验收命令**：
```bash
cmake --build build -j$(nproc)
./build/tiny_llm_tests --gtest_filter='*GQA*'
# 期望：GQA 测试通过
```

**完成标准**：GQA 映射在 kernel 中实现，有独立测试覆盖 `Hq != Hkv` 场景。

---

### 任务 A3：tiny-llm RoPE 进入计算路径

**背景**：RoPE kernel 可能已实现但未被 `transformer.cpp` 调用，导致所有位置相关 logits 错误。

**改动文件**：
- `src/transformer.cpp`
- `kernels/rope.cu` / `kernels/rope.cuh`
- `tests/test_kernels.cu`

**实施步骤**：

1. 在 `src/transformer.cpp` 的 `TransformerLayer::forward` 中，找到 RMSNorm 之后、attention 之前的代码路径，确认是否调用了 RoPE kernel。
2. 如果未调用，在 Q 和 K 投影之后、attention 之前插入 RoPE 调用：
   ```cpp
   kernels::rope(q_proj, num_q_heads, head_dim, position, stream_);
   kernels::rope(k_proj, num_kv_heads, head_dim, position, stream_);
   ```
3. 确认 RoPE 使用 half-split 语义（`rotate_half` 而非 interleaved pair）：
   ```
   x = [x_first, x_second]
   rotate_half(x) = [-x_second, x_first]
   y = x * cos + rotate_half(x) * sin
   ```
4. 在 `tests/test_kernels.cu` 中增加一个 RoPE 端到端测试：
   - 构造 Q/K 张量，已知 position，调用 RoPE kernel
   - 与 Python/PyTorch 参考实现逐元素比较，容差 `1e-3f`

**验收命令**：
```bash
cmake --build build -j$(nproc)
./build/tiny_llm_tests --gtest_filter='*RoPE*'
# 期望：RoPE 测试通过，且 transformer 单层测试也通过
```

**完成标准**：RoPE 在 transformer 前向路径中被调用，有独立测试覆盖。

---

### 任务 A4：tiny-llm 模型权重契约完整

**背景**：GGUF 模型可能包含 bias 张量、tied output embedding（lm_head 权重与 token embedding 共享）等，当前加载逻辑可能忽略或处理不正确。

**改动文件**：
- `src/gguf_parser.cpp`
- `src/model_loader.cpp`
- `tests/test_model_loader.cpp`

**实施步骤**：

1. 在 `src/gguf_parser.cpp` 中检查 GGUF metadata 是否声明了 bias 张量（如 `*.bias` 或 `*.attn_q.bias`）。如果存在但当前代码不支持，添加**显式报错**（不是静默忽略）：
   ```cpp
   if (tensor_name.find(".bias") != std::string::npos) {
       return Result<T>::err("Bias tensors not yet supported: " + tensor_name);
   }
   ```
2. 检查是否存在 tied output embedding（`token_embd.weight` 与 `output.weight` 是同一个 tensor id）。如果 GGUF 中 `output.weight` 不存在但 `token_embd.weight` 存在，将 `lm_head` 指向 `token_embd.weight`。
3. 在 `tests/test_model_loader.cpp` 中增加一个测试：构造一个最小 GGUF（或使用 mock），验证 bias 和 tied embedding 的处理路径。

**验收命令**：
```bash
cmake --build build -j$(nproc)
./build/tiny_llm_tests --gtest_filter='*ModelLoader*'
# 期望：新增测试通过
```

**完成标准**：bias 和 tied embedding 的路径有明确行为（支持或显式报错），有测试覆盖。

---

### 任务 A5：paged-infer P0 修复包

**背景**：paged-infer 有 8 个 P0 任务（T0-T8），分别修复 fmt/clippy、执行输出校验、内存水位线、队头阻塞、生命周期钩子、NaN 校验、Unicode 偏移、文档对齐。

**重要**：这些任务已在 `paged-infer/DEVELOPMENT_PLAN.md` 中详细描述。**请直接按照该文档的 T0-T8 逐任务执行**，不要重复发明。

**执行顺序**：T0 → T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8

**每个任务的验收命令**：参见 `paged-infer/DEVELOPMENT_PLAN.md` 中各任务的"验收"部分。

**总体验收**：
```bash
cd paged-infer
cargo fmt --all -- --check   # 无输出，退出码 0
cargo clippy --all-targets -- -D warnings  # 无 warning/error
cargo test                   # 全部通过
```

**完成标准**：T0-T8 全部完成，CI 绿色，文档与代码一致。

---

## 3. 阶段 B：基准与对比基线（P0）

### 任务 B1：tiny-llm benchmark 驱动

**背景**：需要产出 `tiny_llm_bench` 可执行文件，输出可复现的标准表格。

**注意**：此任务已在 `tiny-llm/DEVELOPMENT_PLAN.md` 的任务 2.1 中详细描述。**请直接按照该文档执行**。

**简要步骤**：
1. 新增 `src/benchmark.cpp`
2. 修改 `CMakeLists.txt` 添加 `tiny_llm_bench` 目标
3. 实现 CLI：`--prompt`、`--max-tokens`、`--warmup`、`--iters`、`--json`
4. 指标：TTFT、TPOT、decode tok/s、峰值显存

**验收命令**：
```bash
cmake --build build -j$(nproc)
./build/tiny_llm_bench model.gguf --prompt "你好" --max-tokens 64 --warmup 3 --iters 10
# 期望：输出标准表格，有 mean/p50/p95/min/max
```

**完成标准**：`tiny_llm_bench` 可执行，输出标准表格；README 状态表更新。

---

### 任务 B2：tiny-llm llama.cpp 对比方法论文档

**注意**：此任务已在 `tiny-llm/DEVELOPMENT_PLAN.md` 的任务 2.2 中详细描述。**请直接按照该文档执行**。

**改动文件**：新增 `docs/performance/benchmark-methodology.md`

**完成标准**：文档中的命令可原样复制执行；表格有真实数字（有 GPU 时补）。

---

### 任务 B3：cuflash-attn benchmark 刷新

**背景**：当前 benchmark 文档是 v0.4.0 快照，需要刷新到当前版本。

**改动文件**：
- `docs/performance/benchmarks.md`
- 可能需要更新 `benchmarks/` 目录下的 benchmark 代码

**实施步骤**：

1. 在可用的 GPU 上运行现有 benchmark：
   ```bash
   cmake --preset release && cmake --build --preset release
   ./build/release/cuflash_attn_bench
   ```
2. 记录输出，更新 `docs/performance/benchmarks.md` 中的数字和日期。
3. 补充 `head_dim=128` 的 benchmark 数据（如果当前只覆盖了 `head_dim=64`）。
4. 在文档中明确标注：GPU 型号、驱动版本、CUDA 版本、commit hash。

**验收命令**：
```bash
cmake --preset release && cmake --build --preset release
./build/release/cuflash_attn_bench
# 期望：benchmark 正常运行，输出数字
```

**完成标准**：benchmark 文档更新到当前版本，有硬件/软件版本信息。

---

### 任务 B4：triton-fused-ops GPU benchmark

**背景**：triton-fused-ops 的 benchmark 基础设施已存在，但缺少真实 GPU 数字。

**改动文件**：
- `README.md`（更新 benchmark 数字）
- 可能在 `tests/benchmarks/` 中调整 benchmark 参数

**实施步骤**：

1. 在 GPU 上运行 benchmark：
   ```bash
   python -m pytest tests/benchmarks/ -v
   ```
2. 记录输出，更新 `README.md` 中的 benchmark 数字。
3. 明确标注：GPU 型号、PyTorch 版本、Triton 版本。

**验收命令**：
```bash
python -m pytest tests/benchmarks/ -v
# 期望：benchmark 正常运行，输出数字
```

**完成标准**：README 中有真实 GPU benchmark 数字。

---

## 4. 阶段 C：深度锚点（P1，强烈建议）

### 任务 C1：tiny-llm CUDA Graphs 加速 decode

**背景**：decode 阶段每步有大量 kernel launch 开销（24 层 × 每层约 10+ kernel），CUDA Graphs 可以显著降低 TPOT。

**注意**：此任务已在 `tiny-llm/DEVELOPMENT_PLAN.md` 的阶段 3（任务 3.1-3.3）中详细描述。**请直接按照该文档执行**。

**摘要**：
- 3.1：`visible_len` 参数间接化（从 kernel 参数移到 device memory）
- 3.2：decode graph capture/replay
- 3.3：正确性差分测试与文档

**完成标准**：`TLLM_CUDA_GRAPHS=1` 时 decode 输出与关闭时逐 token 一致；benchmark 有 before/after 对比数字。

---

### 任务 C2：cuflash-attn 一轮优化迭代

**背景**：当前与 PyTorch SDPA 的比值只有 0.42×–0.67×，需要一轮有数字的优化迭代。

**改动文件**：
- `src/forward/flash_attention_forward_typed.cu` 或 `src/forward/flash_attention_forward_wmma.cu`
- `docs/performance/benchmarks.md`

**实施步骤**（选 1-2 项）：

**选项 1：双缓冲 / 异步拷贝（cp.async）**
1. 在 forward kernel 中，将 global → shared memory 的加载改为 `cp.async`（需要 SM 8.0+）。
2. 使用双缓冲：一个 buffer 计算时，另一个 buffer 加载下一块数据。
3. 记录 before/after 的 ncu 指标（memory throughput、stall reasons）。

**选项 2：warp 级 softmax 归约重构**
1. 当前 softmax 归约可能使用了多次 shared memory 往返。改为 warp shuffle 归约，减少 shared memory 访问。
2. 记录 before/after 的 ncu 指标。

**验收命令**：
```bash
cmake --preset release && cmake --build --preset release
./build/release/cuflash_attn_bench --benchmark_filter=Forward
# 期望：优化后 benchmark 数字有提升
```

**完成标准**：优化有 before/after 数字 + profiling 证据；文档更新。

---

### 任务 C3：triton-fused-ops Triton SGEMM + torch.library 注册

**背景**：需要补一个 Triton SGEMM（与 cuda-foundations 的 CUDA SGEMM 做"同题异构"对比），并给算子注册 `torch.library` 自定义 op。

**改动文件**：
- 新增 `triton_ops/kernels/sgemm.py`
- 新增 `triton_ops/ops.py`（torch.library 注册）
- `tests/` 新增对应测试

**实施步骤**：

1. **实现 Triton SGEMM**：
   - 参照 `triton_ops/kernels/gated_mlp.py` 的结构
   - 实现 tiled SGEMM：`tl.load` from global → `tl.dot` → `tl.store` to global
   - 支持 `M, N, K` 可配置，block size 可调
   - 与 PyTorch `torch.mm` 差分测试

2. **注册 torch.library 自定义 op**：
   ```python
   import torch
   from torch.library import custom_op

   @custom_op("triton_ops::fused_rmsnorm_rope", mutates_args=())
   def fused_rmsnorm_rope(x: torch.Tensor, weight: torch.Tensor, ...) -> torch.Tensor:
       ...
   ```
   为三个算子（RMSNorm+RoPE、Gated MLP、SGEMM）各注册一个 custom op。

3. **测试**：
   - SGEMM 与 `torch.mm` 差分测试
   - custom op 可 `import triton_ops` 后直接调用
   - 边界 shape（M=1, N=1, K=1, 非 16 倍数等）

**验收命令**：
```bash
python -m pytest tests/ -v -k "sgemm or ops"
# 期望：新增测试通过
```

**完成标准**：Triton SGEMM 通过差分测试；3 个 custom op 可 import 直接调用。

---

## 5. 阶段 D：跨仓对接与边界文档（P1）

### 任务 D1：tiny-llm + paged-infer 分页 KV 端到端

**背景**：当前 paged-infer 与 tiny-llm 的对接使用"策略 2"（连续 KV），需要启用"策略 1"（分页 KV）。

**注意**：此任务在 `paged-infer/DEVELOPMENT_PLAN.md` 的任务 T11 中描述。**请参考该文档执行**。

**简要步骤**：
1. 在 tiny-llm 侧确保 `ffi.h` 的分页 KV 接口完整（`tinyllm_allocate_block`、`tinyllm_free_block`、`tinyllm_set_block_table` 等）
2. 在 paged-infer 侧实现 `TinyLlmExecutor` 的分页 KV 调用路径
3. 端到端测试：3 并发请求，分页 KV + continuous batching + 真实 token 生成

**验收命令**：
```bash
cd paged-infer
TINY_LLM_DIR=... TINY_LLM_MODEL=... cargo test --features tiny-llm --test tiny_llm_backend -- --nocapture
# 期望：3 并发请求全部成功，输出与 llama.cpp 一致
```

**完成标准**：分页 KV 路径跑通，并发请求资源守恒不变量成立。

---

### 任务 D2：各项目 README 补完 IN/OUT 边界

**背景**：每个项目的 README 需要有明确的"负责什么 / 不负责什么"声明。

**改动文件**：各项目的 `README.md`

**实施步骤**：

对每个仓库，在 README 的"项目概述"或"项目边界"部分，确保有以下内容：

1. **cuda-foundations**：
   - IN：CUDA 编程模型、GEMM 优化阶梯、算子库设计、profiling 方法
   - OUT：生产级推理运行时（见 tiny-llm）、完整 FlashAttention（见 cuflash-attn）、Serving（见 paged-infer）
   - 04-inference-engine 标注为"教学预览，非独立作品"

2. **triton-fused-ops**：
   - IN：Triton 融合算子、autotuner、torch.library 注册、验证方法论
   - OUT：CUDA C++ kernel 学习（见 cuda-foundations）、完整 FlashAttention（见 cuflash-attn）、模型加载与生成（见 tiny-llm）
   - FlashAttention kernel 标注为"cuflash-attn 的参考实现"

3. **cuflash-attn**：
   - IN：FlashAttention 前向+反向、多精度、causal mask、FlashDecoding、优化叙事
   - OUT：GEMM 基础（见 cuda-foundations）、Triton 实现（见 triton-fused-ops）、完整推理运行时（见 tiny-llm）

4. **tiny-llm**：
   - IN：GGUF 加载、W8A16 量化推理、KV Cache、tokenizer、采样、端到端生成、性能基准、CUDA Graphs
   - OUT：调度/批处理/paged KV（见 paged-infer）、FlashAttention 深挖（见 cuflash-attn）、Triton 算子（见 triton-fused-ops）

5. **paged-infer**：
   - IN：Paged KV、continuous batching、准入控制、OpenAI 兼容 API、属性测试
   - OUT：计算 kernel（见 tiny-llm）、模型加载（见 tiny-llm）、FlashAttention（见 cuflash-attn）

**验收命令**：
```bash
# 检查每个 README 是否有 "IN" / "OUT" 或 "Scope" / "Out of Scope" 段落
grep -l "Out of Scope\|OUT\|不负责\|明确不做" */README.md
# 期望：5 个 README 都有
```

**完成标准**：所有 5 个 README 都有明确的 IN/OUT 边界声明。

---

### 任务 D3：04-inference-engine 降级为"教学预览"

**背景**：cuda-foundations 的 04-inference-engine 与 tiny-llm 职责重叠，需要明确标注为教学预览。

**改动文件**：
- `cuda-foundations/04-inference-engine/README.md`
- `cuda-foundations/LEARNING_PATH.md`
- `cuda-foundations/README.md`

**实施步骤**：

1. 在 `04-inference-engine/README.md` 顶部添加：
   ```markdown
   > ⚠️ **教学预览**：本模块是 tiny-llm 的简化预习版，用于展示 kernel/内存/流如何
   > 组装成小系统。**真实推理运行时见 [tiny-llm](https://github.com/aicl-lab/tiny-llm)**。
   > 本模块不追求模型兼容性、量化精度或推理性能。
   ```
2. 在 `LEARNING_PATH.md` 的阶段 4 描述中，将 04-inference-engine 标注为"教学预览，非独立作品"。
3. 在 `README.md` 的项目地图中，04 行尾加 `（教学预览）`。

**验收命令**：
```bash
grep -r "教学预览\|tutorial preview\|简化预习" cuda-foundations/04-inference-engine/README.md cuda-foundations/LEARNING_PATH.md
# 期望：两处都有明确标注
```

**完成标准**：04-inference-engine 的定位在教学文档中明确降级。

---

### 任务 D4：triton-fused-ops FlashAttention 降级为"参考实现"

**背景**：triton-fused-ops 中的 FlashAttention 应该定位为 cuflash-attn 的参考实现，而非独立交付物。

**改动文件**：
- `triton-fused-ops/README.md`
- `triton-fused-ops/triton_ops/kernels/flash_attention.py`（文档字符串）

**实施步骤**：

1. 在 `README.md` 的快速示例和功能描述中，将 FlashAttention 标注为：
   ```markdown
   > ℹ️ **定位**：Triton FlashAttention 是 [cuflash-attn](https://github.com/aicl-lab/cuflash-attn)
   > 的独立参考实现，用于验证 CUDA C++ 版本的正确性。完整 FlashAttention 前后向 +
   > 优化叙事见 cuflash-attn。
   ```
2. 在 `triton_ops/kernels/flash_attention.py` 的文档字符串中添加同样的说明。

**验收命令**：
```bash
grep -r "参考实现\|reference implementation\|cuflash-attn" triton-fused-ops/README.md triton-fused-ops/triton_ops/kernels/flash_attention.py
# 期望：两处都有明确标注
```

**完成标准**：FlashAttention 在 triton-fused-ops 中的定位明确为参考实现。

---

## 6. 阶段 E：收尾（P2/P3，有余力再做）

### 任务 E1：cuda-foundations 剩余问题修复

**注意**：此任务已在 `cuda-foundations/DEV_PLAN.md` 中详细描述（T1-T12）。**请直接按照该文档执行**。

**简要列表**：
- T1：删除或修复 `initRandomMatrixGPU`
- T2：MoE router 输入校验
- T3：conv2d/sparse/fusion 测试覆盖补齐
- T4：权重文件读取健壮性
- T5：Tensor 输入校验
- T6：03 模块尊重根 CMake 选项
- T7：统一 GoogleTest FetchContent
- T8：清理编译警告
- T9：Nsight 运行手册
- T10：一键基准脚本

**验收命令**：
```bash
cd cuda-foundations
cmake --preset default && cmake --build --preset default
ctest --preset default
# 期望：全部通过
```

---

### 任务 E2：paged-infer chunked prefill 或优先级调度

**背景**：vLLM/SGLang 面试常见追问是 chunked prefill 和优先级调度。选一个实现，作为"我理解这个主题"的证据。

**改动文件**：`paged-infer/src/scheduler.rs`、`paged-infer/tests/`

**选项 A：chunked prefill**
- 将长 prefill 拆成多个 chunk，每步只处理一个 chunk
- 优点：减少 prefill 对 decode 的延迟影响
- 复杂度：中

**选项 B：优先级调度**
- 为请求添加优先级字段，高优先级请求先被调度
- 优点：展示调度策略的灵活性
- 复杂度：低

**验收命令**：
```bash
cd paged-infer
cargo test
# 期望：新增测试通过
```

**完成标准**：选一个实现，有测试覆盖，文档说明设计选择。

---

### 任务 E3：cuflash-attn FlashDecoding / Split-KV

**背景**：FlashDecoding 是 decode 阶段（query_len=1）的 KV 分块并行 + reduce，是推理加速面试的高频主题。

**改动文件**：
- 新增 `src/forward/flash_decoding.cu`
- `tests/` 新增对应测试
- `docs/` 新增说明文档

**实施步骤**：

1. 实现 FlashDecoding kernel：
   - 将 KV 沿序列维度分块，每个 block 计算一个 chunk 的 attention
   - 最后 reduce 各 chunk 的 partial results
2. 与标准 FlashAttention decode 做差分测试
3. benchmark 对比（尤其是长序列场景）

**验收命令**：
```bash
cmake --preset release && cmake --build --preset release
./build/release/cuflash_attn_bench --benchmark_filter=Decode
# 期望：FlashDecoding 测试通过，benchmark 有数字
```

**完成标准**：FlashDecoding kernel 通过差分测试，有 benchmark 数字。

---

### 任务 E4：cuda-foundations 改名 cuda-foundations（可选）

**背景**：当前名称 "cuda-foundations" 范围偏窄（漏掉了 04-inference-engine 的系统组装内容），品牌后缀 "Academy" 与同级项目风格不一致，且暗示持续扩张但仓库已 maintenance mode。

**推荐新名**：`cuda-foundations`

**影响范围**（63 个文件）：
- 所有 `.h`、`.cuh`、`.cu`、`.cpp` 中的 `cuda_academy` namespace → `cuda_foundations`
- 所有 `#include "cuda_academy/..."` → `#include "cuda_foundations/..."`
- `CMakeLists.txt` 中的 project name 和 target name
- `README.md`、`README.zh-CN.md`、`LEARNING_PATH.md` 中的仓库名引用
- `docs/` 中的 VitePress 配置（base URL）
- `.github/workflows/` 中的 CI 配置
- `CHANGELOG.md` 中的链接
- `package.json` 中的 name

**执行方式**：
1. 先在 GitHub 上重命名仓库（Settings → Rename）
2. 本地 `git pull` 更新 remote
3. 用 `sed` 批量替换 namespace 和 include 路径
4. 更新 CMake、docs、CI 配置
5. 运行完整构建和测试确认无破坏
6. 提交并推送

**注意**：此任务涉及大量机械替换，建议在所有其他任务完成后、面试前统一执行。

**验收命令**：
```bash
cmake --preset default && cmake --build --preset default
ctest --preset default
# 期望：全部通过，无 broken reference
```

**完成标准**：仓库名、namespace、include 路径、文档链接全部更新，构建和测试全绿。

---

## 7. 总完成定义（Definition of Done）

当以下所有条件满足时，五仓作品集可以宣布"面试就绪"：

### 7.1 每个项目都满足

- [ ] 能从干净环境构建（CMake/Cargo/pip）
- [ ] 有独立参考实现的差分测试
- [ ] 正常 / 边界 / 失败路径都有测试
- [ ] README 有 IN/OUT 边界、真实完成度、已知限制
- [ ] 能 10 分钟讲清：瓶颈、设计选择、验证方法、下一步

### 7.2 阶段 A 全部完成

- [x] tiny-llm: QKV layout 统一（2026-08-18, tiny-llm@35bfabc）
- [x] tiny-llm: GQA 映射实现（2026-08-18, tiny-llm@fdbabcc）
- [x] tiny-llm: RoPE 进入计算路径（2026-08-18, tiny-llm@1038639）
- [x] tiny-llm: 模型权重契约完整（2026-08-18, tiny-llm@d234157）
- [x] paged-infer: T0-T8 全部完成（2026-08-18 复核：fmt/clippy/cargo test 全绿，215 passed；T0-T8 逐任务见 paged-infer DEVELOPMENT_PLAN 执行日志）

### 7.3 阶段 B 全部完成

- [x] tiny-llm: benchmark 驱动 + llama.cpp 对比文档（2026-08-18：bench 复核通过；对比实测 tiny-llm@753d913）
- [x] cuflash-attn: benchmark 刷新（2026-08-18, cuflash-attn@52c4bfd，RTX 3060 实测 + head_dim=128）
- [x] triton-fused-ops: GPU benchmark 数字（2026-08-18, triton-fused-ops@b16a4c9，RTX 3060 / torch 2.5.1 / triton 3.1.0 实测）

### 7.4 阶段 C 至少完成 1 项

- [x] tiny-llm: CUDA Graphs（2026-08-18 复核：gated 差分测试通过，TPOT -6.8%，a2a9c58/efd035a）
- [x] cuflash-attn: 一轮优化迭代（2026-08-18, PHASE2_NEXT_E E2b：causal 边界块跳过，before/after 见 docs/performance/causal-boundary-skip.md）
- [x] triton-fused-ops: Triton SGEMM + torch.library（2026-08-18, PHASE2_NEXT_E E1a/E1b：SGEMM 差分测试 + torch.ops.triton_ops.* 注册）

### 7.5 阶段 D 全部完成

- [x] tiny-llm + paged-infer: 分页 KV 端到端（2026-08-18：策略 1 block_tables + 3 并发 e2e 与 llama.cpp greedy 对齐，资源守恒测试全绿，`phase-2-d` tag）
- [x] 5 个 README 都有 IN/OUT 边界（2026-08-18：cuda-foundations@3b73d7b, tiny-llm@ef56907, paged-infer@7d10389, cuflash-attn@d9ab221, triton 已有）
- [x] 04-inference-engine 降级标注（2026-08-18, cuda-foundations@8483ed3）
- [x] triton-fused-ops FlashAttention 降级标注（2026-08-18, triton-fused-ops@dbab4e0）

---

## 8. 面试讲述清单（完成后自查）

### 8.1 每个项目 10 分钟讲述

| 项目 | 核心叙事 |
|------|----------|
| cuda-foundations | "我写的是 GEMM 阶梯，但真正交付的是用测量数据说话的能力" |
| triton-fused-ops | "CUDA 给我底层控制力，Triton 给我开发速度；我用同一个 GEMM 把两者的工程权衡量化了" |
| cuflash-attn | "我能手推 online softmax、解释为什么不物化 O(N²)、说出我的实现和 FA2/FA3 每一档差距来自哪里" |
| tiny-llm | "我做的不是低配 llama.cpp，而是一个能精确回答瓶颈在反量化、访存还是 launch 开销的最小运行时" |
| paged-infer | "计算面（P4）和控制面（P5）通过一个最小 FFI 契约解耦，这本身就是 AI Infra 的分层意识" |

### 8.2 常见面试追问准备

- [ ] 为什么不用 llama.cpp/vLLM？——学习目的、可控性、从它们学到的设计
- [ ] W8A16 vs Q4_K_M 量化精度差异如何影响 token 输出？
- [ ] KV cache 内存布局为什么选这个？
- [ ] TTFT/TPOT 怎么测量？口径是什么？
- [ ] 你的 FlashAttention 和 FA2/FA3 差距在哪里？
- [ ] PagedAttention 为什么能把 KV 浪费降到 <5%？
- [ ] 没有抢占的边界在哪里？vLLM 的 swap 怎么补这个缺口？
- [ ] 什么时候选 Triton 什么时候选 CUDA C++？

---

## 9. 执行顺序建议

```
第 1 周：A1 → A2 → A3 → A4（tiny-llm P0 正确性）
第 2 周：A5（paged-infer P0 修复包，T0-T8）
第 3 周：B1 → B2（tiny-llm benchmark） + B3 → B4（cuflash-attn + triton benchmark）
第 4 周：C1 或 C2 或 C3（选一个深度锚点）
第 5 周：D1 → D2 → D3 → D4（边界文档与跨仓对接）
第 6 周：E1 → E2 → E3 → E4（收尾，可选）
```

每个任务完成后，在对应 `[ ]` 前打 `[x]`，记录完成日期和 commit hash。

---

## 10. 附录：各仓库现有 DEV_PLAN 索引

| 仓库 | 现有 DEV_PLAN 文件 | 覆盖内容 |
|------|-------------------|----------|
| cuda-foundations | `DEV_PLAN.md` | T1-T12（剩余修复 + 基准 + 工程化） |
| tiny-llm | `DEVELOPMENT_PLAN.md` | 阶段 1-4（半成品收尾 + benchmark + CUDA Graphs + 工程完整性） |
| paged-infer | `DEVELOPMENT_PLAN.md` | T0-T13（P0 正确性 + P1 补全 + P2 冻结） |
| cuflash-attn | `ROADMAP.md` | 阶段 1-4（数据刷新 + 优化迭代 + 推理扩展 + 输出沉淀） |
| triton-fused-ops | `ROADMAP.md` | 面试前建议 + 可选扩展 + 明确不做 |

本 `MASTER_PLAN.md` 是跨仓库的总协调文档。各仓库的 DEV_PLAN/ROADMAP 是具体任务的详细执行说明。执行时，**先看本 MASTER_PLAN 确定优先级和顺序，再到对应仓库的 DEV_PLAN 中查看具体实施步骤**。