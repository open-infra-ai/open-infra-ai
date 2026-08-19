# Freeze Audit — 2026-08-18

> Phase 3 T1。本文件只记录**本次实际跑过的命令**与输出摘要。数字不是从记忆回填的。
> 审计时五仓已有未推送的 `docs:` 提交（对齐 ROADMAP）；测试覆盖的是这些 docs 提交下的源码（docs 提交不改 kernel）。

## 1. 环境

| 项 | 值 |
|----|----|
| 日期（UTC） | 2026-08-18T12:32Z – 12:44Z |
| 主机 | WSL2 `Linux 6.18.33.2-microsoft-standard-WSL2` |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU，6144 MiB |
| 驱动 | 591.44 |
| CUDA toolkit | nvcc 12.0.140（`Build cuda_12.0.r12.0/compiler.32267302_0`） |
| 工作根目录 | `/home/shane/github/aicl` |
| 模型（tiny-llm 门控测试） | `/home/shane/github/aicl/models/qwen2.5-0.5b-instruct-q4_k_m.gguf` |

## 2. 六仓 git 状态（测试结束后）

`phase-2-e` 打在「docs: link portfolio landing repo」那次提交上。其后各仓有 1–2 个未推送的 `docs:` commit（本轮 ROADMAP/README 对齐），**源码与 `phase-2-e` 相同**。

| 仓库 | HEAD | 相对 origin | HEAD 上的 tag | 备注 |
|------|------|-------------|---------------|------|
| cuda-foundations | `38ccdcd` docs: record freeze ctest… | ahead 2 | （空；`phase-2-e` 在 HEAD~2） | 测试跑在 `44ac954`；`38ccdcd` 只改正 skip 表述 |
| triton-fused-ops | `317347e` docs: check off GPU benchmark… | ahead 1 | （空；`phase-2-e` 在 HEAD~1） | pytest 与 ROADMAP 提交并行，未改测试代码 |
| cuflash-attn | `e0862b4` docs: check off completed ROADMAP… | ahead 1 | （空；`phase-2-e` 在 HEAD~1） | |
| tiny-llm | `15001c5` docs: align ROADMAP and README… | ahead 1 | （空；`phase-2-e` 在 HEAD~1） | 测试带 `TLLM_GGUF_TEST_MODEL` |
| paged-infer | `fb9d670` docs: mark paged KV strategy 1… | ahead 1 | （空；`phase-2-e` 在 HEAD~1） | 默认 `cargo test`，**未**开 `tiny-llm` feature |
| aicl-lab | `42fad33` docs: sync Phase 2 E-batch… | ahead 0 | （无 tag） | landing 仓；无 `phase-2-e` / `phase-3-interview` |

GitHub 可见性（`gh api repos/aicl-lab/<name> --jq .full_name`）：六个仓库均返回 `aicl-lab/<name>`。

## 3. 测试结果

### 3.1 cuda-foundations

```bash
cmake --preset default && cmake --build --preset default -j$(nproc) && ctest --preset default
```

- 墙钟：约 33 s（含配置/构建）
- CTest：**0 failed / 209 collected**；`100% tests passed, 0 tests failed out of 209`
- **78 skipped**（CTest “did not run”）：`AdvancedTest.*`、`FusionTest.*`、`GemmTest.*`、`InferenceTest.*`、`MemoryPoolTest.*`、`StreamManagerTest.*`、`TensorTest.*`
- 131 项实际执行并通过（01 模块与部分 03 测试在列）
- 口径：CTest 把 skip 算进 “0 failed / 209”，**不能说 209 项都在 GPU 上跑过**。MASTER_PLAN 已记录 04 GPU skip 为既有环境现象；本次 skip 集合更大，原样记录。

### 3.2 triton-fused-ops

```bash
.venv/bin/python -m pytest -q
```

- 墙钟：54.8 s
- **116 passed, 1 skipped**
- skip：`tests/test_torch_library.py::test_torch_compile_smoke`（文档约定：`torch.compile` 失败则 skip，不伪造通过）
- 收集 117 items（含 `test_sgemm.py` 24 项）

### 3.3 cuflash-attn

```bash
cmake --preset release && cmake --build --preset release -j$(nproc) && ctest --preset release --output-on-failure
```

- 墙钟：25.3 s
- **100% tests passed, 0 tests failed out of 71**
- 1 skipped：`cuflash_attn_pytorch_comparison`（did not run）
- 70 项执行通过

### 3.4 tiny-llm

```bash
cmake --build build -j$(nproc)
TLLM_GGUF_TEST_MODEL=/home/shane/github/aicl/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  ./build/tiny_llm_tests
```

- 墙钟：240.8 s
- **175 tests from 32 suites；174 passed；1 skipped**
- skip：`SecondModelTest.LoadsAndGeneratesWithDistinctGQA`（需要 `TLLM_GGUF_TEST_MODEL_2`，本机未提供第二份 GGUF）
- 门控真实模型路径已跑：GGUF 加载、tokenizer 差分、`CudaGraphsGenerateMatchesNonGraph`、W8A16、GQA kernel 等
- 与 README「第二模型待用户提供 GGUF」一致

### 3.5 paged-infer

```bash
cargo fmt --all -- --check && cargo clippy --all-targets -- -D warnings && cargo test
```

- 墙钟：11.5 s
- fmt / clippy：通过（`-D warnings`）
- `cargo test`（**未** `--features tiny-llm`）：

| 目标 | 结果 |
|------|------|
| lib unit | 144 passed |
| bin unit | 0（无测试） |
| concurrency_stress | 4 passed |
| integration_tests | 15 passed |
| server_integration | 37 passed |
| tiny_llm_backend | 0（`cfg(feature = "tiny-llm")`，未编译进本次运行） |
| tiny_llm_text_e2e | 0（同上） |
| tokenizer_real_diff | 1 passed |
| doctests | 17 passed |
| **合计** | **218 passed，0 failed** |

真实 tiny-llm 3 并发 e2e **本次未重跑**（T1 命令是无 feature 的 `cargo test`）。证据仍指向 `paged-infer@9c3700b` / `9c974d3` 与 `tests/tiny_llm_text_e2e.rs`。

## 4. 结论

- 五个开发仓：本次命令 **0 failed**。skip 均有对应门控（第二模型、torch.compile smoke、cuflash pytorch 对比、cuda-foundations 02/04 GPU 二进制、paged-infer tiny-llm feature）。
- 五仓相对 origin **ahead 1 或 2**（本轮 docs）；**尚未 push**。
- meta 仓 `aicl-lab` 无 phase tag。
- 下一步：T2 证据矩阵、T3 数字卡；本文件不发明性能数字。

## 5. K 阶段后补记（2026-08-18）
- 本审计撰写时的 ahead/无 tag 状态已由 PLAN_v3 阶段 K 清零；
- 当前 tag 链：phase-2-e → phase-3-docs（五仓）→ phase-3-interview（meta=9e0b4f7）。
