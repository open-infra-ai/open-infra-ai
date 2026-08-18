# PHASE 2 最终批次（Batch 5：E0 → E4，作品集收尾）

> **生成时间**：2026-08-18（D0–D5 完成后）
> **执行方**：DeepSeek Flash / 其他低成本模型
> **上游文档**：`PHASE2_PLAN.md` 第 8 节（E 阶段）；本批是其可执行细化。
> **状态**：D 阶段 ✅ 完成。tiny-llm HEAD=1c0abdd、paged-infer d32e435、cuda-foundations f91db3d，三仓 ahead 0，`phase-2-d` tag 已推送。
>
> 本批目标：① 补齐 D4 缺的 llama.cpp fixture（诚实记录分歧，不伪造）；② Triton SGEMM + torch.library；③ cuflash 一轮真实修复 + 一次有数字的优化；④ 组织 landing 与 release 校验。完成后五仓作品集进入"面试就绪"冻结态。
>
> 环境事实（已核实，不要重复验证）：
> - `gh` CLI 已登录且有 `repo`/`workflow` scope；
> - llama-cli：`/tmp/llama.cpp/build-cuda/bin/llama-cli`（885c5bb，CUDA 版）；
> - triton-fused-ops venv：torch 2.13.0+cu130 / triton 3.7.1，`torch.library.custom_op` 与 `torch.library.triton_op` 均可用；
> - 模型 `$MODEL=/home/shane/github/aicl/models/qwen2.5-0.5b-instruct-q4_k_m.gguf`；tokenizer `$TOK=/home/shane/github/aicl/models/tokenizer.json`。

---

## 0. 执行协议

1. 一次一个任务，验收全绿后 commit。
2. 性能数字只写本机实测（附 commit、GPU、命令）；拿不到参考的 fixture 只做可证明的弱断言，**禁止伪造完整 token 序列**。
3. 只改任务列出的文件/范围；每个任务一个 commit。
4. 本批不做：paged KV kernel 优化、CUDA Graphs 接入 paged、chunked prefill、新仓库级别的重写。

---

## 任务 E0：给 D4 请求 2 补上 llama.cpp 真实 fixture（paged-infer）

**背景（人工已核实，直接采用结论）**：
- llama-cli（同模型、greedy、`-st`）对 `<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n` 的参考输出 token 序列为：
  `[17, 10, 17, 16819, 220, 19, 13, 151645]`（"2+2 equals 4."）
- 当前 tiny-llm 策略 1 输出 `[17, 10, 17, 374, 220, 19, 13, 151645]`（"2+2 is 4."）。
- **第 4 个 token 不一致不是 bug**：`16819`("equals") 与 `374`("is") 是 W8A16（tiny-llm）与 Q4_K_M（llama.cpp）量化精度在 argmax 边界翻转，与 tiny-llm README 已记录的"同 prompt 前 N token 一致、后续因量化方案分歧"结论一致。
- 因此 D4 请求 2 **不能**写成全序列相等断言；改成"前缀一致 + EOS 终止 + 分歧注释"的诚实 fixture。

**改动文件**：`tests/tiny_llm_text_e2e.rs`（`qwen2_three_concurrent_paged_requests_match_llama_cpp`）。

**步骤**：
1. 在请求 2 的断言区替换为：
   ```rust
   // llama.cpp 参考： [17, 10, 17, 16819, 220, 19, 13, 151645] ("2+2 equals 4.")
   // tiny-llm 当前：  [17, 10, 17, 374, 220, 19, 13, 151645] ("2+2 is 4.")
   // 第 4 个 token 是 W8A16 vs Q4_K_M 的 argmax 边界翻转（量化分歧），
   // 因此只断言公共前缀 + EOS 终止，不伪装全序列一致。
   let out2 = &completed[1].output_tokens;
   assert!(completed[1].success);
   assert_eq!(&out2[..3], &[17, 10, 17], "公共前缀应与 llama.cpp 一致");
   assert_eq!(out2.last(), Some(&151645), "应以 EOS 终止");
   assert!(!completed[1].output_text.is_empty());
   ```
2. 保留请求 1 的全序列严格相等断言（24+EOS，已与 llama.cpp 对齐）与请求 3 的弱断言。
3. 更新测试顶部注释：记录请求 2 的两个序列与分歧原因。

**验收**：
```bash
cd /home/shane/github/aicl/paged-infer
cargo fmt --all -- --check && cargo clippy --all-targets -- -D warnings && cargo test
TINY_LLM_DIR=/home/shane/github/aicl/tiny-llm TINY_LLM_MODEL=$MODEL PINF_TOKENIZER_JSON=$TOK \
  cargo test --features tiny-llm --test tiny_llm_text_e2e -- --nocapture
```

**提交**：`test(e2e): honest llama.cpp divergence fixture for 2+2 prompt`

---

## 任务 E1：triton-fused-ops 补 Triton SGEMM 与 torch.library 注册

### E1a：Triton SGEMM kernel（一个 commit）

**改动文件**：新增 `triton_ops/kernels/sgemm.py`；`triton_ops/kernels/__init__.py`（导出）；新增 `tests/test_sgemm.py`。

**实现规格**：
1. `@triton.jit def sgemm_kernel(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr)`：
   - `pid_m = tl.program_id(0); pid_n = tl.program_id(1)`；
   - 用 `tl.arange` 构造 `offs_m/offs_n/offs_k`，所有 `tl.load` 带 mask（`offs_m < M` 等）；
   - 主循环 `for k in range(0, tl.cdiv(K, BLOCK_K))`，`acc = tl.dot(a, b, acc)`；
   - 输出 `tl.store(C, acc.to(C.dtype.element_ty), mask=...)`。
   - 默认 tile：`BLOCK_M=64, BLOCK_N=64, BLOCK_K=32`；`num_warps=4`。
2. Python wrapper `sgemm(a: torch.Tensor, b: torch.Tensor, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32) -> torch.Tensor`：
   - 输入为 row-major `[M,K]` 与 `[K,N]`，支持 `float32/float16/bfloat16`，`M,N,K ≥ 1`；
   - CUDA 张量 + 连续张量校验，错误抛仓库已有 `UnsupportedDtypeError`/`ShapeMismatchError` 风格异常；
   - `num_warps=4, num_stages=3`（与仓库 kernel 风格一致）。
3. 测试 `tests/test_sgemm.py`：
   - 与 `torch.mm` 差分：`(M,N,K) ∈ {(64,64,64), (1,128,896), (128,1,896), (17,33,65)}`，fp16/fp32 各一，rtol=1e-2/atol=1e-2（FP16）；
   - 边界：M=1、N=1、K=1、非 2 幂；
   - 失败路径：CPU tensor、错误 dtype、非连续张量。

**验收**：
```bash
cd /home/shane/github/aicl/triton-fused-ops
.venv/bin/python -m pytest -q tests/test_sgemm.py
.venv/bin/python -m pytest -q   # 88 + 新增 全绿
```

**提交**：`feat(triton): SGEMM kernel with differential tests`

### E1b：torch.library 注册三个自定义算子（一个 commit）

**改动文件**：新增 `triton_ops/ops.py`；修改 `triton_ops/__init__.py`（注册导出）；新增 `tests/test_torch_library.py`；修改 `README.md`（新增"PyTorch 集成"小节）。

**实现规格**：
1. `ops.py` 使用 `torch.library.triton_op`（本机 torch 2.13 可用）注册三个算子：
   - `triton_ops::sgemm(a, b) -> Tensor`
   - `triton_ops::fused_rmsnorm_rope(x, weight, cos, sin, eps=1e-6) -> Tensor`
   - `triton_ops::fused_gated_mlp(x, gate_weight, up_weight, activation="silu") -> Tensor`
2. 每个注册：
   - 只允许 CUDA 张量（CPU 直接抛 `NotImplementedError`/`RuntimeError`，信息写明）；
   - `mutates_args=()`；
   - 内部只调用 `triton_ops.kernels.*` 的公开函数，不复制 kernel 逻辑；
   - 若 `torch.library.triton_op` 不可用则 fallback 到 `torch.library.custom_op + register_fake`（写一个 helper，两种路径都要能通过测试）。
3. `triton_ops/__init__.py`：`from triton_ops import ops as _ops  # noqa: F401`，确保 `import triton_ops` 后 `torch.ops.triton_ops.*` 可调用；把 `sgemm` 加入 `__all__`。
4. `tests/test_torch_library.py`：
   - `import triton_ops` 后调用 `torch.ops.triton_ops.sgemm(a,b)`，与 `torch.mm` 差分；
   - 另两个 op 与 `triton_ops.kernels.*` 直接调用逐元素一致（容差 1e-2）；
   - `torch.compile` smoke：对 sgemm 做一次 `torch.compile` 调用（若 compile 失败只记录 skip，不伪造通过）。
5. `README.md`：新增"torch.library 自定义算子"小节：用法、注册命名空间、与 vLLM/SGLang custom op 接入的对应关系。

**验收**：
```bash
.venv/bin/python -c "import torch, triton_ops; print(torch.ops.triton_ops.sgemm)"   # 应打印 schema
.venv/bin/python -m pytest -q tests/test_torch_library.py
.venv/bin/python -m pytest -q
```

**提交**：`feat(torch): register custom ops via torch.library`

---

## 任务 E2：cuflash-attn 一轮修复 + 一次有数字的优化

### E2a：修复 grid.y 65535 限制（P1 正确性，一个 commit）

**问题**：`BM_Forward*`/`BM_Forward_Causal*` 等 forward kernel 用 `grid.y = batch_size * num_heads`；当 `B*H > 65535` 时 launch 非法（历史审计 T4）。

**改动文件**：
- `src/forward/flash_attention_forward_typed.cu`
- `src/forward/flash_attention_forward_wmma.cu`
- `tests/`（新增回归测试）

**步骤**：
1. 把 grid 展平到 x 维：
   ```cuda
   const int num_q_blocks = (seq_len + BLOCK_M - 1) / BLOCK_M;
   const int total_blocks = num_q_blocks * batch_heads;
   dim3 grid(total_blocks);
   ```
2. kernel 内部把 `blockIdx.y` 替换为：
   ```cuda
   const int q_block = blockIdx.x % num_q_blocks;
   const int batch_head_idx = blockIdx.x / num_q_blocks;
   const int q_start = q_block * BLOCK_M;
   ```
   （num_q_blocks 需要作为 kernel 参数传入，或在 kernel 内用 `(seq_len + BLOCK_M - 1) / BLOCK_M` 重算——推荐后者，避免改签名。）
3. 两个 forward kernel 同步修改；backward 若同样使用 `grid.y=batch_heads` 且存在 65535 风险，一并用相同方法修（先 grep 确认，只修 forward 也可以，但要在 commit message 与文档写明范围）。
4. 回归测试：新增 `BM_Forward_GridYOverflowSmoke`（FP32 路径）：
   - `batch_size=512, num_heads=128, seq_len=1, head_dim=64`（`B*H=65536 > 65535`，Q/K/V/O/L 总显存约 33MB）；
   - 调用 `cuflash::flash_attention_forward`，断言 `SUCCESS` 且输出与 CPU/逐元素参考一致（容差 1e-3）。
   - WMMA/FP16 路径若 seq_len=1 不满足其 tile 约束，可只测 FP32；文档写明。
5. benchmark/ctest 全跑。

**验收**：
```bash
cd /home/shane/github/aicl/cuflash-attn
cmake --preset release && cmake --build --preset release
ctest --preset release --output-on-failure   # 69 + 新增测试
./build/release/cuflash_attn_bench --benchmark_filter='GridYOverflow|Forward'
```

**提交**：`fix(forward): flatten grid.y batch*heads for >65535 launches`

### E2b：causal 边界块跳过优化（一次有数字的优化，一个 commit）

**改动文件**：`src/forward/flash_attention_forward_typed.cu`、`src/forward/flash_attention_forward_wmma.cu`（若其 causal 分支适用）、`docs/performance/`、`tests/`。

**步骤**：
1. 在 forward kernel 的 KV 块循环前计算当前 Q 块最后可见位置：
   ```cuda
   const int q_last = min(q_start + BLOCK_M - 1, seq_len - 1);
   ```
2. KV 循环内，在加载 K/V tile 之前：
   ```cuda
   // causal：整块都在"未来"且后续块必然更远 → 直接结束循环
   if (causal && kv_start > q_last) break;
   ```
   只允许 `break`（因为块按 kv_start 递增），不允许 `continue` 跳过中间块。
3. 保持 mask 内 `kv <= q_row` 原逻辑不变（部分重叠块仍走 mask）。
4. 测试：
   - 现有 causal 差分测试全绿（PyTorch SDPA 对齐）；
   - 新增非对称形状测试（seq_len=257，非整 tile 边界）确保 `q_last` 计算正确。
5. 实测 benchmark（FP16，causal vs non-causal，N=256/512/1024/2048，head_dim=64）：
   ```bash
   ./build/release/cuflash_attn_bench --benchmark_filter='Forward_Causal'
   ./build/release/cuflash_attn_bench --benchmark_filter='Forward_FP16'
   ```
6. 把 before/after 表写入 `docs/performance/`（快照版本+日期+硬件+commit）；如果提升 <10%，保留改动（数值不回归）并在文档写明"增益低于噪声，主要价值是减少无效访存"。

**验收**：
```bash
ctest --preset release --output-on-failure
./build/release/cuflash_attn_bench --benchmark_filter='Forward'
# 期望：causal FP16 各长度有可测下降；所有测试全绿
```

**提交**：`perf(forward): skip fully-future KV blocks in causal path`

---

## 任务 E3：组织 landing README（可选但推荐，一个任务）

**背景**：根目录 `/home/shane/github/aicl/.git` 是空占位，不是仓库；MASTER_PLAN/PHASE2 计划目前不进版本控制。用一个 `aicl-lab` meta 仓库收口。

**执行步骤**：
1. 尝试创建仓库（有权限则执行，无权限则记录 BLOCKED 并跳过本任务）：
   ```bash
   gh repo create aicl-lab/aicl-lab --public --description "AICL-Lab AI Infra learning portfolio: 5-repo project map" --confirm
   ```
2. 在 `/tmp/aicl-lab-meta` 准备内容：
   - `README.md`：五仓地图（四层能力表）、阅读顺序、各仓定位与状态链接、Phase 2 完成证据摘要（TPOT 6.09ms、paged KV 3 并发、改名 cuda-foundations）；
   - `MASTER_PLAN.md`、`PHASE2_PLAN.md`、`PHASE2_NEXT.md`、`PHASE2_NEXT_C.md`、`PHASE2_NEXT_D.md`、`PHASE2_NEXT_E.md` 的副本；
   - `docs/organization-audit/` 归档副本。
3. `git init && git add ... && git commit`，`git remote add origin https://github.com/aicl-lab/aicl-lab.git`，`git push -u origin master`。
4. 回改五仓 README 顶部加一行 `> 📚 Portfolio map: https://github.com/aicl-lab/aicl-lab`（各仓一个 docs commit，本任务可合并成 5 个小提交）。

**验收**：
```bash
gh repo view aicl-lab/aicl-lab --json name,url
curl -sI https://github.com/aicl-lab/aicl-lab | head -1   # 200
```

**提交**：meta 仓一个初始 commit；五仓各 `docs: link portfolio landing repo`。

---

## 任务 E4：release tag 与跨仓 badge/link 终检

**改动**：无源码改动，主要是执行与修复发现的坏链。

**步骤**：
1. 五个仓库各打并推送 `phase-2-e` tag：
   ```bash
   for d in cuda-foundations triton-fused-ops cuflash-attn tiny-llm paged-infer; do
     (cd /home/shane/github/aicl/$d && git tag phase-2-e && git push origin phase-2-e)
   done
   ```
2. 终检脚本（输出必须全部干净）：
   ```bash
   # 旧名 0 命中（audit 归档除外）
   grep -rn "cuda-kernel-academy" --exclude-dir=.git --exclude-dir=build --exclude-dir=target \
     --exclude-dir=.venv --exclude-dir=node_modules /home/shane/github/aicl | grep -v organization-audit || true
   # 五个仓库在 GitHub 上可见
   for r in cuda-foundations triton-fused-ops cuflash-attn tiny-llm paged-infer aicl-lab; do
     gh api repos/aicl-lab/$r --jq '.full_name' 2>/dev/null || echo "MISSING $r"
   done
   # 每个 README 的 badge 链接 curl -I 返回 200/301/302
   for r in cuda-foundations triton-fused-ops cuflash-attn tiny-llm paged-infer; do
     echo "== $r =="; curl -sI https://github.com/aicl-lab/$r | head -1
   done
   ```
3. 若发现死链/旧名，直接修复对应 README 并 push（`docs: fix stale links`）。
4. 更新根 `MASTER_PLAN.md` / `PHASE2_PLAN.md` 最终状态（E 阶段全勾选、Phase 2 DoD 勾选）。

**验收**：五仓 `phase-2-e` tag 远端可见；终检脚本无 `MISSING`、无 404；旧名 grep 0 命中。

---

## 全部完成后的最终汇报格式

1. 五仓 + meta 仓 `git status -sb` 一行；
2. 本批 commit hash 列表；
3. E1 的 `torch.ops.triton_ops.*` schema 输出；
4. E2a grid overflow 测试结果、E2b causal before/after benchmark 表；
5. E3 meta 仓 URL（或 BLOCKED 原因）；
6. E4 终检输出；
7. 最终性能/正确性看板：
   - tiny-llm：TPOT 6.09ms、170 tests（166 passed/8 skipped? 以实际为准）、paged KV 策略 1 差分通过、3 并发 llama.cpp 对齐；
   - paged-infer：cargo test 全绿；
   - cuflash：ctest 全绿 + causal 优化数字；
   - triton：pytest 全绿 + torch.library 可用；
   - cuda-foundations：ctest 全绿 + 改名完成。

---

## 面试冻结声明（E 阶段完成后写入每个 README 的状态表）

完成 E 阶段后，五仓进入"**面试就绪冻结**"：不再扩新功能；只修正确性 bug 与文档漂移；新想法记入各仓 ROADMAP 的"不做什么"清单。这是作品集交付纪律，不是降低标准。
