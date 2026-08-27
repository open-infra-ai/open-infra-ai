# PHASE 2 开发计划：重命名 + 边界收口 + 深度重构 + 旗舰补完

> **版本**：2026-08-18
> **上游**：`MASTER_PLAN.md`（Phase 1 已由低成本模型执行完毕，报告见其第 7 节勾选状态）
> **目标读者**：低成本 AI 编程模型（一次性只给一个任务执行）
> **本阶段核心问题**：① 项目命名统一；② 五仓边界与代码所有权收口；③ 补完两个旗舰缺口（tiny-llm decode 性能、paged KV 端到端）；④ 作品集在 GitHub 上可见且可复现。
> **执行状态**：A0–B5 ✅（五仓改名/推送完成）；C0–C3 ✅（TPOT 6.09ms，比值 1.65×）；D0–D5 ✅（分页 KV 策略 1 + 3 并发 llama.cpp 对齐，`phase-2-d` tag）；**E0–E4 ✅**（llama.cpp 诚实分歧 fixture、Triton SGEMM + torch.library、cuflash grid.y 修复 + causal 优化、`aicl-lab` landing 仓、五仓 `phase-2-e` tag 推送完成）。
> **当前批次任务**：Phase 2 已冻结，后续见 [`PLAN_v3.md`](PLAN_v3.md)（阶段 K 收尾 / 阶段 I 面试执行 / 阶段 D 可选深度增量）。

---

## 0. 执行协议（每个任务都要遵守）

1. **一次只做一项任务**：完成任务末尾的验收命令，全绿后再做下一个。
2. **先跑验收再提交**：`build/test` 不通过 = 任务未完成，不允许提交。
3. **不做计划外重构**：发现新问题写入 NOTE，不扩大 diff。
4. **禁区**（除非任务明确要求）：
   - tiny-llm `src/tokenizer.cpp`、`scripts/gen_unicode_tables.py`：已与 HF 逐 id 对齐。
   - tiny-llm `src/gguf_parser.cpp`、`src/quantization.cpp` 的反量化格式：已与 Python gguf 参考对齐。
   - tiny-llm KV Cache 的连续布局：`ffi.h` C ABI 契约依赖它（D 阶段按新 ABI 显式升级除外）。
   - paged-infer `src/scheduler.rs` 的 P0 行为：已被属性测试锁定。
5. **每个任务一个 commit**，commit message 用 `<type>(<scope>): <summary>` 格式。
6. **性能数字只写实测**：必须有 GPU 型号、CUDA/driver 版本、commit hash、复现命令。
7. 本机环境：RTX 3060 Laptop 6GB，CUDA 12.x，驱动 591.44，`nsys`/`ncu` 可用。

---

## 1. 结构决策：命名方案与"4 还是 5"

### 1.1 最终结构：四层能力、五个物理仓库

```
┌─ 层 1  Kernel 基础层（两个仓库，两种范式，刻意对比）
│     cuda-foundations        CUDA C++ 基础与测量（改名自 cuda-foundations）
│     triton-fused-ops        Triton 算子与 PyTorch 集成
├─ 层 2  Kernel 深度层
│     cuflash            FlashAttention 唯一 owner
├─ 层 3  Runtime 运行时层
│     tiny-llm                单 GPU 推理运行时（量化 decode 性能为锚点）
└─ 层 4  Serving 控制面层
      paged-infer             分页 KV + continuous batching（后端解耦）
```

### 1.2 为什么不合并成 4 个物理仓库

合并 CUDA 与 Triton 到同一个 repo 的代价与收益不对等：

| 维度 | 代价 |
|---|---|
| 构建系统 | CMake+CUDA 与 Python/pip 两套 CI、打包、依赖锁混在一个仓 |
| 命名空间 | 两个语言生态的包名/namespace 相互干扰 |
| 面试展示 | "同题异构"是两个 repo 可并排打开的加分项，合仓后反而难讲 |
| 既有投入 | 五个仓已完成 D2 边界文档、D3/D4 降级标注、全部测试全绿；合仓=推倒重做 |
| GitHub 生态 | 五个小仓的 README/文档站 URL 已经互相链接，合并产生 404 链 |

**决策：保持 5 个物理仓库。** 只有一种情况再合并：你拿到目标岗位的明确反馈说"repo 太多不聚焦"，且离面试还有 4 周以上。

### 1.3 命名方案总表

| 项 | 旧值 | 新值 |
|---|---|---|
| GitHub 仓库 | `AICL-Lab/cuda-foundations` | `AICL-Lab/cuda-foundations` |
| 本地目录 | `cuda-foundations/` | `cuda-foundations/` |
| C++ namespace | `cuda_academy` | `cuda_foundations` |
| 公共头目录 | `common/include/cuda_academy/` | `common/include/cuda_foundations/` |
| 聚合头文件 | `cuda_academy/cuda_academy.hpp` | `cuda_foundations/cuda_foundations.hpp` |
| CMake 项目 | `CUDAKernelAcademy` | `CUDAFoundations` |
| CMake common 项目 | `cuda-academy-common` | `cuda-foundations-common` |
| CMake 库 target | `cuda_academy_common` / `CudaAcademy::common` | `cuda_foundations_common` / `CUDAFoundations::common` |
| CMake export | `CudaAcademyTargets` | `CUDAFoundationsTargets` |
| 文档站 base | `/cuda-foundations/` | `/cuda-foundations/` |
| npm 包名 | `cuda-foundations-docs`（或现值） | `cuda-foundations-docs` |

**不变**：org 名 `AICL-Lab`；其余四仓名 `triton-fused-ops`、`cuflash`、`tiny-llm`、`paged-infer` 全部不变。

---

## 2. 边界划分与代码所有权（执行重构的依据）

### 2.1 每个仓库拥有什么（代码级）

| 仓库 | 拥有（可改） | 不拥有（禁止添加/复制） |
|---|---|---|
| cuda-foundations | `01-sgemm-tutorial`、`02-tensorcraft-core`、`03-hpc-advanced`、`04-inference-engine`（教学预览）、`common/`、`docs/`、GEMM/算子的**教学实现** | 真实 GGUF/量化 runtime；生产 FlashAttention；调度器 |
| triton-fused-ops | `triton_ops/kernels/`、`reference/`、`autotuner/`、`benchmark/`、`tests/`；**Triton FA 只作为 cuflash 的参考实现** | CUDA C++ FA；完整模型加载；Serving |
| cuflash | `src/forward/`、`src/backward/`、`src/kernels/`、FlashDecoding、FA 前后向算法与优化 | GEMM 教程；Triton 教学；推理运行时；调度 |
| tiny-llm | `kernels/`、`src/`（GGUF/tokenizer/量化/transformer/KV/采样/bench/ffi）、`include/`；单 GPU 运行时与 C ABI | 调度/批处理/paged 控制面（paged-infer）；FA 教学 |
| paged-infer | `src/`（scheduler/kv_cache/server/executor）、`tests/`、`benches/`；控制面与后端 trait | 任何 CUDA kernel；模型加载/tokenizer 实现 |

### 2.2 跨仓规则（面试时会被追问，文档也要写清）

1. **一个算法一个权威 owner**：FlashAttention → cuflash；量化反量化 GEMM → tiny-llm；Paged KV 控制面 → paged-infer；GEMM 教学阶梯 → cuda-foundations。
2. **同题异构允许**：同一 GEMM 分别在 cuda-foundations（CUDA C++）与 triton-fused-ops（Triton）出现，但 README 必须互相指向，并说明对比结论。
3. **跨仓只通过窄 ABI**：paged-infer ↔ tiny-llm 只允许走 `ffi.h` / `tiny_llm_ffi.rs` 的 C ABI，禁止 include 对方头文件。
4. **共享的是契约和 fixture，不是源码**：layout/RoPE/KV 语义以 `docs/organization-audit/.../03-cross-repo-contracts.md` 为准；测试向量用 JSON/NPZ 文件，不复制实现代码。
5. **教学仓不得被运行时依赖**：tiny-llm 禁止 include cuda-foundations 的任何头文件（教学实现的 API 稳定性目标不同）。

### 2.3 本阶段要做/不做的重构

| 重构 | 判定 | 原因 |
|---|---|---|
| R1 tiny-llm `execution_common` 收口（修 WIP） | ✅ 做（P0） | 消除 FFI 与 InferenceEngine 最终层逻辑漂移，且当前工作区已断构建 |
| R2 cuda-foundations 机械改名 | ✅ 做（P1） | 纯文本/路径替换，无行为变化，风险低 |
| R3 tiny-llm FFI v2 分页 KV | ✅ 做（P1，旗舰） | 完成 D1 的唯一正确方式，mini-vLLM 叙事闭环 |
| R4 paged-infer executor 策略 1 | ✅ 做（P1） | 配合 R3；策略 2 保留为 fallback feature |
| R5 triton-fused-ops 脏文件 commit/revert | ✅ 做（P0） | 工作区有未提交改动，先定性再处理 |
| R6 cuflash 脏文件 commit/revert | ✅ 做（P0） | 同上 |
| R7 把 triton-fused-ops 物理合入 cuda-foundations | ❌ 不做 | 见 1.2 |
| R8 把 04-inference-engine 代码搬进 tiny-llm | ❌ 不做 | 04 是教学预览，搬迁引入无用耦合；保持文档边界即可 |
| R9 paged-infer 改 C++ | ❌ 不做 | Rust + C ABI 是加分项，不是负担 |
| R10 tiny-llm 重写 tokenizer/GGUF parser | ❌ 不做 | 已与权威实现对齐，重写只带来回归风险 |

---

## 3. 任务总览与依赖

```
阶段 A（P0）：仓库卫生 + 恢复绿构建 —— 先让每个仓库在 GitHub 上干净可见
  A0 修复 tiny-llm 4.3 WIP（完成 execution_common 重构）   ← 阻塞一切
  A1 处置 triton-fused-ops 未提交改动
  A2 处置 cuflash 未提交改动
  A3 推送全部 unpushed commits + 打 phase-1 基线 tag
  出口门槛：五个仓库 git status clean，远端领先，CI/本地测试全绿

阶段 B（P1）：整体重命名 cuda-foundations
  B0（人工）GitHub 重命名仓库（唯一需要你手动做的步骤）
  B1 本地目录与 remote 更新
  B2 仓内机械改名（namespace/头文件/CMake/宏/注释标识）
  B3 仓内文档与文档站改名（VitePress base、链接、package.json）
  B4 跨仓链接更新（其余 4 仓 + 根文档）
  B5 全量构建/测试/Pages 链接检查
  出口门槛：`cuda-foundations` 名称在任何仓库源码中 0 命中

阶段 C（P1）：tiny-llm decode 性能攻坚（量化 decode 深度锚点的真正证据）
  C0 profiling 基线（nsys + ncu，产出 top-5 kernel 表）
  C1 按证据优化第 1 瓶颈
  C2 按证据优化第 2 瓶颈
  C3 复测 + 文档化（目标与口径见任务明细）
  出口门槛：TPOT ≤ 12ms 且 greedy 输出与优化前逐 token 一致（先 12ms，再冲刺 8ms）

阶段 D（P1）：分页 KV 端到端（补完 MASTER_PLAN D1，旗舰叙事）
  D1 tiny-llm ABI v2：TinyLlmConfig 增加 max_num_blocks + 分页 KV 池
  D2 tiny-llm scatter/gather 块式 KV kernel + ffi 步进接入 block_tables
  D3 paged-infer TinyLlmExecutor 切换策略 1（保留策略 2 fallback）
  D4 跨仓端到端：3 并发真实模型 + llama.cpp 对齐 + 资源守恒
  D5 文档与边界收口
  出口门槛：paged-infer + tiny-llm 走真实 block_tables 跑通，输出与 llama.cpp 逐 token 一致

阶段 E（P2，有余力）：作品集收尾
  E1 triton-fused-ops：Triton SGEMM + torch.library 注册（补 MASTER_PLAN C3）
  E2 cuflash：一轮有数字的优化迭代（补 MASTER_PLAN C2）
  E3 组织级 landing README（新建 meta 仓，可选）
  E4 各仓 release tag + README badge 校验
```

---

## 4. 阶段 A 任务明细

### 任务 A0：完成 tiny-llm 4.3 `execution_common` 重构（P0，阻塞项）

**当前问题（已核实）**：
- `include/tiny_llm/execution_common.h` 与 `src/execution_common.cpp` 未提交；
- `computeLogitsFromHidden(const half *hidden, ...)` 对 `kernels::rmsnorm` 传入 const 指针导致编译错误（`rmsnorm` 第 3 参数是 `half*`，就地写）；
- 更严重：WIP 把 `InferenceEngine::computeLogits()` 的语义从"只做 lm_head"改成了"rmsnorm + lm_head"，与旧函数名语义不符，属于隐性行为变更。

**正确收口方案（必须按此实现，不要抄 WIP 原样）**：

1. `include/tiny_llm/execution_common.h`：
   - 函数签名改为 `void finalNormAndComputeLogits(half *hidden, const ModelWeights &, const ModelConfig &, half *logits, cudaStream_t = 0);`
   - 文档注释写明：`hidden` 为单 token 隐藏态，**就地做 final RMSNorm**，再做 lm_head；**不负责采样**；只支持 `num_tokens == 1`。
2. `src/execution_common.cpp`：
   - 函数名同步改为 `finalNormAndComputeLogits`，`hidden` 参数为非 const `half*`。
   - 内部保持现有逻辑：`if (final_norm_weight) rmsnorm(...)`；然后 `lm_head_fp16` 优先、`lm_head.isValid()` W8A16 后备。
3. `src/inference_engine.cpp`：
   - `computeLogits(hidden_states, num_tokens, logits)` **恢复原语义**：只做 lm_head 投影，不调用 rmsnorm，不使用新 helper（把 WIP 里 `(void)num_tokens` 的 hack 删掉）。
   - `runDecodeDevicePath` 与 `sampleFromHidden` 两处改为调用 `finalNormAndComputeLogits(token_state, weights_, config_, logits_, stream_)`。
4. `src/ffi.cpp`：
   - `sample_from_hidden` 改为调用 `tiny_llm::finalNormAndComputeLogits(...)`。
5. `CMakeLists.txt`：
   - 当前 `file(GLOB_RECURSE SOURCES "src/*.cpp" "kernels/*.cu")` 不会在增量构建时发现新文件。**显式追加**：
     ```cmake
     list(APPEND SOURCES src/execution_common.cpp)
     ```
     同时保留原有 GLOB（本次不重构 GLOB 问题，记录 NOTE）。
6. 给 helper 补一个最小测试（新增或加入 `tests/test_ffi.cpp`）：
   - 构造 fake `ModelWeights`（final_norm 权重全 1、`lm_head_fp16` 小矩阵）；
   - 与"先 rmsnorm 再 fp16_matmul"的两步调用逐元素比较，容差 `1e-2`；
   - 覆盖 `final_norm_weight == nullptr` 的分支。

**验收命令**：
```bash
cd tiny-llm
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build build -j$(nproc)          # 必须 0 error
./build/tiny_llm_tests                   # 期望 165 tests / 157 passed / 8 skipped（gated）
git status --short                      # 只有本任务列出的文件被修改
```

**提交**：`refactor(runtime): extract finalNormAndComputeLogits shared by engine and FFI`

**注意**：不要顺手改 `tokenizer.cpp`、`gguf_parser.cpp`、`kv_cache` 布局；如果 build 还报其它错误，先记录再报告，不要自行大改。

---

### 任务 A1：处置 triton-fused-ops 未提交改动（P0）

**现状**：8 个已修改文件（`examples/rmsnorm_rope_example.py`、5 个 test 文件、`kernels/gated_mlp.py`、`reference/rmsnorm_rope.py`、`validation.py`），未提交。

**执行规则**：
1. 先跑 `python -m pytest -q`。
2. 逐文件 `git diff --stat` 并阅读 diff，判断属于哪类：
   - **B4 收尾/兼容修复**（numpy 兼容、容差、Triton 3.x 适配、测试参数）→ 归属 `fix(triton): B4 benchmark compatibility follow-ups`；
   - **风格/临时改动**且无测试意义 → `git checkout -- <file>` 还原。
3. 判定标准：改动必须让测试从绿到绿，或修复明确 bug；任何"看起来更对但无测试支撑"的改动一律还原。
4. `git status` 干净后提交剩余文件。

**验收**：
```bash
python -m pytest -q    # 88/88 全绿
git status --short     # 干净
```

**提交**：`fix(triton): commit verified B4 compatibility follow-ups`

---

### 任务 A2：处置 cuflash 未提交改动（P0）

**现状**：11 个 docs 修改 + 1 个未跟踪 `PLAN.md`。

**执行规则**：
1. 读 `git diff`，若全部是文档措辞/链接修正且与当前代码事实一致 → 提交 `docs(cuflash): sync documentation with v0.5.0 code facts`。
2. `PLAN.md`：与仓库内已跟踪的 `PLAN.md`？不存在则检查内容；若它是陈旧规划或与 `ROADMAP.md` 重复 → **不提交，删除**；若包含 D1 之后仍有效的任务 → 移入 `docs/` 并提交。
3. 禁止把未经 `ctest` 验证的文档改动直接提交；先跑测试。

**验收**：
```bash
cmake --preset release && cmake --build --preset release
ctest --preset release --output-on-failure   # 69/69
git status --short                            # 干净
```

**提交**：按上述规则。

---

### 任务 A3：推送全部 unpushed commits + 打基线 tag（P0）

**现状**：academy ahead 2、triton ahead 3、cuflash ahead 4、tiny-llm ahead 19、paged-infer ahead 15。

**执行**：
1. 每个仓库依次：
   ```bash
   git log origin/master..HEAD --oneline   # 人工核对：确认都是 Phase 1 报告里的任务提交
   git push origin master
   git tag phase-1-complete && git push origin phase-1-complete
   ```
2. tiny-llm 必须在 A0 完成并提交后再推送（否则会把断构建的 WIP 推上去）。
3. 若某个仓库 push 被拒绝（non-fast-forward），**不要 force push**，报告情况。

**验收**：
```bash
for d in cuda-foundations triton-fused-ops cuflash tiny-llm paged-infer; do
  (cd "$d" && echo "== $d ==" && git status -sb && git log origin/master..HEAD --oneline | wc -l)
done
# 期望：每个仓库 ahead 0，status 干净
```

---

## 5. 阶段 B 任务明细：重命名 cuda-foundations

### 任务 B0（人工）：GitHub 重命名仓库

**执行人：你（用户），不是模型。**
1. 打开 `https://github.com/open-infra-ai/cuda-foundations/settings`
2. Repository name 改为 `cuda-foundations`
3. GitHub 会自动重定向旧 URL；确认 `https://github.com/open-infra-ai/cuda-foundations` 可访问。
4. 完成后把"重命名已完成"告诉模型，再继续 B1。

> 若你有 `gh` CLI 且有权限，等效命令：`gh repo rename AICL-Lab/cuda-foundations cuda-foundations`

---

### 任务 B1：本地目录与 remote 更新

**文件**：本地目录、`.git/config`

1. `mv cuda-foundations cuda-foundations`
2. `cd cuda-foundations && git remote set-url origin git@github.com:AICL-Lab/cuda-foundations.git`（HTTPS 则用 `https://github.com/open-infra-ai/cuda-foundations.git`）
3. `git fetch && git status -sb` 确认 tracking 正常。

**验收**：
```bash
git -C cuda-foundations remote -v
# 期望：origin 指向新 URL
```

---

### 任务 B2：仓内机械改名（namespace / 头文件 / CMake）

**范围（已核实，16 个文件 + 目录）**：

1. **目录与文件名**：
   - `common/include/cuda_academy/` → `common/include/cuda_foundations/`
   - `common/include/cuda_foundations/cuda_academy.hpp` → `cuda_foundations.hpp`
2. **文本替换顺序（防止半替换状态）**：
   ```bash
   # 在 cuda-foundations 根目录
   grep -rl "cuda_academy" --exclude-dir=.git --exclude-dir=build --exclude-dir=node_modules . | while read f; do
     sed -i 's/cuda_academy/cuda_foundations/g; s/CudaAcademy/CUDAFoundations/g; s/CUDAKernelAcademy/CUDAFoundations/g' "$f"
   done
   grep -rl "cuda-foundations\|CUDAKernelAcademy\|cuda-academy" --exclude-dir=.git --exclude-dir=build --exclude-dir=node_modules . | while read f; do
     sed -i 's/cuda-foundations/cuda-foundations/g; s/CUDAKernelAcademy/CUDAFoundations/g; s/cuda-academy-common/cuda-foundations-common/g' "$f"
   done
   ```
3. **人工核对的特殊项**（sed 后逐个确认）：
   - `common/CMakeLists.txt`：`add_library(cuda_foundations_common INTERFACE)` + `add_library(CUDAFoundations::common ALIAS ...)` + `EXPORT CUDAFoundationsTargets`；
   - 根 `CMakeLists.txt`：`project(CUDAFoundations ...)`、`HOMEPAGE_URL "https://github.com/open-infra-ai/cuda-foundations"`、顶部注释；
   - `02-tensorcraft-core/include/tensorcraft/core/cuda_check.hpp`：`using CudaError = cuda_foundations::core::CudaError;` 及宏内命名空间；
   - `common/include/cuda_foundations/core/*.hpp`：namespace 开闭注释；
   - 宏 `CA_CUDA_CHECK` 保留原宏名（宏名是公共 API，本次只改命名空间限定，不改宏名）。
4. **不要改**：git history、`VERSION` 文件内容（版本号保持）、LICENSE 年份。

**验收**：
```bash
grep -rn "cuda_academy\|CudaAcademy\|CUDAKernelAcademy\|cuda-foundations" \
  --exclude-dir=.git --exclude-dir=build --exclude-dir=node_modules . | wc -l
# 期望：0
cmake --preset default && cmake --build --preset default
ctest --preset default
# 期望：209/209 全绿（本机 04 GPU 测试可能 skip，保持 skip 数不变）
```

**提交**：`refactor: rename namespace/CMake to cuda_foundations (mechanical, no behavior change)`

---

### 任务 B3：文档站与仓库内部链接改名

**文件**：`docs/.vitepress/config.ts`、`docs/package.json`、`docs/tests/*.test.mjs`、`docs/public/404.html`、README 内的 URL、CI 中的 Pages 路径。

1. `docs/.vitepress/config.ts`：`base: '/cuda-foundations/'`、站点标题/仓库 URL 全部替换。
2. `docs/package.json`：name 与 homepage 替换。
3. `docs/tests/site-canonical-links.test.mjs`：断言里的旧 URL 全部替换为新 URL。
4. `.github/workflows/pages.yml`：action 配置中的 base 或 path 引用（若有）。
5. `README.md` / `README.zh-CN.md` / `LEARNING_PATH.md`：旧仓库 URL 替换。

**验收**：
```bash
grep -rn "cuda-foundations" docs README.md README.zh-CN.md LEARNING_PATH.md .github || true
# 期望：0 命中
cd docs && npm run build 2>/dev/null || npm run docs:build   # 按仓库 package.json 实际脚本
# 期望：build 成功，无死链报错
```

**提交**：`docs: rename site and links to cuda-foundations`

---

### 任务 B4：跨仓链接更新

**范围（已核实）**：`paged-infer/README.md`、`tiny-llm/README.md`、`cuflash/README.md`、`triton-fused-ops/README.md`、`triton-fused-ops/triton_fused_ops.egg-info/PKG-INFO`、根 `MASTER_PLAN.md`、`docs/organization-audit/**`。

**规则**：
1. 把所有 `github.com/open-infra-ai/cuda-foundations` 替换为 `github.com/open-infra-ai/cuda-foundations`。
2. `docs/organization-audit/` 是历史审计记录：正文替换 URL，但审计文档标题/文件名（`repos/cuda-foundations.md`）**保留原名**，并在其文件头加一行"（本审计归档于仓库改名 cuda-foundations 之前，下同）"。
3. `triton_fused_ops.egg-info/PKG-INFO` 是构建产物，若被 git 跟踪则同步替换；若未跟踪，忽略。
4. 每个被改仓库独立提交 `docs: point links to cuda-foundations`，并跑各自快速验证（triton pytest 不必须全跑，grep 即可；tiny-llm/paged-infer README 改动不跑测试）。

**验收**：
```bash
grep -rn "cuda-foundations" --exclude-dir=.git --exclude-dir=build --exclude-dir=target \
  --exclude-dir=.venv --exclude-dir=node_modules /home/shane/github/aicl | grep -v organization-audit
# 期望：0 命中（organization-audit 归档除外）
```

---

### 任务 B5：全量验证 + 推送

1. cuda-foundations 全量：`cmake --preset default && cmake --build --preset default && ctest --preset default`。
2. 推送 `git push origin master`。
3. 回到根目录，把 `MASTER_PLAN.md` 中所有旧仓库名更新后也记录到本计划状态表。

**出口门槛**：GitHub 上 `AICL-Lab/cuda-foundations` 可见，旧 URL 自动跳转；所有仓库链接新名。

---

## 6. 阶段 C 任务明细：tiny-llm decode 性能攻坚

> **背景数字**：当前 TPOT 22.1ms/token（45.3 tok/s），llama.cpp（tg64，同模型同 prompt）TPOT 3.7ms（272 tok/s），比值 0.17。CUDA Graphs 只带来 -6.8%。**这 6 倍差距是下一步旗舰叙事必须正面回答的问题。**
>
> **方法论**：不预设优化方案。先 profiling 出 top-3 瓶颈，每个优化必须有 ncu before/after + greedy 输出不回归证据。

### 任务 C0：profiling 基线

1. 确保 A0 完成后构建可用；运行：
   ```bash
   cd tiny-llm
   TLLM_CUDA_GRAPHS=0 ./build/tiny_llm_bench models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
     --prompt "你好" --max-tokens 64 --warmup 3 --iters 10 --json > /tmp/tllm_base.json
   nsys profile -o /tmp/tllm_decode ./build/tiny_llm_bench <同上模型路径> \
     --prompt "你好" --max-tokens 64 --warmup 1 --iters 3
   ncu --set full --launch-count 5 --launch-skip 10 ./build/tiny_llm_bench <同上> \
     --prompt "你好" --max-tokens 64 --warmup 1 --iters 2
   ```
2. 产出并提交 `docs/performance/results/2026-08-XX-decode-profile.md`：
   - nsys 时间线结论（CPU/GPU 空洞、kernel 数量、launch 总开销）；
   - ncu top-5 kernel 表：kernel 名 / 调用次数 / 平均时长 / 主要 stall 原因 / 理论瓶颈类型；
   - 每层 decode 的 kernel 序列清单。
3. **不要优化任何代码**，本任务只产出证据。

**验收**：文档包含上述三张表；`ncu` 输出能回答"GEMM、attention、rmsnorm、launch 谁占大头"。

### 任务 C1：按证据优化第 1 瓶颈

执行规则：
1. 从 C0 的 top-5 表里选耗时占比最高的**一个** kernel 或 launch 组。
2. 先写它的 CPU/PyTorch 参考与差分测试（或复用现有测试），确保优化不破坏正确性。
3. 每次只改一个因素（如访存布局、grid 映射、共享内存、向量化、精度），跑 ncu 记录 before/after。
4. 必须同时跑：
   ```bash
   ./build/tiny_llm_tests --gtest_filter='*W8A16*:*Attention*:*KVCache*'
   ./build/tiny_llm_bench <model> --prompt "你好" --max-tokens 32 --warmup 2 --iters 5
   # 以及 greedy 输出与优化前 diff：逐 token 必须一致
   ```
5. 若 3 次尝试后 ncu 指标无改善，**回滚该优化**，把尝试与证据写进文档（负结果也是面试素材）。

**预期候选（供假设，不预设）**：
- `fp16_matmul_m1_kernel`（lm_head，N=151936）：当前每 warp 一列、K 维 stride 访存，可能 memory-bound；候选：共享内存分块 + 向量化加载、每 warp 多列复用 hidden、或 cuBLAS GEMV 对照实验；
- W8A16 24 层 decode GEMM；
- attention_decode 或 prefill kernel。

**验收**：1 个优化落地，有 before/after 数字与 ncu 指标变化；测试全绿。

**提交**：`perf(kernel): <kernel name> optimization with ncu evidence`

### 任务 C2：按证据优化第 2 瓶颈

同 C1 规则，选第二个瓶颈。如果 C1 已把 TPOT 降到 ≤12ms，可降低强度（只做文档化）；如果仍 >12ms，继续攻坚。

**验收**：TPOT 相比 C0 基线下降 ≥30%，或（若达不到）写出经过验证的瓶颈解释文档，明确"为什么没有便宜方案"。

### 任务 C3：复测与锚点文档化

1. 更新 README 基准快照表与 `docs/performance/` 结果归档。
2. 更新 llama.cpp 对比表，诚实标注比值变化。
3. 写 500 字以内"decode 瓶颈与优化叙事"（面试 2 分钟版）：瓶颈在哪、证据是什么、改了什么、数字变化、下一步。

**验收**：README 出现新的性能表；TPOT ≤ 12ms（阶段目标）或文档明确给出未达标原因与新方案。

---

## 7. 阶段 D 任务明细：分页 KV 端到端（D1 补完）

### 7.1 总体方案（先正确，后优化）

- tiny-llm 侧新增**独立于现有连续 KVCacheManager 的 paged 池**，只在 FFI 路径使用，不动 InferenceEngine 连续 KV 路径（避免破坏禁区）。
- paged-infer 的 BlockPool 继续是物理块的唯一 owner；tiny-llm 只做"给定 block_tables 的读写执行器"。
- ABI 升级为 v2：`TinyLlmConfig` 增加 `max_num_blocks`；`tinyllm_step` 增加 `num_blocks` 数组参数。Rust 布局守卫测试同步更新。

### 任务 D1：tiny-llm ABI v2 + 分页 KV 池

**文件**：`include/tiny_llm/ffi.h`、`src/ffi.cpp`、`include/tiny_llm/types.h`（如需要）、`tests/test_ffi.cpp`

1. `TinyLlmConfig` 在 `max_batch_size` 后增加 `int max_num_blocks;`（repr(C) 布局：9 个 int）。
2. `tinyllm_step` 签名增加一个参数（放在 `block_tables` 之后）：
   ```c
   int tinyllm_step(TinyLlmHandle *handle, const int *seq_ids, const int *input_tokens,
                    const int *positions, const int *seq_lens, const int *block_tables,
                    const int *num_blocks, const unsigned char *is_prefill,
                    int num_sequences, int *next_tokens, float *logprobs, int logprobs_k);
   ```
   语义：`block_tables` 是扁平化 `sum(num_blocks)` 个物理块索引；`num_blocks[i] = ceil((positions[i]+1)/block_size)`（decode）或 `ceil(seq_lens[i]/block_size)`（prefill，由实现统一为前者更稳）。
3. `TinyLlmHandleImpl` 新增：
   - `half *paged_k_pool, *paged_v_pool`：一次 `cudaMalloc`，大小 `max_num_blocks * block_size * num_layers * num_kv_heads * head_dim * sizeof(half)`（K/V 各一份，layout 为 `[layer][block][head][block_size][D]` 或等价的 flat index，**必须在 ffi.h 注释中写清公式**）；
   - `DeviceBuffer<int> d_block_tables`：每步上传块表；
   - 每层 scratch：`half *k_scratch, *v_scratch`（按 `max_num_blocks * block_size` 分配，层间复用）。
4. `tinyllm_load` 校验 `max_num_blocks > 0 && block_size > 0`，否则返回错误。
5. `tinyllm_allocate_sequence` / `tinyllm_free_sequence` 在策略 1 下改为 no-op（返回 0），但**保留符号**，文档注明"块由 paged-infer BlockPool 管理，此二函数仅为 ABI 兼容"。
6. 新增 C++ 测试：
   - 配置校验（max_num_blocks=0 失败）；
   - 池内存大小公式（`cudaMemGetInfo` 前后差，或直接单测 helper 函数 `paged_pool_bytes(config)`）。

**验收**：
```bash
cmake --build build -j && ./build/tiny_llm_tests --gtest_filter='*FFI*'
```

**提交**：`feat(ffi): ABI v2 paged KV pool and step signature`

### 任务 D2：scatter/gather 块式 KV kernel + FFI 步进接入

**文件**：`kernels/paged_kv.cu`（新增）、`kernels/paged_kv.cuh`（新增）、`src/ffi.cpp`

1. 实现两个 kernel（先写最简单正确版，每个元素一个线程，后续再优化）：
   - `scatter_blocks(const half *src, half *pool, const int *block_ids, int num_blocks, int block_size, int chunk_dim, int offset_tokens, cudaStream_t)`：把连续 `[tokens, chunk_dim]` 按块写到 pool；
   - `gather_blocks(half *dst, const half *pool, const int *block_ids, int num_blocks, int block_size, int chunk_dim, int visible_tokens, cudaStream_t)`：把 pool 中块读到连续 `[visible_tokens, chunk_dim]`。
   - 块内 token 索引 = block_id * block_size + token_offset；offset_tokens 为当前 step 首 token 在块内的偏移。
2. FFI 步进逻辑改为：
   - 对每个序列、每层：计算 Q/K/V（复用现有 per-layer 代码），RoPE 后：
     - **prefill**：`scatter_blocks` 写 K/V 到 pool；`gather_blocks` 读回连续 scratch；调用现有 attention prefill；
     - **decode**：把新 token 的 K/V `scatter_blocks` 写 pool；`gather_blocks` 读可见长度；调用现有 attention_decode；
   - 所有 layer 完成后统一推进位置（与现有 `advanceSeqLen` 语义对齐，但策略 1 下位置由 FFI 的 `SeqState` 管理）。
3. 数值不变量测试（`tests/test_kernels.cu`）：
   - scatter→gather 往返等于输入（多 block_size、跨块边界如 block_size=16, tokens=17）；
   - 多 layer 不互相覆盖（layer 1 写后 layer 0 读回不变）；
   - `head_dim` 非 2 幂、`num_blocks=1` 边界。
4. **先以策略 2 的数值为 oracle**：同一 prompt 下，`tiny_llm_step` 策略 1 与策略 2 输出逐 token 一致（可在 ffi 内加 `TLLM_PAGED_KV=1` 环境开关，默认 0 走连续路径，便于差分）。

**验收**：
```bash
cmake --build build -j
./build/tiny_llm_tests --gtest_filter='*Paged*:*FFI*'
TLLM_PAGED_KV=0 ./build/tiny_llm_demo <model> --prompt "你好" --max-tokens 32 --show-tokens > /tmp/t0.txt
TLLM_PAGED_KV=1 ./build/tiny_llm_demo <model> --prompt "你好" --max-tokens 32 --show-tokens > /tmp/t1.txt
diff /tmp/t0.txt /tmp/t1.txt && echo IDENTICAL
```

**提交**：`feat(kernels): paged KV scatter/gather and FFI strategy-1 path`

### 任务 D3：paged-infer TinyLlmExecutor 切换策略 1

**文件**：`src/tiny_llm_ffi.rs`、`src/tiny_llm_executor.rs`、`tests/` 中 FFI 布局守卫

1. Rust 侧 `TinyLlmConfig` 增加 `max_num_blocks: i32`（字段顺序与 C 一致），更新布局守卫测试。
2. `symbols::tinyllm_step` 声明增加 `num_blocks: *const c_int` 参数。
3. `TinyLlmExecutor`：
   - 从 `config.max_num_blocks` 传入 `max_num_blocks`（由 `EngineConfig` 计算）；
   - `execute()` 中把 `batch.block_tables` 展平 + 构造 `num_blocks[i]`（对 prefill 用 `blocks_for(context_len)` 或 batch 已有信息；对 decode 用 `blocks_for(context_len + 1)`），替换当前 `std::ptr::null()` 传参；
   - 删除/绕过 `ensure_allocated` 对 `tinyllm_allocate_sequence` 的依赖（策略 1 下 KV 由 BlockPool 分配），但保留策略 2 代码路径（用 cargo feature 或运行时配置开关）。
4. 能力声明与错误路径：`tinyllm_step` 非 0 时把错误传播为 `EngineError::BackendError`，不能卡死请求。

**验收**：
```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test    # 布局守卫 + executor 单测全绿
```

**提交**：`feat(executor): strategy-1 paged KV via ABI v2 block tables`

### 任务 D4：跨仓端到端 + 资源守恒

**文件**：`tests/tiny_llm_text_e2e.rs`（扩展）、`tests/concurrency_stress.rs`（复用）

1. 设置 `TINY_LLM_DIR=<tiny-llm>`、`TINY_LLM_MODEL=<qwen gguf>` 运行现有 e2e 测试，确认策略 1 下通过。
2. 新增/扩展测试：
   - 3 个并发请求、不同 prompt 长度、greedy；
   - 每个请求输出与 llama.cpp 同 prompt 逐 token 一致（用已有 fixture/方法）；
   - 运行前后 `engine` KV 利用率回到基线（`used + free == total`）；
   - 一个请求中途取消后，其物理块全部回收。
3. 记录 serving 层指标（请求数、完成数、失败数）。

**验收**：
```bash
TINY_LLM_DIR=... TINY_LLM_MODEL=... cargo test --features tiny-llm --test tiny_llm_text_e2e -- --nocapture
TINY_LLM_DIR=... TINY_LLM_MODEL=... cargo test --features tiny-llm --test concurrency_stress -- --nocapture
```

**提交**：`test(e2e): paged KV 3-way concurrency with llama.cpp token parity`

### 任务 D5：文档与边界收口

1. tiny-llm `ffi.h` 顶部注释更新为策略 1 已启用；`README` 项目状态表把"分页 KV"标 ✅。
2. paged-infer `README` 路线图 T11 打勾；`DEVELOPMENT_PLAN.md` 状态同步。
3. 在 `LEARNING_PATH.md`（cuda-foundations）的阶段 5 完成证据处，把 D1 的"待完善：分页 KV 暂未启用"改成已启用 + 引用测试名。
4. 写 5 条面试 QA（为什么块表在控制面、为什么 gather/scatter 先正确后优化、无抢占边界、资源守恒怎么验证、与 vLLM 的差距）。

**验收**：三个 README 描述与代码事实一致；`grep -n "暂未启用\|忽略 block_tables"` 只允许命中历史文档/CHANGELOG，不允许命中当前 README。

---

## 8. 阶段 E 任务明细（P2 收尾，按剩余时间选做）

> **状态**：E0–E4 ✅ 全部完成（2026-08-18）。可执行版见 [`PHASE2_NEXT_E.md`](PHASE2_NEXT_E.md)。

### E0 paged-infer：llama.cpp 诚实分歧 fixture（D4 请求 2 补真）

- ✅ D4 请求 2（`What is 2+2?`）改为"前缀一致 + EOS 终止 + 分歧注释"：
  llama.cpp `[17,10,17,16819,220,19,13,151645]` vs tiny-llm `[17,10,17,374,220,19,13,151645]`，
  第 4 个 token 为 W8A16 vs Q4_K_M 的 argmax 边界翻转（量化分歧），不伪装全序列一致。
- 提交：`test(e2e): honest llama.cpp divergence fixture for 2+2 prompt`

### E1 triton-fused-ops：Triton SGEMM + torch.library（补 MASTER_PLAN C3）

- ✅ 新增 `triton_ops/kernels/sgemm.py`：tiled GEMM，`tl.dot` 实现（fp32 用 `input_precision="ieee"`），
  与 `torch.mm` 差分测试（容差 1e-2，覆盖 M/N/K 非 2 幂边界），tests/test_sgemm.py 24 项全绿。
- ✅ 新增 `triton_ops/ops.py`：`torch.library.triton_op` 注册
  `triton_ops::sgemm` / `triton_ops::fused_rmsnorm_rope` / `triton_ops::fused_gated_mlp`；
  不可用时 fallback 到 `custom_op + register_fake`（两种路径均过测试）。
- ✅ README 增加"torch.library 自定义算子"小节（含与 vLLM/SGLang custom op 的对应关系）。
- 提交：`feat(triton): SGEMM kernel with differential tests` / `feat(torch): register custom ops via torch.library`

### E2 cuflash：一轮有数字的优化（补 MASTER_PLAN C2）

- ✅ E2a 修复 forward/backward `grid.y = batch*heads` 在 B*H > 65535 时 launch 非法：
  grid 展平到 x 维；回归测试 `ForwardTest.GridYOverflowSmoke`（B*H=65536）。
- ✅ E2b causal 边界块跳过：`q_last = min(q_start + BLOCK_M - 1, seq_len - 1)`，
  整块"未来"KV 块 break；`CausalNonTileAlignedSeqLen`（seq_len=257）验证。
- ✅ before/after 表写入 `docs/performance/causal-boundary-skip.md`：增益 ±2% 内（低于噪声，
  <10% 阈值），保留改动（数值不回归），主要价值是减少无效访存。
- 提交：`fix(forward): flatten grid.y batch*heads for >65535 launches` /
  `perf(forward): skip fully-future KV blocks in causal path`

### E3 组织级 landing README（可选）

- ✅ 新建 GitHub 仓库 `aicl-lab/aicl-lab`（public）：五仓地图（四层能力表）+ 阅读顺序 +
  Phase 2 证据摘要（TPOT 6.1ms、分页 KV 3 并发、改名 cuda-foundations）+ 计划归档副本
  （MASTER_PLAN / PHASE2_PLAN / PHASE2_NEXT*.md / docs/organization-audit）。
- ✅ 五仓 README 顶部各加一行 `📚 Portfolio map: https://github.com/open-infra-ai/aicl-lab`。

### E4 release tag 与 badge 校验

- ✅ 五仓打并推送 `phase-2-e` tag（cuda-foundations / triton-fused-ops / cuflash /
  tiny-llm / paged-infer）。
- ✅ 终检：旧名 `cuda-kernel-academy` 在实时内容 0 命中（计划文档与审计归档除外）；
  六仓 GitHub 可见无 MISSING；五个 README 链接全 200。

---

## 9. 阶段出口门槛（Phase 2 Definition of Done）

- [x] 五个仓库 `git status` 全部干净，`origin/master` 领先 0。
- [x] `cuda-foundations` 在 GitHub 可见；全组织源码 `cuda-foundations` 旧名 0 命中（审计归档除外）。
- [x] tiny-llm 构建/测试恢复全绿（157 passed / 8 skipped 基线）。
- [x] tiny-llm TPOT ≤ 12ms（或写出有 ncu 证据的不可达分析），README 性能表更新。
- [x] `TLLM_PAGED_KV=1` 与 `TLLM_PAGED_KV=0` 输出逐 token 一致。
- [x] paged-infer 策略 1 通过 3 并发 e2e，与 llama.cpp 对齐，资源守恒测试全绿。
- [x] 五个 README 的"策略 2 忽略 block_tables"等过期描述清零。
- [x] 每个任务一个 commit，commit message 可审计。

---

## 10. 建议执行顺序与时间

| 周 | 任务 | 关键产出 |
|---|---|---|
| 第 1 周 | A0 → A1 → A2 → A3 | 五仓干净、全推送、基线 tag |
| 第 1 周末 | B0（你手动改名）→ B1 → B2 → B3 → B4 → B5 | cuda-foundations 全链路改名 |
| 第 2 周 | C0 → C1 → C2 → C3 | decode 性能锚点：22ms → ≤12ms |
| 第 3 周 | D1 → D2 → D3 | ABI v2 + paged KV + Rust executor |
| 第 4 周 | D4 → D5 →（有余力 E1/E2/E4） | mini-vLLM 端到端叙事闭环 |

**给低成本模型的单任务提示词模板**：
```
请执行 /home/shane/github/aicl/PHASE2_PLAN.md 的任务 <ID>。
先读"0. 执行协议"和该任务完整明细，再改代码。
只改任务列出的文件；完成后运行"验收命令"，全绿后做一次 commit，
commit message 按任务末尾指定格式。最后用 5 行汇报：改了什么、验证输出、commit hash、遗留 NOTE。
```
