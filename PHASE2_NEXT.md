# PHASE 2 下一批任务（Batch 2：A1 → A3 → B0 → B5）

> **生成时间**：2026-08-18（A0 完成后）
> **执行方**：DeepSeek Flash / 其他低成本模型
> **上游文档**：`PHASE2_PLAN.md`（总计划，C/D/E 阶段明细以它为准）
> **状态更新**：任务 A0 ✅ 已完成（commit `3ddafcc`，tiny-llm 166 tests / 158 passed / 8 skipped）
>
> 本批任务的唯一目标：**五仓干净、全部推送、cuda-foundations 改名全链路完成**。
> C0 之后的 decode 性能攻坚与分页 KV 任务，完成本批后按 `PHASE2_PLAN.md` 第 6/7 节继续。

---

## 0. 执行协议（每条任务都遵守）

1. 一次只做一个任务，验收命令全绿后提交，再进入下一个。
2. 只改任务列出的文件/范围，不顺手重构。
3. commit message 按每个任务末尾指定格式，`git commit` 后 `git status --short` 必须干净（B 阶段改名过程除外）。
4. 所有性能数字只写实测；本批不涉及性能改动。
5. 遇到非预期错误：先记录原样输出到汇报里，不要 force push、不要 rebase、不要改历史。

---

## 任务 A1：提交 triton-fused-ops 的 8 个已验证修复（P0）

**背景（已由人工预先核实，不要再做主观判断）**：工作区 8 个文件的改动属于两类合法修复：
- TRIT-001 RoPE half-split 约定修正（`examples/rmsnorm_rope_example.py`、`triton_ops/reference/rmsnorm_rope.py`）
- Triton 3.x 兼容（`tl.math.erf`、torch dtype→triton dtype 映射、out_dtype）+ FP16 容差修正 + `num_heads*head_dim==hidden_dim` 校验（其余文件）

这些改动已经过 `python -m pytest -q` 验证：88 passed。

**执行步骤**：
1. `cd /home/shane/github/aicl/triton-fused-ops`
2. 运行并确认：
   ```bash
   python -m pytest -q          # 期望 88 passed
   git diff --check             # 期望无输出（无空白错误）
   ```
3. `git add` 以下 8 个文件（不要加其他文件）：
   ```
   examples/rmsnorm_rope_example.py
   tests/test_benchmark.py
   tests/test_edge_cases.py
   tests/test_gated_mlp.py
   tests/test_rmsnorm_rope.py
   triton_ops/kernels/gated_mlp.py
   triton_ops/reference/rmsnorm_rope.py
   triton_ops/validation.py
   ```
4. 提交：
   ```
   fix(triton): TRIT-001 half-split RoPE convention and Triton 3.x compatibility
   ```

**验收**：
```bash
git status --short --branch   # 干净，ahead origin/master 计数 +1
python -m pytest -q           # 88 passed
```

---

## 任务 A2：提交 cuflash-attn 文档同步 + 归档 PLAN.md（P0）

**背景（已预先核实）**：10 个已修改文件全部是 `docs/` 文档（补 BF16 API 分节、修正表述、同步 v0.5.0 代码事实），不含源码；未跟踪的 `PLAN.md` 是"v1.0 收敛计划"，内容与 `ROADMAP.md` 不重复且对未来 E2 有价值，应归档而不是删除。

**执行步骤**：
1. `cd /home/shane/github/aicl/cuflash-attn`
2. 先跑测试确认源码未受影响：
   ```bash
   cmake --preset release && cmake --build --preset release
   ctest --preset release --output-on-failure   # 期望 69/69
   ```
3. 提交 10 个文档修改（不要用 `git add -A`）：
   ```bash
   git add docs/algorithm.md docs/api-reference.md docs/architecture.md docs/building.md \
     docs/design/design-decisions.md docs/design/kernel-deep-dive.md \
     docs/design/tensor-core-migration.md docs/guide/quick-start.md \
     docs/performance/roofline-analysis.md docs/project-status.md
   git commit -m "docs(cuflash): sync v0.5.0 code facts and BF16 API sections"
   ```
4. 归档 PLAN.md：
   ```bash
   mkdir -p docs/development
   git mv PLAN.md docs/development/PLAN-v1.0.md
   ```
   然后在 `docs/development/PLAN-v1.0.md` 标题下一行插入：
   ```markdown
   > 状态：v1.0 计划草稿（归档），尚未执行；后续 cuflash 优化任务以组织级 PHASE2_PLAN.md 的 E2 为准。
   ```
5. 提交：
   ```
   docs(cuflash): archive v1.0 development plan under docs/development
   ```

**验收**：
```bash
git status --short --branch   # 干净
ls docs/development/PLAN-v1.0.md
```

---

## 任务 A3：推送五仓全部 ahead 提交 + 打 phase-1 基线 tag（P0）

**执行步骤**：对以下 5 个仓库依次执行（从 tiny-llm 开始，因为它 ahead 最多）：

```bash
cd /home/shane/github/aicl/tiny-llm
git log origin/master..HEAD --oneline    # 人工核对：全部是 Phase 1 + A0 的任务提交，无陌生改动
git push origin master
# 若 tag 已存在则跳过，否则：
git tag phase-1-complete && git push origin phase-1-complete
```

仓库顺序与预期 ahead：
1. `tiny-llm`（ahead 20，含 3ddafcc）
2. `paged-infer`（ahead 15）
3. `cuflash-attn`（ahead 4 + 本批 A2 的 2 个提交）
4. `triton-fused-ops`（ahead 3 + 本批 A1 的 1 个提交）
5. `cuda-kernel-academy`（ahead 2）

**注意**：
- 先跑 `git tag | grep phase-1-complete`，已存在就跳过打 tag。
- push 若被拒绝（non-fast-forward），**不要 force push**，停下来报告。
- 本机 GitHub 已通过 `gh` 认证，HTTPS push 应可成功；若提示输入密码，报告等待人工处理。

**验收**：
```bash
for d in tiny-llm paged-infer cuflash-attn triton-fused-ops cuda-kernel-academy; do
  (cd /home/shane/github/aicl/$d && echo "== $d ==" && git status -sb | head -1 && git log origin/master..HEAD --oneline | wc -l)
done
# 期望：每仓 ahead 0，工作区干净，tag phase-1-complete 已推送（检查 git ls-remote --tags origin | grep phase-1-complete）
```

---

## 任务 B0：GitHub 重命名 cuda-kernel-academy → cuda-foundations

**已确认**：本机 `gh` CLI 已登录（账号 LessUp，scopes 含 `repo`、`workflow`），可直接执行。

1. 重命名：
   ```bash
   gh repo rename -R aicl-lab/cuda-kernel-academy cuda-foundations --yes
   ```
2. 验证新仓库存在、旧 URL 重定向：
   ```bash
   gh repo view aicl-lab/cuda-foundations --json name,url
   gh api repos/aicl-lab/cuda-foundations --jq '.name'
   ```
3. 不要手动创建新仓库、不要 transfer owner、不要删除旧仓库。

**验收**：`gh api ... --jq '.name'` 输出 `cuda-foundations`；`https://github.com/aicl-lab/cuda-foundations` 可访问（`gh repo view` 成功即算）。

---

## 任务 B1：本地目录与 remote 更新

1. ```bash
   cd /home/shane/github/aicl
   mv cuda-kernel-academy cuda-foundations
   cd cuda-foundations
   git remote set-url origin https://github.com/aicl-lab/cuda-foundations.git
   git fetch
   git status -sb
   ```
2. 若 `git status -sb` 显示 tracking 正常（`## master...origin/master`）即为成功。

**验收**：
```bash
git -C /home/shane/github/aicl/cuda-foundations remote -v
# 期望 origin 指向 https://github.com/aicl-lab/cuda-foundations.git
```

---

## 任务 B2：仓内机械改名（namespace / 头文件 / CMake）

**执行步骤（严格按顺序）**：

1. `cd /home/shane/github/aicl/cuda-foundations`
2. 目录与文件重命名（git 跟踪）：
   ```bash
   git mv common/include/cuda_academy common/include/cuda_foundations
   git mv common/include/cuda_foundations/cuda_academy.hpp common/include/cuda_foundations/cuda_foundations.hpp
   ```
3. 文本替换（一次执行完，中间不要单独验证半替换状态）：
   ```bash
   grep -rl "cuda_academy\|CudaAcademy" --exclude-dir=.git --exclude-dir=build --exclude-dir=node_modules . | while read f; do
     sed -i 's/cuda_academy/cuda_foundations/g; s/CudaAcademy/CUDAFoundations/g' "$f"
   done

   grep -rl "cuda-kernel-academy\|CUDAKernelAcademy\|cuda-academy-common" \
     --exclude-dir=.git --exclude-dir=build --exclude-dir=node_modules . | while read f; do
     sed -i 's/cuda-kernel-academy/cuda-foundations/g; s/CUDAKernelAcademy/CUDAFoundations/g; s/cuda-academy-common/cuda-foundations-common/g' "$f"
   done
   ```
4. **人工核对 5 个特殊点**（sed 之后逐个 `grep -n` 确认）：
   - `common/CMakeLists.txt`：`add_library(cuda_foundations_common INTERFACE)`、`add_library(CUDAFoundations::common ALIAS ...)`、`EXPORT CUDAFoundationsTargets`
   - 根 `CMakeLists.txt`：`project(CUDAFoundations ...)`、`HOMEPAGE_URL "https://github.com/aicl-lab/cuda-foundations"`、文件顶部注释同步
   - `02-tensorcraft-core/include/tensorcraft/core/cuda_check.hpp`：`using CudaError = cuda_foundations::core::CudaError;`
   - `common/include/cuda_foundations/core/*.hpp`：namespace 开闭注释
   - 宏 `CA_CUDA_CHECK` 等**宏名保持不变**，只改宏展开里的命名空间限定
5. 明确不改：git 历史、`VERSION` 文件、LICENSE。

**验收**：
```bash
grep -rn "cuda_academy\|CudaAcademy\|CUDAKernelAcademy\|cuda-kernel-academy" \
  --exclude-dir=.git --exclude-dir=build --exclude-dir=node_modules . | wc -l   # 期望 0
cmake --preset default && cmake --build --preset default
ctest --preset default    # 期望与改名前一致（209 passed / 0 failed，skip 数不变）
```

**提交**：
```
refactor: rename namespace/CMake to cuda_foundations (mechanical, no behavior change)
```

---

## 任务 B3：文档站与仓库内链接改名

**执行步骤**：
1. `docs/.vitepress/config.ts`：
   - `repoUrl` → `https://github.com/aicl-lab/cuda-foundations`
   - `pagesUrl` → `https://aicl-lab.github.io/cuda-foundations/`
   - `base:` → `'/cuda-foundations/'`
2. `docs/package.json` 与根 `package.json`：`"name"` 中 `cuda-kernel-academy` → `cuda-foundations`。
3. `docs/tests/site-canonical-links.test.mjs` 等测试里的旧 URL 全部替换。
4. `.github/workflows/pages.yml`：检查并替换旧路径（若存在）。
5. `README.md`、`README.zh-CN.md`、`LEARNING_PATH.md` 内旧 URL 替换。
6. 构建文档站：
   ```bash
   cd docs && npm install && npm run docs:build   # 按 package.json 实际 script 名执行
   ```
   （无网络/依赖失败时记录报错，不阻塞本任务，但链接 grep 验收必须过。）

**验收**：
```bash
grep -rn "cuda-kernel-academy" README.md README.zh-CN.md LEARNING_PATH.md docs .github || true
# 期望 0 命中
```

**提交**：
```
docs: rename site and links to cuda-foundations
```

---

## 任务 B4：跨仓链接更新（其余 4 仓 + 根文档）

**范围**：
- `paged-infer/README.md`
- `tiny-llm/README.md`
- `cuflash-attn/README.md`
- `triton-fused-ops/README.md`
- `triton-fused-ops/triton_fused_ops.egg-info/PKG-INFO`（仅当该文件被 git 跟踪）
- `/home/shane/github/aicl/MASTER_PLAN.md`、`PHASE2_PLAN.md`
- `/home/shane/github/aicl/docs/organization-audit/**`（正文 URL 替换；归档文件名 `repos/cuda-kernel-academy.md` 保留原名）

**执行步骤**：
1. 全部把 `github.com/aicl-lab/cuda-kernel-academy` / `github.com/aicl-lab/cuda-kernel-academy` 替换为 `github.com/aicl-lab/cuda-foundations`（保持大小写与原文一致即可，主 URL 用 AICL-Lab）。
2. `docs/organization-audit/2026-08-13/repos/cuda-kernel-academy.md` **文件名不改**，在其标题下加一行：
   ```markdown
   > 归档说明：本审计完成于仓库改名 cuda-foundations 之前，文件名保留历史名称。
   ```
3. 每个被改仓库独立提交 `docs: point links to cuda-foundations`；根目录的 `MASTER_PLAN.md` / `PHASE2_PLAN.md` 改动也在同一提交里说明（根目录暂无 git 仓库，只改文件，不提交）。
4. README 改动不需要跑该仓库的完整测试（纯文档），但 triton 的 egg-info 若被跟踪，改完跑 `python -m pytest -q` 确认打包元数据未破坏（可选）。

**验收**：
```bash
grep -rn "cuda-kernel-academy" --exclude-dir=.git --exclude-dir=build --exclude-dir=target \
  --exclude-dir=.venv --exclude-dir=node_modules /home/shane/github/aicl | grep -v organization-audit
# 期望：0 命中
```

---

## 任务 B5：改名后全量验证与推送

1. `cd /home/shane/github/aicl/cuda-foundations`
2. 全量：
   ```bash
   cmake --preset default && cmake --build --preset default
   ctest --preset default        # 期望与 B2 一致
   git log origin/master..HEAD --oneline
   git push origin master
   git tag phase-2-rename && git push origin phase-2-rename
   ```
3. 用 `gh` 确认新仓库默认分支和 Pages：
   ```bash
   gh repo view aicl-lab/cuda-foundations --json name,defaultBranchRef
   ```
4. 报告里附上 `https://github.com/aicl-lab/cuda-foundations` 和旧链接重定向确认（`gh api repos/aicl-lab/cuda-kernel-academy` 应返回新名或 301，记录实际返回）。

**验收**：cuda-foundations 的 `origin/master` ahead 0；`git ls-remote --tags origin | grep phase-2-rename` 有输出。

---

## 本批完成后的下一批入口

本批（A1–B5）全部完成后，**不要直接开始 C 阶段**。先按以下方式汇报，等待下一步指令：

1. 每仓最终 `git status -sb` 一行；
2. `grep` 旧名命中数；
3. 每个任务的 commit hash 列表；
4. 遗留 NOTE。

下一步将执行 `PHASE2_PLAN.md` 第 6 节 **C0（tiny-llm decode profiling 基线）** 及后续性能攻坚；其任务明细已经写好在 PHASE2_PLAN，届时逐条下发即可。
