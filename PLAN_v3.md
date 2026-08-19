# PLAN v3 — 仓库完成状况分析与下一阶段执行计划

> **版本**：v3，2026-08-18
> **上游**：`MASTER_PLAN.md`（Phase 1）、`PHASE2_PLAN.md` + `PHASE2_NEXT*.md`（Phase 2 A–E）、`PHASE3_PLAN.md`（面试证据包）
> **执行方**：DeepSeek Flash / 其他低成本模型
> **本文档作用**：在 Phase 2（开发）与 Phase 3（面试材料）主体完成后，重新核验六仓真实状态，定义**收尾批次**与**面试执行期**的更新任务。本文件是后续唯一执行入口。

---

## 1. 代码完成状况分析（2026-08-18 核验）

### 1.1 六仓整体结论

**开发工作已经完成，源码处于冻结态，但 Git 收尾不完整。** 具体：

- 五个开发仓源码与 `phase-2-e` tag 完全一致，其后只有 1–2 个 **docs-only** 提交（ROADMAP/README 勾选对齐）；
- meta 仓 `aicl-lab` 已完成 Phase 3 全部 10 个任务的产物（interview/ 全套文档），但 **ahead 10 未推送，无 `phase-3-interview` tag**；
- `PHASE3_PLAN.md` 自身尚未归档进 meta 仓。

### 1.2 逐仓状态

| 仓库 | HEAD | 相对 origin | 源码完成度 | 未推送内容 |
|---|---|---|---|---|
| cuda-foundations | `38ccdcd` | ahead 2 | ✅ 冻结 | 2 docs commit（README/README.zh-CN/ROADMAP） |
| triton-fused-ops | `317347e` | ahead 1 | ✅ 冻结 | 1 docs commit（ROADMAP） |
| cuflash-attn | `e0862b4` | ahead 1 | ✅ 冻结 | 1 docs commit（ROADMAP） |
| tiny-llm | `15001c5` | ahead 1 | ✅ 冻结 | 1 docs commit（README/ROADMAP） |
| paged-infer | `fb9d670` | ahead 1 | ✅ 冻结 | 1 docs commit（ROADMAP） |
| aicl-lab (meta) | `8c3c6d4` | ahead 10 | ✅ 完成 | 10 interview docs commits |

### 1.3 冻结核验测试结果（来自 `interview/FREEZE_AUDIT.md`，已真实执行）

| 仓库 | 结果 | 说明 |
|---|---|---|
| cuda-foundations | 0 failed / 209 collected | **78 skipped**；不得声称 209 全执行，131 项实际执行 |
| triton-fused-ops | 116 passed + 1 skipped | skip = torch.compile smoke（已约定） |
| cuflash-attn | 71 tests，100% passed（1 skip） | PyTorch 对比项环境 skip |
| tiny-llm | 175 tests，174 passed + 1 skipped | 真模型门控已跑；skip = 第二 GQA 模型未提供 |
| paged-infer | 218 passed，0 failed | 本次未开 `--features tiny-llm`；3 并发 e2e 证据指向 `9c3700b`/`9c974d3` |
| aicl-lab | 无测试 | 文档仓 |

### 1.4 Phase 3 产物完整性核验

`aicl-lab/interview/` 已包含全部计划产物，且质量抽查通过：

- `EVIDENCE_MATRIX.md`：30/30 条（`grep -c '^### E'` = 30）
- `QA_BANK.md`：60/60 问（`grep -c '^### Q'` = 60）
- `NUMBERS_CARD.md`：有复现命令、commit、口径，并把无法归档的显存对比数字明确标为"不要当作实测"
- `FREEZE_AUDIT.md`：诚实区分 0 failed 与 skip；指出五仓 ahead 未 push
- `talks/` 7 篇、`MOCK_INTERVIEW.md`、`cross-cutting.md`、中英文简历条目、`PRESENTATION_CHECKLIST.md` 均在
- `grep -rn "未验证\|TODO" interview/` = 0 命中

### 1.5 诚实短板清单（面试必须主动承认，不修）

1. ncu/nsys 在本机不可用，性能证据链是 microbench + CUDA Event，不是 profiler 指标。
2. llama.cpp 对比口径不同：W8A16（重重量化）vs 原生 Q4_K_M。
3. "2+2" prompt 与 llama.cpp 第 4 token 分歧（is/equals），已诚实记录为前缀一致。
4. cuflash causal 边界跳过优化只有 ±2%，是负结果。
5. paged KV 第一版有 scatter/gather 显存往返，未做 kernel 级零拷贝。
6. 3 并发是学习验证规模，不是生产并发能力。

---

## 2. 当前未完成事项（唯一需要立即处理的 gap）

| ID | 问题 | 影响 | 优先级 |
|---|---|---|---|
| G1 | 五个开发仓各有 1–2 个 docs commit 未 push | GitHub 上 ROADMAP/README 落后于本地 | 🔴 P0 |
| G2 | meta 仓 ahead 10 未 push | 面试证据包只存在于本地 | 🔴 P0 |
| G3 | 无 `phase-3-interview` tag | 无法标识"面试材料完成"版本 | 🔴 P0 |
| G4 | `PHASE3_PLAN.md` 未归档到 meta 仓 | meta 计划档案缺最后一份 | 🟠 P1 |
| G5 | 根目录 `MASTER_PLAN.md`/`PHASE2_PLAN.md` 状态行仍指向 Phase 3 执行中 | 文档漂移 | 🟠 P1 |
| G6 | 五仓 docs-only 提交没有 tag | 冻结链路上 `phase-2-e` 与 HEAD 之间无标记 | 🟡 P2 |

---

## 3. 更新后的执行计划

### 阶段 K：收尾批次（模型可执行，预计 20 分钟）

#### K1：推送五个开发仓的 docs 提交并打 `phase-3-docs` tag

```bash
cd /home/shane/github/aicl
for d in cuda-foundations triton-fused-ops cuflash-attn tiny-llm paged-infer; do
  (cd $d && git log origin/master..HEAD --oneline && git push origin master && \
   git tag phase-3-docs && git push origin phase-3-docs)
done
```

**验收**：五仓 ahead 0；`git ls-remote --tags origin | grep phase-3-docs` 每个仓库都命中。

#### K2：归档 PHASE3_PLAN 并推送 meta 仓

1. `cp /home/shane/github/aicl/PHASE3_PLAN.md /home/shane/github/aicl/aicl-lab/`
2. 在 `aicl-lab/PHASE3_PLAN.md` 顶部加执行状态块：
   ```markdown
   > 执行状态：T1–T10 全部完成（2026-08-18）。产物在 interview/；冻结核验见 FREEZE_AUDIT.md；
   > 收尾推送与 tag 见 PLAN v3 阶段 K。
   ```
3. 同步更新根目录 `PHASE3_PLAN.md` 同样状态块（两个文件内容一致）。
4. meta 仓 commit：`docs: archive PHASE3_PLAN and mark T1-T10 complete`。

#### K3：推送 meta 仓全部 10 个 interview commits + 打 `phase-3-interview` tag

```bash
cd /home/shane/github/aicl/aicl-lab
git push origin master
git tag phase-3-interview && git push origin phase-3-interview
```

**验收**：ahead 0；`gh api repos/aicl-lab/aicl-lab/tags` 可见 `phase-3-interview`。

#### K4：更新根目录三个总控文档状态行

- `MASTER_PLAN.md`：Phase 3 状态改为 `✅ T1–T10 完成（aicl-lab@phase-3-interview）`；加一行"当前唯一执行入口：PLAN_v3.md"。
- `PHASE2_PLAN.md`：保持 Phase 2 完成标记，无需大改；把当前批次指针改为"Phase 2 已冻结，后续见 PLAN_v3.md"。
- `PHASE3_PLAN.md`：状态块按 K2 同步。
- 根目录这些文件本身不进 git（根目录不是仓库）；把变更复制进 meta 仓并 commit：`docs: sync master status pointers to PLAN v3`。

**阶段 K 完成定义**：
- [ ] 五仓 + meta 仓全部 `ahead 0`、工作区 clean；
- [ ] 远端 tags：五仓 `phase-3-docs`、meta `phase-3-interview`；
- [ ] meta 仓有 `PLAN_v3.md`（本文档）与 `PHASE3_PLAN.md`；
- [ ] 旧名 grep 0 命中（audit 归档除外）。

---

### 阶段 I：面试执行期（默认不开发，模型提供支持材料）

> 执行状态：I0–I7 ✅（aicl-lab@phase-i-ready，2026-08-18）。执行细节见 PLAN_I.md；

> **细化计划**：见 [`PLAN_I.md`](PLAN_I.md)（I0 文档漂移清理 → I1 防御手册 → I2 白板公式卡 → I3 快问训练卡 → I4 demo 脚本 → I5 复盘模板 → I6 岗位手册 → I7 推送 tag）。

> 这是给"你"的阶段，不是给模型刷代码的阶段。模型只做材料维护与模拟题生成。

| ID | 任务 | 产出 | 触发条件 |
|---|---|---|---|
| I1 | 按 `MOCK_INTERVIEW.md` 做第 1 次完整模拟并录音/文字复盘 | 复盘笔记（自己保存，不进仓库） | 阶段 K 完成后 |
| I2 | 根据第 1 次模拟的失分题，在 `interview/QA_BANK.md` 补"追问加固"段 | meta 仓 commit | 每次模拟后 |
| I3 | 生成 3 套 10 分钟快问卷（随机抽 QA_BANK 编号，附答案页） | `interview/drill-sets/01–03.md` | 面试前 2 周内 |
| I4 | 线上面试 demo 彩排：tiny-llm bench → paged-infer 3 并发测试 → triton op schema | 按 `PRESENTATION_CHECKLIST.md` 执行 | 面试前 3 天内 |

**阶段 I 纪律**：
- 不改五仓源码；
- 模拟中暴露的问题先记入"知识缺口"，禁止当场改代码证明自己；
- 每次面试后更新一次 `EVIDENCE_MATRIX.md`（如果面试官问到了新问题）。

---

### 阶段 D（原 P4）：按岗位的深度增量（默认冻结，必须明确解锁）

只有在你**明确指定一个编号**后才允许开工；每个编号单独出任务文档，不批量执行。

| 编号 | 主题 | 适合岗位 | 面试性价比 | 解锁建议 |
|---|---|---|---|---|
| D1 | paged decode 直接读 pool（消除 gather 往返） | kernel / runtime | 高 | 有 runtime/kernel 面试 |
| D2 | paged 路径接 CUDA Graphs | runtime | 高 | 有 runtime 面试 |
| D3 | chunked prefill 调度模拟 | serving | 中 | 有 serving 面试 |
| D4 | vLLM / SGLang good-first-issue 实做并提交 PR | serving / 通用 | 最高（外部证据） | 拿到任一面试邀约后 |
| D5 | FP8 E4M3 GEMM（CUDA 与 Triton 同题异构） | kernel | 中 | 目标 Hopper/Blackwell 岗位 |
| D6 | 多 GPU TP/EP 设计文档或 sharding 抽象 | serving / runtime | 中 | 目标大模型推理岗 |
| D7 | torch.compile / Inductor 补课 | 编译器方向 | 方向级 | 岗位要求编译器 |
| D8 | DDP/FSDP + allreduce 补课 | 训练 Infra | 方向级 | 岗位是训练而非推理 |

**默认推荐顺序**：先 I1 模拟一次 → 若目标是推理岗，解锁 **D4**（外部 PR 证据比任何第 6 个仓库都值钱）→ 再考虑 D1/D2。

---

## 4. 总完成定义（PLAN v3 的 DoD）

1. 阶段 K 完成：所有 gap 清零，tag 链路完整（`phase-2-e` → `phase-3-docs` → `phase-3-interview`）。
2. 阶段 I 至少完成一次 I1 模拟复盘。
3. 保持冻结：除面试材料与文档漂移外，五仓零源码改动。
4. 若触发阶段 D，每个 D 任务必须回到"一次一个任务 + 验收命令 + 证据指针"的老纪律。

---

## 5. 给 Flash 的单任务提示词模板

```
请执行 /home/shane/github/aicl/PLAN_v3.md 的阶段 K 任务 <编号>。
只做该任务列出的操作；push 前先 git log origin/master..HEAD --oneline 核对。
完成后输出：各仓 git status -sb、远端 tag 命中结果、commit hash、遗留 NOTE。
```
