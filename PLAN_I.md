# PLAN I — 面试执行期细化计划（Phase 3 收尾后的下一步）

> **版本**：2026-08-18
> **上游**：`PLAN_v3.md` 阶段 I；`PHASE3_PLAN.md`（T1–T10 已完成，`phase-3-interview` tag 已推送）
> **执行方**：DeepSeek Flash / 其他低成本模型
> **工作目录**：`/home/shane/github/aicl/aicl-lab`（meta 仓）
> **前置状态**：六仓全部 ahead 0 / clean；tag 链路 `phase-2-e → phase-3-docs（五仓）→ phase-3-interview（meta）` 完整。
>
> **本阶段定位**：不写任何五仓源码。只做三件事——① 清掉 K 阶段推送后残留的文档漂移；② 把 interview/ 从"完整"升级到"可直接排练"；③ 为 kernel / runtime / serving 三种目标岗位各准备一份角色化使用手册。

---

## 0. 执行协议

1. 一次一个任务，验收后独立 commit；commit message 按任务末尾格式。
2. 所有数字、命令、tag、commit 必须与 K 阶段报告一致（六仓 ahead 0、`phase-3-interview=9e0b4f7`、五仓 `phase-3-docs`）。
3. 禁止修改五仓源码；只允许改 `aicl-lab` 仓库内容。
4. 引用 QA_BANK 时只引用编号，不复制整段答案（保持单一事实源）。
5. "演练任务"（模拟面试、真人问答）由用户完成；本计划只让模型准备材料与模板，不替用户宣称"已通过模拟"。

---

## 1. 任务总览

```
I0 推送后文档漂移清理（P0，立即做）
I1 面试弱项防御手册 DEFENSE_PLAYBOOK.md
I2 白板公式卡 WHITEBOARD_CHEATSHEET.md
I3 快问训练卡 drill-sets/01-03.md
I4 线上面试 demo 脚本 LIVE_DEMO_SCRIPT.md
I5 模拟面试复盘模板 MOCK_DEBRIEF_TEMPLATE.md
I6 岗位角色手册 ROLE_PLAYBOOKS/（kernel / runtime / serving）
I7 推送 + phase-i-ready tag + PLAN_v3 状态收口
```

I0–I6 全部由模型完成；I7 是打包动作。完成后用户按 I5 做第一次模拟面试，再决定是否解锁 `PLAN_v3.md` 阶段 D（深度增量）。

---

## 2. 任务明细

### I0：推送后文档漂移清理（P0）

**已核实漂移点**：
1. `interview/PRESENTATION_CHECKLIST.md:3`——"推送与 phase-3-interview tag 需你明确说 push 后再做"，已过时。
2. 同文件 GitHub profile 表 aicl-lab 行——"声称已 push phase-3-interview（未 push 前）"，已过时。
3. `interview/EVIDENCE_MATRIX.md` E30 附近——"aicl-lab 无 tag / 五仓 docs 超前 origin 1–2"，已过时。
4. `README.md` 结尾——"phase-3-interview tag 在 push 之后才存在；未 push 前不要写"，已过时。

**改动规则**：
- 直接替换为 K 阶段事实：`phase-3-interview = 9e0b4f7`、五仓 `phase-3-docs`、六仓 ahead 0。
- `FREEZE_AUDIT.md` 是历史审计，**不要改写正文**；在文件末尾追加一节：
  ```markdown
  ## 5. K 阶段后补记（2026-08-18）
  - 本审计撰写时的 ahead/无 tag 状态已由 PLAN_v3 阶段 K 清零；
  - 当前 tag 链：phase-2-e → phase-3-docs（五仓）→ phase-3-interview（meta=9e0b4f7）。
  ```
- 其余文件出现"未 push/无 tag"的，区分历史陈述（保留并标注时态）与现状陈述（更新）。

**验收**：
```bash
grep -rn "需你明确说 push\|未 push 前\|尚未 push\|ahead 1 或 2" interview README.md || true
# 期望：只允许命中 FREEZE_AUDIT.md 的历史正文与其新增补记；其余 0 命中
git -C /home/shane/github/aicl/aicl-lab status --short   # 干净
```

**提交**：`docs(interview): reflect phase-K push and tag chain in stale documents`

---

### I1：`interview/DEFENSE_PLAYBOOK.md` 面试弱项防御手册

**内容**：把 PLAN_v3 §1.5 的诚实短板升级为"可排练的防御剧本"。至少 8 个条目，每条固定结构：

```markdown
### D<n>. <弱点>
- 面试官最可能怎么问：（一句尖锐提问）
- 10 秒承认口径：（先承认，不辩解）
- 证据式回答：（3–5 句，引用仓库文件/commit/命令）
- 绝对不能说：（红线句）
- 追问 2 个 + 答案要点：
```

**必须覆盖的 8 条**：
1. ncu/nsys 不可用（profiler 证据链替代）；
2. llama.cpp 对比是 W8A16 vs 原生 Q4_K_M，非同量化；
3. "2+2 is/equals"量化分歧；
4. cuflash causal skip 只有 ±2%（负结果）；
5. paged KV 第一版有 scatter/gather 额外往返；
6. 3 并发是学习验证，不是生产并发；
7. cuda-foundations 209 collected 里有 78 skip；
8. 单卡 RTX 3060 Laptop、单模型 Qwen2.5-0.5B 的泛化边界。

**验收**：8 条齐全；每条至少有 1 个 commit/文件指针；`grep -c '绝对不能说'` = 8。

**提交**：`docs(interview): defense playbook for known weak points`

---

### I2：`interview/WHITEBOARD_CHEATSHEET.md` 白板公式卡

**内容**：面试被要求手推/画图时用的速查。每节固定：公式 → 3–5 行推导 → 与本仓库数字的对应 → 常见错误。

**必须包含 12 节**：
1. online softmax 递推（含 m/l/O 三行递推）；
2. FlashAttention 为什么是 O(N) 辅助内存（分块 + rescale）；
3. Attention 标准 vs Flash 的 HBM 读写量（列出 Q/K/V/O 与中间矩阵的字节量）；
4. Roofline 与 arithmetic intensity（含如何判断 memory-bound）；
5. GEMM 分块算术：tile 大小、shared memory 容量、bank conflict 消除；
6. occupancy 粗算：block 线程数/共享内存/寄存器对 occupancy 的影响；
7. W8A16：量化与反量化公式、per-group scale 的 scale 数量；
8. GQA：`kv_head = q_head / (Hq/Hkv)` 与 KV cache 节省比；
9. RoPE half-split：`rotate_half` 公式与 interleaved 的区别；
10. KV cache 字节数公式（`2 × L × Hkv × D × S × bytes`）；
11. TTFT / TPOT / decode tok/s 口径与互相换算；
12. PagedAttention 碎片与 `<5%` 浪费说法的适用条件。

每节末尾用一行说明对应本仓库哪个文件/测试可现场打开。

**验收**：12 节齐全；第 1、2、12 节必须能被独立手推；无"略"字。

**提交**：`docs(interview): whiteboard formula cheatsheet`

---

### I3：`interview/drill-sets/` 三套 10 分钟快问卡

**内容**：生成 `01.md`、`02.md`、`03.md`。每套：
- 从 `QA_BANK.md` 抽 10 题：4 易 / 4 中 / 2 难；
- 套 01：总叙事 + tiny-llm 侧重；套 02：cuflash + CUDA/Triton 侧重；套 03：paged-infer + 跨仓边界侧重；
- 每题只写：编号、题目、一句话 cue（不复制答案）、答案所在 QA_BANK 小节号、对应 EVIDENCE 编号；
- 文末"10 分钟计时规则"：每题 60 秒，超时标记，答案页供复盘。

**验收**：三套各 10 题、无重复题号；`grep -c '^### '` 每套 = 11（10 题 + 标题不计则按实际）。

**提交**：`docs(interview): three 10-minute drill sets`

---

### I4：`interview/LIVE_DEMO_SCRIPT.md` 线上面试 demo 脚本

**内容**：
1. **场景 A（有 GPU）**：tiny-llm bench → paged-infer 3 并发 → triton op schema，每步给出：完整命令、预期关键输出（TPOT 6.09/6.1、请求 1 的 24+EOS 序列、schema 三行）、15 秒口播词、失败时切换的下一句。
2. **场景 B（无 GPU/共享屏幕）**：打开 `NUMBERS_CARD.md` 第 9 节五数 + `FREEZE_AUDIT.md` 测试表；给出 3 句口播词。
3. **场景 C（面试官自己跑命令）**：给出可直接粘贴到对方终端的 3 条命令与预计耗时（标注 skip 门控）。
4. **Preflight 检查**（demo 前 10 分钟）：六仓 `git status -sb` + tag 检查一条命令。
5. **失败恢复树**：bench 报错 / OOM / tokenizer 缺失 / 无 GPU 四条分支各一句应对。

**验收**：三种场景都有可复制命令；脚本里每个数字都出现在 NUMBERS_CARD 对应行。

**提交**：`docs(interview): live demo script with fallback branches`

---

### I5：`interview/MOCK_DEBRIEF_TEMPLATE.md` 模拟/真实面试复盘模板

**内容**：
- 面试元信息表（岗位、轮次、时长、面试官关注点）；
- 每道题记录表：题目 / 我的回答用时 / 是否引用证据 / 数字是否准确 / 追问表现（0–3）；
- 失分题汇总与"写回 QA_BANK 的新追问"区；
- 三次模拟的进步曲线模板（按 I5 建议连续做 3 次）；
- 用户须知：模拟必须录音或文字复盘；模型不能代替用户宣称"模拟已完成"。

**验收**：模板可直接填空；包含 MOCK_INTERVIEW.md 的 18 题编号对照表。

**提交**：`docs(interview): mock debrief template`

---

### I6：`interview/ROLE_PLAYBOOKS/` 岗位角色手册（三份）

**背景**：同一份材料，不同岗位的讲述顺序和重点应不同。生成：
- `kernel.md`：主推 cuflash-attn + triton-fused-ops + cuda-foundations；tiny-llm 只讲转置优化与 microbench；paged-infer 一分钟带过。
- `runtime.md`：主推 tiny-llm（decode 优化、graphs、FFI、paged KV）+ cuflash 集成点；serving 讲边界。
- `serving.md`：主推 paged-infer（调度/资源不变量/3 并发）+ tiny-llm 作为后端；kernel 只讲"为什么控制面不需要写 kernel"。

每份包含：20 分钟讲述时间轴、推荐的 QA_BANK 题号子集（≥15 题）、简历条目重排顺序、反问问题清单（≥3 个）。

**验收**：三份时间轴互不重复的题号集合覆盖 QA_BANK ≥40 题；每份反问 ≥3。

**提交**：`docs(interview): role playbooks for kernel/runtime/serving`

---

### I7：推送 + `phase-i-ready` tag + 状态收口

1. `git push origin master`；
2. `git tag phase-i-ready && git push origin phase-i-ready`；
3. 更新：
   - `README.md` 的 Interview 小节：加入 I0–I6 新文件链接与三种岗位手册；
   - `PLAN_v3.md` 阶段 I 状态行：`I0–I7 ✅（aicl-lab@phase-i-ready）`；
   - 根目录 `PLAN_v3.md` 与 `PHASE3_PLAN.md` 同步状态块（根目录不进 git，复制进 meta 仓）。
4. commit：`docs(interview): phase-I support pack and phase-i-ready tag`。

**验收**：meta 仓 ahead 0；`gh api repos/aicl-lab/aicl-lab/tags` 可见 `phase-i-ready`；`interview/` 新文件清单与 I1–I6 一致。

---

## 3. 本阶段完成定义

- [ ] I0 漂移清理完成（grep 验收通过）；
- [ ] I1–I6 六个新文件/目录存在且验收达标；
- [ ] `phase-i-ready` tag 已推送；
- [ ] **用户至少完成一次 I5 复盘**（这是用户侧门槛，模型在汇报里写"等待用户执行"，不能替用户打勾）。

---

## 4. 下一步的解锁条件（阶段 D）

完成 I 阶段后：
- 若已有目标岗位：从 `ROLE_PLAYBOOKS` 选对应手册，按 I4 做一次 live demo 彩排，然后**明确告诉模型 `解锁 D<n>`**（D1 paged 去 gather / D2 paged CUDA Graphs / D3 chunked prefill / D4 vLLM good-first-issue / D5 FP8 / D6 多 GPU 设计 / D7 编译器 / D8 训练栈）；
- 未指定岗位前，**不主动解锁任何 D**；保持五仓源码冻结。

---

## 5. 给 Flash 的提示词模板

```
请执行 /home/shane/github/aicl/PLAN_I.md 的任务 I0 → I1 → … → I7。
工作目录 /home/shane/github/aicl/aicl-lab。
遵守协议：只改 meta 仓、数字与 tag 以 K 阶段报告为准、不替用户宣称模拟完成。
每完成一个任务按验收自检并 commit；全部完成后输出：新文件清单、grep 验收结果、
phase-i-ready tag 命中、commit hash 列表、等待用户执行的事项。
```
