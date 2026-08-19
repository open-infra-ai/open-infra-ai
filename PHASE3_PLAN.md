# PHASE 3 执行计划：面试证据包与讲述能力（开发冻结后）

> 执行状态：T1–T10 全部完成（2026-08-18）。产物在 interview/；冻结核验见 FREEZE_AUDIT.md；
> 收尾推送与 tag 见 PLAN v3 阶段 K。阶段 I 支持包已完成：I0–I7 ✅（aicl-lab@phase-i-ready），见 PLAN_I.md。

> **版本**：2026-08-18
> **上游**：`MASTER_PLAN.md` + `PHASE2_PLAN.md`（A–E 全部完成，五仓 + meta 仓 `phase-2-e` tag）
> **执行方**：DeepSeek Flash / 其他低成本模型
> **本阶段目标**：**不再开发新功能**。把六个仓库的真实成果转化为可复述、可查证、能扛追问的面试材料。
> **工作目录**：`/home/shane/github/aicl/aicl-lab`（meta 仓，已 clone 到本地，origin = `https://github.com/open-infra-ai/aicl-lab.git`）
> **交付物目录**：meta 仓 `interview/`；全部任务完成后 push 到 meta 仓并打 `phase-3-interview` tag。

---

## 0. 执行协议（比开发阶段更严）

1. **禁止虚构**：每一条数据、每个"我们做了 X"都必须能指向 仓库/文件/commit/测试名/复现命令 五要素之一；找不到证据就写"未验证"或删掉该条。
2. **禁止为了好看改写数字**：所有数字从 README / docs / 测试输出原样复制，附 commit hash 与复现命令。
3. 一次一个任务，完成验收后 commit（meta 仓为主；若任务要求改五仓 README，则在对应仓独立 commit）。
4. 本阶段默认**不改任何五仓源码**；发现文档漂移只提交 docs 修正，并在 commit message 写 `docs:`。
5. 每个面试答案必须包含"会被追问的下一层问题"，不允许只写结论。

---

## 1. 任务总览

```
阶段 P0：冻结核验 + 证据矩阵（先做，其他任务的数据来源）
  T1 六仓冻结核验（git/测试/关键命令重跑快照）
  T2 EVIDENCE_MATRIX.md：30 条核心声明 → 证据指针
  T3 NUMBERS_CARD.md：全部数字 + 复现命令 + 口径 + 已知局限

阶段 P1：讲述脚本（10 分钟叙事）
  T4 00-master-narrative.md：五仓四层总叙事 + 30 秒电梯版
  T5 五个仓库各自的 10 分钟讲述稿
  T6 cross-cutting.md：三个必答题（为什么不用 llama.cpp/vLLM、Triton vs CUDA、Rust 在 Serving）

阶段 P2：问答库与模拟面试
  T7 QA_BANK.md：60 问，每问含答案要点 + 证据指针 + 追问
  T8 MOCK_INTERVIEW.md：45 分钟模拟脚本 + 自评表

阶段 P3：简历与 GitHub 呈现
  T9 resume-bullets：中英文 STAR 简历条目（每仓 3 条 + 总览 3 条）
  T10 meta README 收口 + 六仓呈现清单 + 推送 + phase-3-interview tag

阶段 P4（可选，需你明确说"开始"才做）：
  按目标岗位选一个深度主题做小增量（见第 8 节候选清单），否则保持冻结。
```

---

## 2. 阶段 P0 任务明细

### T1：六仓冻结核验（一个 meta commit）

**执行**：
1. 对六个仓库依次记录：
   ```bash
   cd /home/shane/github/aicl
   for d in cuda-foundations triton-fused-ops cuflash-attn tiny-llm paged-infer aicl-lab; do
     (cd $d && echo "== $d ==" && git status -sb && git log --oneline -1 && git tag --points-at HEAD)
   done
   ```
2. 重跑五个开发仓的测试（耗时约 20–40 分钟，必须真实执行并粘贴结果尾部）：
   - cuda-foundations：`cmake --preset default && cmake --build --preset default && ctest --preset default`
   - triton-fused-ops：`.venv/bin/python -m pytest -q`
   - cuflash-attn：`cmake --preset release && cmake --build --preset release && ctest --preset release --output-on-failure`
   - tiny-llm：`cmake --build build -j$(nproc) && ./build/tiny_llm_tests`
   - paged-infer：`cargo fmt --all -- --check && cargo clippy --all-targets -- -D warnings && cargo test`
3. 输出文件 `interview/FREEZE_AUDIT.md`：六仓状态表 + 测试结果 + 执行时间 + 环境（GPU/driver/CUDA）。

**验收**：所有测试通过（skip 数与各仓 README 声明一致）；`FREEZE_AUDIT.md` 提交到 meta 仓。

**提交**：`docs(interview): freeze audit for six repositories`

### T2：证据矩阵 `interview/EVIDENCE_MATRIX.md`

**要求**：30 条声明，每条格式：

```markdown
### E<n>. <一句话声明>
- 证据类型：测试 / benchmark / 代码路径 / 文档
- 位置：`<repo>/<file>:<line>`（或测试名）
- 关键 commit：`<short sha> <subject>`
- 复现命令：```bash ...```
- 口径/限制：<诚实说明>
```

**必须包含的 30 条（不允许替换主题，细节从仓库实际内容提取）**：
1. SGEMM 优化阶梯从 naive 到 Tensor Core，每步有实测 TFLOPS。
2. CUDA 与 Triton 的 SGEMM"同题异构"对比存在。
3. Triton 三个融合算子有独立参考实现与差分测试。
4. TRIT-001 RoPE half-split 约定 bug 已修复并有测试。
5. torch.library 注册的三个自定义 op 可直接调用（附 schema）。
6. FlashAttention 前向+反向多精度通过差分测试。
7. grid.y > 65535 的 launch bug 已修复并有回归测试。
8. causal 边界块跳过优化实测 ±2%，诚实记录为负结果。
9. FlashDecoding / Split-KV 前向已实现并有 benchmark。
10. tiny-llm 从 GGUF 到文本一条命令跑通真实 Qwen2.5-0.5B。
11. tokenizer 与 HF tokenizers 差分逐 id 对齐（30 例 417 token 或仓库实际数据）。
12. GGUF 反量化（Q4_0/Q5_0/Q8_0/Q4_K/Q6_K）与 Python gguf 参考一致。
13. W8A16 量化推理路径有 CPU/PyTorch 参考差分测试。
14. GQA（14→2）与 RoPE 进入真实计算路径并有精确规格测试。
15. 转置权重 M==1 GEMM 优化：microbench 与端到端 before/after 数字。
16. CUDA Graphs 默认开启，graphs on/off greedy 输出逐 token 一致。
17. 与 llama.cpp 的对比方法论文档 + 公平性声明。
18. "2+2 is/equals"量化分歧案例被诚实记录为前缀一致 + EOS 断言。
19. FFI C ABI v2：9-int 布局 + num_blocks，Rust 布局守卫测试。
20. paged KV 策略 1 与连续 KV 策略 2 真模型逐 token 差分一致。
21. paged-infer 3 并发 e2e 与 llama.cpp 参考对齐（请求 1 24+EOS）。
22. 分页 KV 显存 3030MiB vs 连续 KV 5118MiB（同机实测）。
23. paged-infer 调度器资源守恒不变量（used+free==total）有属性测试。
24. paged-infer 内存水位线/HOL/优先级调度/NaN/Unicode 修复各有回归测试。
25. paged-infer OpenAI 兼容 API + SSE + metrics 有 server integration 测试。
26. 五仓 IN/OUT 边界声明齐全；04-inference-engine 已降级教学预览。
27. 组织级改名 cuda-kernel-academy → cuda-foundations，旧名 0 命中。
28. ncu/nsys 在本机不可用的替代证据链（kernel microbench + 方法论文档）。
29. 性能数字诚信：所有 benchmark 附硬件/commit/复现命令。
30. 六仓 GitHub 可见 + phase-1/2/rename/e tag + landing repo。

**验收**：30 条全部有位置与命令，无"TODO"。

**提交**：`docs(interview): evidence matrix for 30 core claims`

### T3：数字卡 `interview/NUMBERS_CARD.md`

**要求**：把以下数字全部收录，每一项都带 复现命令 + 硬件 + commit + 口径限制：

| 类别 | 数字（从仓库原文复制，不要凭记忆改写） |
|---|---|
| tiny-llm TPOT/吞吐 | C 阶段前后、graphs on/off、llama.cpp 基线 |
| microbench | lm_head 10.0→0.98ms、N=4864 0.163→0.049ms 等 C0/C1 表 |
| 显存 | W8A16 峰值、转置副本增量、paged vs contiguous KV |
| cuflash | FP16/FP32 causal/non-causal 各 seq_len、grid overflow smoke、causal skip ±2% |
| triton | gated_mlp ≈3.45ms、rmsnorm_rope ≈0.10ms、SGEMM 差分 24 项 |
| 测试规模 | 各仓 test 数量与 skip 数 |
| llama.cpp 对齐 | prompt 1 全序列 24+EOS；prompt 2 前缀 3 + 分歧 |

额外要求：
- 单独一节"**数字的边界**"：W8A16 vs Q4_K_M 重量化差异、llama.cpp 口径差异、RTX 3060 Laptop 单一硬件、ncu 不可用、causal 优化在噪声内。
- 最后一节"**如果面试官只让我报 5 个数**"：给出 5 个首选数字及一句话解释。

**验收**：每个数字行尾有 commit/命令；`grep -c '未验证'` 为 0。

**提交**：`docs(interview): numbers card with reproduction and caveats`

---

## 3. 阶段 P1 任务明细

### T4：总叙事 `interview/talks/00-master-narrative.md`

内容：
1. 30 秒电梯版（中文 + English 各一段，≤120 字/词）。
2. 四层能力图 + 五仓分工 + meta landing。
3. 完整能力链叙事：同一 prompt 如何在五仓间走通（GGUF → W8A16 GEMM → attention → token → scheduler → 3 并发 serving）。
4. 三个最有说服力的故事排序：① 6.6×→1.65× 的 decode 优化；② paged KV 从策略 2 到策略 1 的 ABI v2 升级；③ 差分测试抓到 `append_kv_at` 只写 1 token 的真 bug。

**验收**：电梯版可直接朗读；能力链每步有仓库/文件指向。

**提交**：`docs(interview): master narrative`

### T5：五个仓库 10 分钟讲述稿

目录 `interview/talks/`，每份结构固定：
```
0. 一句话定位（20 字内）
1. 2 分钟：做什么、边界、为什么这样切
2. 3 分钟：一个最难的实现细节（从代码出发）
3. 2 分钟：一个优化/调试故事（before→证据→改动→after→代价）
4. 2 分钟：验证方法（差分/不变量/负结果）
5. 1 分钟：诚实短板与下一步
6. 追问清单（至少 8 问，附答案要点）
```
对应文件：
- `01-cuda-foundations.md`
- `02-triton-fused-ops.md`
- `03-cuflash-attn.md`
- `04-tiny-llm.md`
- `05-paged-infer.md`

**特别要求**：tiny-llm 稿的"优化故事"必须用 microbench 表与 TPOT 24.35→6.56→6.09ms；paged-infer 稿的"验证故事"必须用 3 并发 e2e 与资源守恒；cuflash 稿必须诚实讲 causal ±2% 负结果。

**验收**：每份稿中的所有数字都能在 `NUMBERS_CARD.md` 找到对应行。

**提交**：每份稿一个 commit（可合并为 `docs(interview): per-repo 10-minute talk scripts` 一个 commit，便于 review）。

### T6：三个必答题 `interview/cross-cutting.md`

1. **为什么不用 llama.cpp / vLLM？** 必须包含：学习目标 vs 生产目标、可控性、从它们学到的设计（paged KV、CB、GGUF、CUDA Graphs）、以及"如果做产品我会直接选它们"的清醒结论。
2. **什么时候用 Triton，什么时候用 CUDA C++？** 必须引用：同一 SGEMM 的两套实现、FA 的 Triton 参考 vs CUDA 深挖、Triton 3.x dtype 兼容问题、torch.library 注册。
3. **Serving 控制面为什么用 Rust？** 必须包含：C ABI v2 跨语言边界、布局守卫测试、Rust 属性测试、以及"语言不是重点，边界才是"的升华。

每篇 800–1200 字，文末 3 个追问 + 答案要点。

**验收**：三篇都引用至少一个 commit 或测试名。

**提交**：`docs(interview): cross-cutting answers`

---

## 4. 阶段 P2 任务明细

### T7：问答库 `interview/QA_BANK.md`（60 问）

**硬性结构**（每题）：
```markdown
### Q<n>. <问题>
- 一句话答案：
- 展开（3–5 点）：
- 证据：`<repo>/<file>` 或 `<commit>`
- 追问 1：<问题> → <要点>
```

**60 问分布（题目自拟，但必须覆盖下列全部主题）**：
- CUDA 基础（10 问）：线程/block/warp、occupancy、bank conflict、shared memory、GEMM 阶梯每步动机、WMMA 对齐、cp.async、roofline、launch 开销、教学仓与 runtime 的边界。
- Triton（10 问）：block 抽象、mask、tl.dot、autotuner、RMSNorm+RoPE 融合收益、half-split vs interleaved、TRIT-001 怎么发现的、torch.library 注册、Triton vs CUDA、Triton 3.x dtype 兼容。
- FlashAttention（12 问）：online softmax 推导、O(N) 内存、fwd/bwd 关系、causal mask、grid.y 65535、causal skip 为什么只有 ±2%、FlashDecoding、与 FA2/FA3 差距、FP16/BF16 数值、logical HBM 口径、ctypes vs torch.library、反向数值稳定性。
- tiny-llm（14 问）：GGUF 结构、Q4_K/Q5_0 反量化、W8A16 含义、per-group scale、GQA 映射、RoPE 位置、KV append/advance 两段式、`append_kv_at` bug、转置权重优化原理、CUDA Graphs 捕获范围与 device 参数化、TTFT/TPOT 口径、量化分歧"is vs equals"、FFI ABI v2、与 llama.cpp 差距还剩多少。
- paged-infer / serving（14 问）：PagedAttention 碎片问题、block table、CB vs static batching、状态机、三层准入、内存水位线+decode reserve、HOL 修复、优先级调度、无抢占边界、swap/recompute、分页 KV 策略 1 数据流、3 并发验证、SSE token 级流式的边界、Rust 后端 trait 设计。

**验收**：60 问全部有"证据"与"追问"字段；`grep -c 'TODO'` 为 0。

**提交**：`docs(interview): 60-question QA bank`

### T8：模拟面试脚本 `interview/MOCK_INTERVIEW.md`

内容：
1. 45 分钟标准流程：3min 总叙事 → 10min tiny-llm 深挖 → 10min FA → 10min serving → 5min 基础快问 → 5min 反问准备。
2. 面试官逐题脚本（从 QA_BANK 抽 18 题，按难度梯度排列）。
3. 自评表（每题 0–3 分：证据引用 / 数字准确 / 边界诚实 / 追问应对）。
4. "红灯清单"：哪些说法会被判定减分（如把 LogicalHBM 说成物理带宽、把 ±2% 说成优化成功、把 3 并发说成生产并发能力）。
5. 一次完整模拟的录音/文字复盘模板。

**验收**：18 题都有 QA_BANK 编号对应。

**提交**：`docs(interview): 45-minute mock interview script and rubric`

---

## 5. 阶段 P3 任务明细

### T9：简历条目 `interview/resume-bullets.zh.md` 与 `resume-bullets.en.md`

要求：
1. 总览 3 条（把五仓能力链压缩成 3 个 STAR 条目）。
2. 每仓 3 条，每条：**动作 + 量化结果 + 证据**，≤40 字（中文）/≤25 词（英文）。
3. 数字只从 `NUMBERS_CARD.md` 取。
4. 每个条目尾注 `→ E<n>`（对应 EVIDENCE_MATRIX 编号），面试被问细节时按编号查证。
5. 提供"无 GPU 面试环境下的备选说法"（如果对方线上环境不能跑 demo，如何用文档/测试记录证明）。

**验收**：每一条目都有 E 编号；无未定义数字。

**提交**：`docs(interview): resume bullets zh/en`

### T10：meta README 收口 + 六仓呈现清单 + 推送

1. `aicl-lab/README.md` 增加 `## Interview Evidence` 小节，链接 `interview/` 全部文件。
2. 新增 `interview/PRESENTATION_CHECKLIST.md`：
   - GitHub profile 建议（pinned 仓库顺序：tiny-llm、cuflash-attn、paged-infer、cuda-foundations、triton-fused-ops、aicl-lab）；
   - 每个仓库 README 状态表复查要点；
   - 面试前 24h 检查命令（一条 `git status` 六仓 + 一条测试命令）；
   - 线上面试 demo 顺序（先 tiny-llm bench，再 paged-infer 3 并发测试，再 triton op schema）。
3. 推送 meta 仓并打 tag：
   ```bash
   cd /home/shane/github/aicl/aicl-lab
   git push origin master
   git tag phase-3-interview && git push origin phase-3-interview
   ```
4. 把根目录 `PHASE2_PLAN.md` 状态更新为"Phase 3 完成"（本地文件；如 meta 仓有副本则同步）。

**验收**：meta 仓 ahead 0；`gh api repos/aicl-lab/aicl-lab/tags` 可见 `phase-3-interview`。

**提交**：`docs(interview): presentation checklist and phase-3 tag`

---

## 6. 全部完成后的汇报格式

1. meta 仓 HEAD 与 tag；
2. `interview/` 文件清单与行数；
3. 五仓 freeze audit 的测试结果表；
4. 三个自查：
   - 数字卡中"未验证"计数 = 0；
   - QA_BANK 题数 = 60、TODO = 0；
   - 简历条目 E 编号无缺失。

---

## 7. 阶段 P4（可选深度主题，默认不做）

以下主题**只在用户明确说"开始 P4-<编号>"后才做**，否则保持冻结。候选按"面试性价比"排序：

| 编号 | 主题 | 适合岗位 | 预计收益 |
|---|---|---|---|
| P4-1 | paged decode 直接读 pool（消除 gather 往返） | runtime/kernel | 中高 |
| P4-2 | paged 路径接 CUDA Graphs | runtime | 高 |
| P4-3 | chunked prefill 调度模拟 | serving | 中 |
| P4-4 | vLLM/SGLang good-first-issue 实做 | serving/通用 | 高（外部证据） |
| P4-5 | FP8 E4M3 GEMM（Triton/CUDA 同题异构） | kernel | 中 |
| P4-6 | 多 GPU TP/EP 设计文档或 sharding 抽象 | serving/runtime | 中 |
| P4-7 | torch.compile/Inductor 补课（编译器方向） | 编译器 | 方向级 |
| P4-8 | DDP/FSDP + allreduce 补课（训练方向） | 训练 Infra | 方向级 |

**推荐**：先完成面试准备（P0–P3）并至少做过一次模拟面试，再决定是否开 P4-1/P4-4。

---

## 8. 给 Flash 的单任务提示词模板

```
请执行 /home/shane/github/aicl/PHASE3_PLAN.md 的任务 <T编号>。
工作目录 /home/shane/github/aicl/aicl-lab（meta 仓）。
遵守第 0 节协议：禁止虚构、数字必须带证据指针与复现命令。
完成后按任务验收自检并 commit，汇报 5 行：产出文件、关键证据数、验收命令输出摘要、commit hash、遗留 NOTE。
```
