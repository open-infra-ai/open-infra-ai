# 45 分钟模拟面试

题目全部来自 [`QA_BANK.md`](QA_BANK.md)。数字以 [`NUMBERS_CARD.md`](NUMBERS_CARD.md) 为准。总叙事见 [`talks/00-master-narrative.md`](talks/00-master-narrative.md)。

---

## 1. 时间表

| 分钟 | 块 | 做什么 |
|------|----|--------|
| 0–3 | 总叙事 | 电梯版 + 四层图 +「不是一个进程」。不要展开 kernel。 |
| 3–13 | tiny-llm | 转置、Graphs、`append_kv_at`、口径、量化分歧、ABI |
| 13–23 | FlashAttention | online softmax、grid.y、causal 负结果、LogicalHBM、Split-KV |
| 23–33 | serving | 分页碎片、HOL、3 并发边界、SSE、Rust capabilities |
| 33–38 | 基础快问 | 阶梯负结果、`cp.async` 诚实、RoPE 契约 |
| 38–45 | 反问 | 见第 5 节；不要用反问补讲没问到的 flagship |

**超时规则**：任何一题超过 90 秒还没落到文件/数字，面试官打断，切下一题。

---

## 2. 面试官逐题脚本（18 题，难度递进）

每题：先听 一句话答案，再追问栏。评分用第 3 节四维 0–3。

### 开场暖场（叙事之后立刻问）

**M1 · Q10**（易）教学仓和 tiny-llm 什么关系？
- 期望：04 是预览；runtime 不 include 教学头文件；面试旗舰不是 cuda-foundations。
- 追问：209 tests 全跑过了吗？ → 0 failed ≠ 209 执行；78 skip。

### tiny-llm 深挖（约 10 分钟）

**M2 · Q41**（中）为什么转置能把 decode 打下来？
- 期望：stride=N vs stride=1；24.348 → 6.560 → 6.087；lm_head 10.0002 → 0.9794；显存 2494 → 3368。
- 追问：这是 Tensor Core 吗？ → 不是。

**M3 · Q42**（中）CUDA Graphs 捕获范围？
- 期望：decode device 路径；默认 ON；不 capture advance/采样；device `write_pos`。
- 追问：paged 接 Graphs 了吗？ → 没有。

**M4 · Q40**（难）`append_kv_at` 出过什么真 bug？
- 期望：prefill 多 token 只写 1 行；策略 1 vs 2 差分发现；`7b456cd` / `elementwise.cu`。
- 追问：为什么 decode 测不出来？ → `num_tokens` 永远是 1。

**M5 · Q43**（中）TTFT/TPOT 怎么报才不被打穿？
- 期望：TPOT 6.087 vs llama 3.7，比值 1.65；量化不同；TTFT 禁止直接相除。
- 追问：1.65× 算追上了吗？ → 数量级接近；C1 前约 6.6×。

**M6 · Q44**（难）请求 2 为什么 is/equals 不一致？
- 期望：374 vs 16819；前缀 `[17,10,17]` + EOS 151645；W8A16 vs Q4_K_M。
- 追问：调度写错了吗？ → 没有。

**M7 · Q45**（中）FFI 几个 int？缺了会怎样？
- 期望：9；第 9 个 `max_num_blocks`；Rust `sizeof==36`。
- 追问：ABI 吃掉 6.09 ms 了吗？ → 无微秒表，不报；相对 GEMM 不是主因。

### FlashAttention（约 10 分钟）

**M8 · Q21**（中）online softmax 维护什么？
- 期望：`m` 与 `l`；`exp(m-m_new)` 缩放；不物化 S。
- 追问：为什么 logsumexp 用 FP32？ → 反向重建。

**M9 · Q25**（中）grid.y 65535 是什么？
- 期望：B*H 展平；`GridYOverflowSmoke`；`d144765`。
- 追问：为什么 seq_len=1？ → 打 launch，不缠数值。

**M10 · Q26**（难）causal skip 是优化成功吗？
- 期望：**不是**。±2%，低于 10% 阈值；表 +1.2%～−1.9%。
- 追问：为什么保留？ → 语义与无效访存，不是加速。

**M11 · Q30**（难）LogicalHBM 是物理带宽吗？
- 期望：**不是**。模型化流量；counter 故意叫 LogicalHBM。
- 追问：为什么还要报？ → 对照不同形状；防止自己把模型当 ncu。

**M12 · Q27**（中）FlashDecoding 干什么？接到 tiny-llm 了吗？
- 期望：KV 分块 + reduce；query_len=1；**未**接入 runtime。
- 追问：和训练前向是不是同一 kernel？ → 不是整段替换。

### serving（约 10 分钟）

**M13 · Q47**（中）PagedAttention 解决什么？显存数字报哪对？
- 期望：预留 max_seq 的浪费；**不要** 3030/5118；正确性靠 E20 差分。
- 追问：3368 MB 是分页省下来的吗？ → 不是，是转置副本。

**M14 · Q53**（中）HOL 怎么测？
- 期望：`test_small_pending_request_not_blocked_by_large_one`；无抢占时更要命。
- 追问：这是公平性指标吗？ → 先是活性。

**M15 · Q58**（难）3 并发证明了什么？
- 期望：正确性 fixture；请求 1 的 24+EOS；T1 没开 `--features tiny-llm`。
- 追问：等于生产并发？ → 不等于。适配器还 clamp 4。

**M16 · Q59**（中）SSE 是 token 级流式吗？
- 期望：SimpleTokenizer 是；HF 结束时一整块；`data: [DONE]` 有测。
- 追问：测的是 GPU QPS 吗？ → 不是，CPU 参考后端 37 项。

**M17 · Q60**（中）为什么控制面用 Rust？
- 期望：边界可测；`GREEDY_ONLY`；不是「Rust 更快」。
- 追问：换 C++ 控制面故事还成立吗？ → 成立，若 ABI 与不变量同样严。

### 基础快问（约 5 分钟）

**M18 · Q7 + Q3 + Q16 捆绑**（易→中）
- double buffer 用了 `cp.async` 吗？ → **没有**。
- bank-conflict-free 更快吗？ → 本机 0.66 < tiled 0.92。
- TRIT-001 是性能 bug 吗？ → 是 RoPE 排列契约。

若时间只够一问：优先 Q7（最容易吹过头）。

---

## 3. 自评表

每题四维，0–3 分。满分 18×4=72。建议及格线：每维平均 ≥2，且红灯清单 0 触发。

| 维 | 0 | 1 | 2 | 3 |
|----|---|---|---|---|
| 证据引用 | 纯口嗨 | 仓名 | 文件或测试名 | 文件 + commit/测试名 |
| 数字准确 | 编数 | 数量级对、口径错 | 与数字卡一致 | 主动报口径（量化/iters/硬件） |
| 边界诚实 | 把作品集说成 vLLM | 漏说 OUT | 主动报 OUT | 负结果/skip/未跑 e2e 都主动说 |
| 追问应对 | 卡死或改口 | 能挡一句 | 落到下一层证据 | 能把追问接到另一仓契约 |

**记录表**（模拟时复制）：

| 题 | QA | 证据 0–3 | 数字 0–3 | 边界 0–3 | 追问 0–3 | 红灯？ | 备注 |
|----|----|----------|----------|----------|----------|--------|------|
| M1 | Q10 |  |  |  |  |  |  |
| M2 | Q41 |  |  |  |  |  |  |
| M3 | Q42 |  |  |  |  |  |  |
| M4 | Q40 |  |  |  |  |  |  |
| M5 | Q43 |  |  |  |  |  |  |
| M6 | Q44 |  |  |  |  |  |  |
| M7 | Q45 |  |  |  |  |  |  |
| M8 | Q21 |  |  |  |  |  |  |
| M9 | Q25 |  |  |  |  |  |  |
| M10 | Q26 |  |  |  |  |  |  |
| M11 | Q30 |  |  |  |  |  |  |
| M12 | Q27 |  |  |  |  |  |  |
| M13 | Q47 |  |  |  |  |  |  |
| M14 | Q53 |  |  |  |  |  |  |
| M15 | Q58 |  |  |  |  |  |  |
| M16 | Q59 |  |  |  |  |  |  |
| M17 | Q60 |  |  |  |  |  |  |
| M18 | Q7/Q3/Q16 |  |  |  |  |  |  |
| **合计** | |  |  |  |  |  | /72 |

---

## 4. 红灯清单（出现即该维记 0，并在复盘标「减分句」）

1. 把 **LogicalHBM** 说成 ncu 物理带宽。
2. 把 causal skip **±2%** 说成优化成功 / 「我们更快」。
3. 把 **3 并发** 说成生产并发能力或 QPS。
4. 背 **3030 vs 5118 MiB**（仓库无归档）。
5. 说 **209/209** 全执行（78 skip）。
6. 说 double-buffer / 教学流水线用了 **`cp.async` / TMA**。
7. 说 tiny-llm **比 llama.cpp 快**，或不提 Q4_K_M vs W8A16。
8. 说 cuflash 已接到 generate 路径 / 已注册成 vLLM op。
9. 说 HF tokenizer 路径是 **token-level SSE**。
10. 说本次 freeze **跑过** `--features tiny-llm` e2e。
11. 把 3368 MB 说成 **分页省显存**。
12. 把 skip 的 `torch.compile` / pytorch comparison 说成通过。
13. 标题党：「Rust 比 C++ 更适合 AI Infra」「迷你 vLLM」。
14. 把 `cuda-kernel-academy` 说成当前仓库名（现名 `cuda-foundations`）。

---

## 5. 反问准备（最后 5 分钟，最多问 2 个）

选与岗位相关的，不要把没问到的仓库再讲一遍。

1. 组里 decode 路径的性能证据是 ncu 还是 end-to-end？我这边 ncu 不可用，用的是 kernel microbench。
2. 控制面和 kernel 仓的 ABI 谁拥有？你们如何防布局漂移？
3. 负结果（优化低于噪声）在组里是回滚还是留文档？
4. 若入职先接 vLLM/SGLang，期望我从调度不变量还是从 custom op 接入开始？

不要问：「你们用 C++ 还是 Python？」这类能搜到的问题。

---

## 6. 一次完整模拟的复盘模板

**元信息**

- 日期 / 录音或文字稿位置：
- 计时：总叙事 __s；tiny-llm __min；FA __min；serving __min；快问 __min；反问 __min
- 总分：__/72；红灯触发：__ 条（列出编号）

**逐题（只写失手）**

| 题 | 我说了什么 | 数字卡/QA 正确说法 | 下次改口 |
|----|------------|---------------------|----------|
|     |            |                     |          |

**数字口误清单**（专门一列，面试最致命）

- 把 6.087 说成了：
- 把 1.65× 说成了：
- 把 EOS 说成了：

**边界口误清单**

- 把未做的功能说成做了：
- 把 skip/未跑说成通过：

**下次只改三件事**（不超过三件）

1.
2.
3.
