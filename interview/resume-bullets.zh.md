# 简历条目（中文）

数字只来自 [`NUMBERS_CARD.md`](NUMBERS_CARD.md)。每条汉字 ≤40（专有名词与数字另计），尾注 `→ E<n>` 对应 [`EVIDENCE_MATRIX.md`](EVIDENCE_MATRIX.md)。面试被追问时按编号查证。

---

## 总览（3）

1. 搭建 CUDA→Triton→FA→runtime→serving 四层五仓可验证学习链。→ E30
2. 转置 M==1 GEMM，TPOT 24.348→6.087 ms，相对 llama.cpp 收到 1.65×。→ E15
3. Rust 控制面 3 并发分页请求与 llama.cpp greedy 对齐，量化分歧如实记录。→ E21

## cuda-foundations（3）

1. SGEMM 阶梯 naive→WMMA：0.58/0.92/0.66/0.68/1.09 TFLOPS，负优化留表。→ E1
2. 04-inference-engine 降级教学预览，runtime 禁止依赖教学仓。→ E26
3. 仓库改名 cuda-foundations，五仓源码旧 slug 0 命中。→ E27

## triton-fused-ops（3）

1. RMSNorm+RoPE / SwiGLU / FA 前向各有独立 reference 差分。→ E3
2. 修复 TRIT-001：RoPE 改为 Llama/Qwen half-split concat 契约。→ E4
3. `torch.ops.triton_ops.*` 注册三 op；compile smoke 如实 skip。→ E5

## cuflash-attn（3）

1. FlashAttention 前后向多精度差分；B×H>65535 展平 grid 并回归。→ E7
2. causal 边界块跳过实测 ±2%，文档记为低于噪声的负结果。→ E8
3. 实现 decode 用 FlashDecoding Split-KV，有 CPU 参考与块数不变量。→ E9

## tiny-llm（3）

1. 一条命令从 GGUF 跑通真实 Qwen2.5-0.5B Instruct 生成。→ E10
2. tokenizer 与 HF 逐 id 对齐：30 例共 417 token。→ E11
3. CUDA Graphs 默认捕获 decode，on/off greedy 逐 token 一致。→ E16

## paged-infer（3）

1. C ABI v2 九整型布局，Rust `sizeof==36` 守卫防漂移。→ E19
2. 属性测试锁定 `used+free==total`，取消与失败必须归还块。→ E23
3. OpenAI 兼容 API/SSE/`paged_*` metrics：server 集成 37 项。→ E25

（3 并发 token 对齐已在总览第 3 条，避免重复数字。）

---

## 无 GPU 面试环境的备选说法

线上面试不能跑 demo 时，按这条链证明，不要假装本机能远程出数。

| 想证明 | 不依赖 GPU 的说法 | 指向 |
|--------|-------------------|------|
| 工程闭环 | 「六仓 GitHub 可见；面试包在 aicl-lab/interview；五仓 `phase-2-e`。」 | E30、本目录 |
| CUDA 阶梯 | 「打开 benchmarks 页：0.58→1.09，bank-conflict-free 更慢是刻意留下的。」 | E1 |
| Triton 契约 | 「TRIT-001 是排列 bug；torch.library schema 可在 README 指给面试官。」 | E4、E5 |
| FA 负结果 | 「causal-boundary-skip.md 写明 ±2% 低于 10% 阈值。」 | E8 |
| Runtime | 「freeze 审计：tiny-llm 174 passed；数字卡复制 decode-optimization 表。」 | 数字卡 §1、FREEZE_AUDIT |
| Serving 调度 | 「默认 `cargo test` 218 passed，不含 GPU e2e；不变量不需要 GPU。」 | E23、数字卡 §6 |
| Token 对齐 | 「e2e 源码里写死了 24 个 id 与 is/equals 分歧；T1 未重跑 feature。」 | E18、E21 |
| 禁止临场编 | 「3030/5118 无归档；ncu 不可用；skip 不当 pass。」 | E22、E28 |

口述模板：「我无法在这台机器上复现 GPU 数字；请打开 NUMBERS_CARD 第 N 行，commit 与命令都在。」
