# paged-infer · 10 分钟讲述稿

验证故事必须用 **3 并发 e2e** 与 **used+free==total**（[`../NUMBERS_CARD.md`](../NUMBERS_CARD.md) §6–7）。

## 0. 一句话定位

Rust 写的推理控制面：分页 KV 与连续批处理。

## 1. 2 分钟：做什么、边界、为什么这样切

IN：BlockPool / PageTable、continuous batching、水位线、HOL 修复、优先级、OpenAI API+SSE、属性测试、经 C ABI 调 tiny-llm（策略 1 默认）。

OUT：任何 CUDA kernel、模型加载、tokenizer 权威实现（HF tokenizer 适配除外）。计算在 tiny-llm。

为什么 Rust：调度与不变量适合类型系统和属性测试；和 C++ runtime 用 C ABI 解耦，而不是把调度写进 CUDA 工程。语言不是卖点，**边界**才是。

本次 freeze：默认 `cargo test` **218 passed**（**未**开 `tiny-llm` feature，e2e 0 运行）。3 并发证据仍是 `9c3700b`/`9c974d3`。开口要带这句。

## 2. 3 分钟：最难的实现细节

**准入三层 + HOL。**

状态机：Pending → Prefill → Decode → Completed，失败任意阶段释放 KV。

调度优先级：先 decode（保在途延迟），再高优先级 prefill，再 pending；同级 FCFS（`scheduler.rs` 头注释）。

内存：`used+free==total`。水位线：预估本次占用后利用率 ≤ threshold，且给 decode 增长留 reserve，否则不接新 prefill。

HOL：大 prefill 排在队头时，后面的小请求不能饿死。回归 `test_small_pending_request_not_blocked_by_large_one`：占满 batch 后，1-token 请求须在有限 step 内完成。

无抢占、无 swap：内存不够就拒新请求或失败，不把序列踢到 CPU。这是和 vLLM 的边界，主动讲。

## 3. 2 分钟：优化/调试故事（策略 1）

- Before：控制面有块表，后端仍是连续 KV（策略 2），叙事是假分页。
- 改动：ABI v2 `max_num_blocks` + 真实 `block_tables`（`9e8f6c7`）。Rust 布局守卫 `tiny_llm_config_layout_is_stable`。
- After：策略 1 默认；3 并发请求 1 的 24 个 token id（含 EOS 151645）与 llama.cpp greedy **全等**。请求 2 只断言前缀 `[17,10,17]` + EOS，因为 W8A16 vs Q4_K_M 把 `equals`/`is` 打成 16819 vs 374。
- 代价：未声称 serving 吞吐；无 chunked prefill；无前缀缓存。

## 4. 2 分钟：验证方法（规定）

1. **3 并发 e2e**：`qwen2_three_concurrent_paged_requests_match_llama_cpp`。请求 1 全序列；请求 2 诚实分歧；结束后 `active_sequences==0`、利用率回 0。
2. **资源守恒**：`prop_block_count_invariant`；取消/失败归还 `prop_resources_reclaimed_after_cancel_and_failure`。
3. HTTP：`server_integration` 37 项，含 `/metrics` 的 `paged_*`、SSE `data: [DONE]`。测的是 CPU 参考后端契约，不是 GPU QPS。
4. NaN / Unicode stop：显式回归，避免静默坏掉内存保护和 UTF-8 切分。

## 5. 1 分钟：短板与下一步

短板：无抢占；token 级 SSE 在 HF tokenizer 上是「结束时一整块」（文档已降级）；默认 CI 不含 GPU e2e。下一步：解冻才做 paged 去 gather 或 Graphs；或把练习变成 vLLM 小 PR。

## 6. 追问清单

1. PagedAttention 解决什么碎片？ → 预留 max_seq 的连续 KV 浪费；块化按需。
2. block table 存在哪侧？ → 控制面持有映射；策略 1 上传给 tiny-llm。
3. CB 和 static batching 差在哪？ → 请求异步进出，decode/prefill 混批。
4. 为什么 decode 优先？ → 保 TTFT 之后的 TPOT，避免新 prefill 插队饿死在途。
5. HOL 是什么？ → 队头大请求挡住后面能跑的小请求。
6. 没有抢占会怎样？ → 高优先级来了也不能踢人；内存满则拒。
7. 3 并发等于生产并发？ → 不等于；是正确性 fixture。
8. 为什么 Rust？ → 见 cross-cutting。
9. 策略 2 还在吗？ → `PAGED_INFER_TINY_LLM_STRATEGY=2`。
10. 本次 freeze 跑过 e2e 吗？ → 没有；要 `--features tiny-llm`。
