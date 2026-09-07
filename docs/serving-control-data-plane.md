# Serving 控制面、数据面与批量执行边界

[打开当前 P2 交互式架构图](serving-control-data-plane-p2-batch-postprocess.html) ·
[查看当前图的 JSON 源规格](serving-control-data-plane-p2-batch-postprocess.architecture.json) ·
[查看已冻结的 device-greedy 快照](serving-control-data-plane-p2-greedy.html) ·
[查看已冻结的初版快照](serving-control-data-plane.html)

这张图服务于 `tiny-llm + paged-serving` 的系统/Serving 面试讲述。它把已经验证的
请求路径与尚未实现的批量计算优化刻意分开：控制面能调度多个序列，并不等于 CUDA
执行路径已经融合。

## 当前已验证的请求路径

1. `loadgen` 向 paged-serving 的 OpenAI 兼容端点发送请求，并以第一个非空文本 SSE
   chunk 计 TTFT；P2 将 Poisson `arrival_seed` 同时写入 `summary.json` 和
   `run_metadata.json`。
2. paged-serving 的 `InferenceEngine` / `Scheduler` 负责准入、连续批处理和 paged KV
   的 `block_tables`；它通过受测试的同进程 C ABI 调用 tiny-llm。
3. tiny-llm 的 `tinyllm_step` 接受多序列描述符，但在
   [`ffi.cpp`](https://github.com/open-infra-ai/tiny-llm/blob/master/src/ffi.cpp)
   中的 Transformer layer forward 仍按序列循环。正常请求（`logprobs_k == 0`）会在每个
   序列 layer forward 后把末层 hidden D2D 写入独立的 `[B, hidden]` GPU buffer；循环结束后
   批量执行 final RMSNorm、LM head 与 greedy argmax，最后一次 D2H 回传 `int[B]` token。
   请求 logprobs 时仍走主机完整 logits / top-k 路径，因而 ABI 布局与输出顺序均未改变。
4. HuggingFace 流式解码只在文本能够安全追加时发出片段，最终片段拼接必须等于一次性
   decode；该契约及 ABI 双源布局见
   [`cross-repo-contracts.md`](cross-repo-contracts.md)。

初版 P2 流式矩阵归档了 21 个 run、1344 条逐请求记录、模型 SHA-256、双仓 commit、
固定 Poisson seed、CSV、图表和限制说明：
[2026-09-04 结果包](https://github.com/open-infra-ai/paged-serving/tree/master/benchmarks/serving/results/2026-09-04-RTX3060Laptop-paged-serving-p2-streaming)。
其中若干 TTFT p95 重复未收敛，故它是历史边界证据而不是通用 SLO。
当前批量末端后处理已通过真实模型 device/host 逐 token 对照、分页/连续 KV 差分和
paged-serving feature e2e（含三并发文本与 llama.cpp 对照）。本地 C++ `ctest` 定义 198 项，
其中第二模型测试因未配置而跳过、其余无失败；这只是正确性证据。上述 21-run 矩阵早于该
改动，不能从它推导吞吐或 TTFT 改善。
[当前干净提交的 closed c=4 HTTP 功能 canary](https://github.com/open-infra-ai/paged-serving/tree/master/benchmarks/serving/results/2026-09-05-RTX3060Laptop-paged-serving-p2-batch-postprocess-canary)
另行归档了 4/4 成功、模型 SHA-256 与双仓 commit；它同样是可运行性证据，而非性能结果。
批量末端后处理后的 [2026-09-07 正式矩阵](https://github.com/open-infra-ai/paged-serving/tree/master/benchmarks/serving/results/2026-09-07-RTX3060Laptop-paged-serving-p2-batch-postprocess-streaming)
已在当前双仓 clean commit 上重新采集 21 个 run：closed-loop 全部成功，Poisson 0.64 / 1.28
req/s 分别有 9 / 74 个 429；closed c1/c4 通过 TTFT p95 与吞吐的 10% 重复波动检查，c2/c8
与三档 Poisson 仍未收敛。它是当前路径的边界证据；由于没有交替、配对基线，不能把它与
旧包的差异归因于末端后处理，更不能把它说成 fused layer batch 的吞吐扩展。

## 下一项 P2 性能工作

目标不是改 ABI 表面形状，而是在保持现有 `tinyllm_step` 输入/输出顺序和 greedy
对齐的条件下，逐层替换其内部逐序列实现：

1. 已完成：正常 greedy 的 final RMSNorm、LM head、argmax 与单次 batch token 回传；它们
   仅覆盖 layer forward 之后的末端输出阶段。
2. 已完成但未接入 FFI：RoPE 内部 CUDA API 已可读取 `[num_tokens]` device 绝对位置数组，
   非连续、非单调位置与 CPU half-split 参考逐元素对照；它只解除 ragged position 的原语
   限制，不代表 layer batch compute。
3. 为 ragged sequence、`block_tables`、prefill/decode 混合批建立 batch-aware workspace
   与可比较的逐 token oracle，并逐层替换当前的逐序列 Transformer layer forward。
4. 保持 `next_tokens[s]` 与可选 logprobs 的 ABI 顺序不变；若 ABI 必须变化，先更新
   live 契约、`ffi.h`、Rust 双源定义与两仓 CHANGELOG。
5. 每一步都重跑 tiny-llm 策略 1/2 差分、paged-serving feature e2e、真实 HTTP canary；
   只有 correctness 通过后，才采集新的独立结果包。

面试时应直接说出这条边界："调度器已经在构造 batch，正常 greedy 的末端 final RMSNorm、
LM head 与 argmax 也已经按 batch 在 GPU 执行并一次回传 token；但 FFI 的 Transformer layer
forward 仍逐序列执行。下一步是 ragged batch layer compute，而不是把这项末端输出优化包装
成 fused batch；是否有吞吐扩展必须靠同一 workload 的新原始结果证明。"
