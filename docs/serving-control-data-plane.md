# Serving 控制面、数据面与批量执行边界

[打开当前 P2 交互式架构图](serving-control-data-plane-p2-greedy.html) ·
[查看当前图的 JSON 源规格](serving-control-data-plane-p2-greedy.architecture.json) ·
[查看已冻结的上一版快照](serving-control-data-plane.html)

这张图服务于 `tiny-llm + paged-serving` 的系统/Serving 面试讲述。它把已经验证的
请求路径与尚未实现的批量计算优化刻意分开：控制面能调度多个序列，并不等于 CUDA
执行路径已经融合。

## 当前已验证的请求路径

1. `loadgen` 向 paged-serving 的 OpenAI 兼容端点发送请求，并以第一个非空文本 SSE
   chunk 计 TTFT；P2 将 Poisson `arrival_seed` 同时写入 `summary.json` 和
   `run_metadata.json`。
2. paged-serving 的 `InferenceEngine` / `Scheduler` 负责准入、连续批处理和 paged KV
   的 `block_tables`；它通过受测试的同进程 C ABI 调用 tiny-llm。
3. tiny-llm 的 `tinyllm_step` 接受多序列描述符，但当前在
   [`ffi.cpp`](https://github.com/open-infra-ai/tiny-llm/blob/master/src/ffi.cpp#L345)
   中仍按序列循环。正常请求（`logprobs_k == 0`）的
   [`sample_from_hidden_device`](https://github.com/open-infra-ai/tiny-llm/blob/master/src/ffi.cpp#L104)
   会在 GPU 完成 greedy argmax；各序列的 token 写入独立 device slot，整个 batch 在
   `tinyllm_step` 末尾一次 D2H 回传并同步。请求 logprobs 时仍走主机完整 logits / top-k
   路径，因而 ABI 布局与输出顺序均未改变。
4. HuggingFace 流式解码只在文本能够安全追加时发出片段，最终片段拼接必须等于一次性
   decode；该契约及 ABI 双源布局见
   [`cross-repo-contracts.md`](cross-repo-contracts.md)。

P2 当前流式矩阵归档了 21 个 run、1344 条逐请求记录、模型 SHA-256、双仓 commit、
固定 Poisson seed、CSV、图表和限制说明：
[完整结果包](https://github.com/open-infra-ai/paged-serving/tree/master/benchmarks/serving/results/2026-09-04-RTX3060Laptop-paged-serving-p2-streaming)。
其中若干 TTFT p95 重复未收敛，故它是边界证据而不是通用 SLO。
本次 device-greedy 改动已通过真实模型 device/host 逐 token 对照和 paged-serving feature
e2e，但尚未重新采集 serving 矩阵，不能从它推导吞吐或 TTFT 改善。

## 下一项 P2 性能工作

目标不是改 ABI 表面形状，而是在保持现有 `tinyllm_step` 输入/输出顺序和 greedy
对齐的条件下，逐层替换其内部逐序列实现：

1. 先为 ragged sequence、`block_tables`、prefill/decode 混合批建立 batch-aware
   workspace 与可比较的逐 token oracle。
2. 将 final norm 与 LM head 改为真正的批量设备侧路径；greedy argmax 和单次 batch
   token 回传已完成，但当前 layer forward 仍逐序列提交。
3. 保持 `next_tokens[s]` 与可选 logprobs 的 ABI 顺序不变；若 ABI 必须变化，先更新
   live 契约、`ffi.h`、Rust 双源定义与两仓 CHANGELOG。
4. 每一步都重跑 tiny-llm 策略 1/2 差分、paged-serving feature e2e、真实 HTTP canary；
   只有 correctness 通过后，才采集新的独立结果包。

面试时应直接说出这条边界："调度器已经在构造 batch，正常 greedy 也已从每序列整词表
D2H 收敛为一次 batch token 回传；但 FFI 的 layer forward 仍逐序列执行。下一步是
ragged batch compute，而不是把这项输出路径优化包装成 fused batch；是否有吞吐扩展必须靠
同一 workload 的新原始结果证明。"
