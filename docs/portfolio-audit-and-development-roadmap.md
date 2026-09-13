# AI Infra 作品集审计与技术开发路线

> 审计日期：2026-09-13  
> 范围：`cuda-foundations`、`cuflash`、`trifuse`、`tiny-llm`、`paged-serving`、
> `kvtier` 与本组织总仓。  
> 本文是跨仓的稳定技术判断与开发优先级；具体性能数字仍以各技术仓中的 clean commit、
> 原始数据、环境和复现命令为准。

## 1. 结论

这组项目已经明显超过“只跟教程写 demo”的阶段，足以支持以下方向的初级到初中级面试：

- LLM Inference Runtime；
- CUDA / Triton Kernel；
- 单机 AI Serving；
- GPU Performance Engineering。

但它们目前证明的是：

1. 能把 kernel、Runtime、KV Cache、调度和 HTTP/SSE 串成真实闭环；
2. 能建立正确性测试、资源不变量和基础 benchmark；
3. 能诚实记录负结果与适用边界。

它们还不能证明：

- 生产级 vLLM / TensorRT-LLM 等价实现；
- FA2/FA3、Hopper TMA/WGMMA 或 Tensor Core INT8 的成熟实现；
- 真正的 direct paged attention 或完整 ragged batch layer execution；
- 多机多卡、Tensor Parallel、故障恢复等分布式经验；
- 跨 GPU、跨模型、跨引擎都成立的性能优势；
- 高级或 Staff 级 AI Infra 资历。

因此，最准确的作品集定位是：

> **可审计的单机 AI Infra 学习型系统：从 CUDA/Triton 算子、真实权重 Runtime 到 Rust
> Serving 控制面，具备深入面试价值，但尚未达到生产级或分布式系统标准。**

## 2. 作品集不应按七个平级项目展示

### 旗舰一：单 GPU LLM Runtime + Serving

组合：

- [`tiny-llm`](https://github.com/open-infra-ai/tiny-llm)：数据面；
- [`paged-serving`](https://github.com/open-infra-ai/paged-serving)：控制面。

这条主线证明：

- GGUF、量化、Tokenizer、Transformer、KV Cache、采样和 CUDA Graph；
- C ABI、调度状态机、Paged KV block accounting、背压、取消和 SSE；
- 从请求进入到 token 流出的端到端状态变化；
- Runtime 正确性与 Serving 资源生命周期。

面试时必须主动说明：

- 当前为单模型、单 GPU；
- Paged KV 策略仍会 scatter/gather 到连续 scratch；
- 调度层有 continuous batching，但核心 Transformer layer 仍主要逐序列执行；
- 没有分布式执行和生产 SLO。

### 旗舰二：CUDA / Triton Attention 与融合算子

组合：

- [`cuflash`](https://github.com/open-infra-ai/cuflash)：CUDA C++ 深挖；
- [`trifuse`](https://github.com/open-infra-ai/trifuse)：Triton 与 PyTorch 集成对照；
- [`cuda-foundations`](https://github.com/open-infra-ai/cuda-foundations)：基础学习档案。

这条主线证明：

- online softmax、causal mask、tiling、WMMA 和 Split-KV；
- CUDA 与 Triton 的线程/块表达差异；
- 数值稳定性、边界 shape、动态 shape 和差分测试；
- custom op 与 `torch.library` 接口意识。

面试时不得将这条主线包装成：

- 生产级 FlashAttention-2/3；
- 已完成 Hopper TMA/WGMMA/FP8；
- 已在所有 shape 上超过 PyTorch、cuBLAS 或官方 FlashAttention；
- 已接入旗舰 Runtime 的生产请求路径。

### 辅助证据

- [`kvtier`](https://github.com/open-infra-ai/kvtier)：KV offload 源码研究、实验编排和
  telemetry 门禁；在产生真实 GPU 结果和至少一项 runtime 优化前不作为简历主项目。
- 本仓：状态、证据与跨仓契约入口，不作为独立技术项目计数。

## 3. 七仓审计摘要

| 仓库 | 当前价值 | 主要硬伤 | 建议 |
|------|----------|----------|------|
| `tiny-llm` | 最强数据面证据；真实覆盖模型加载、W8A16、Transformer、KV、Graph、C ABI | GPU/真实模型测试未成为稳定门禁；无完整 batch、direct paged attention、多 GPU 和可消费 profiler 报告 | 旗舰主打 |
| `paged-serving` | 调度、资源不变量、背压、取消、SSE、metrics 与负载实验是真实实现 | 调度 batch 不等于 fused GPU batch；缺抢占、prefix cache、chunked prefill、分布式与稳定 SLO | 与 `tiny-llm` 联合主打 |
| `cuflash` | online-softmax forward、WMMA、backward、Split-KV 有源码和追问价值 | GPU CI 与 profiler 证据不足；decode workspace 的并发/stream 安全需强化；不是 FA2/FA3 | Kernel 岗第二旗舰 |
| `trifuse` | Triton kernel 和 `torch.library` 集成真实存在 | Gated MLP GEMM/FLOPs 与计时口径需统一；缺公平 baseline、raw artifact 和 profiler | 与 `cuflash` 合并展示 |
| `cuda-foundations` | 适合证明 CUDA 基础、优化阶梯和负结果纪律 | 高级术语中存在 placeholder/fallback；系统深度不足 | 学习档案，不占主项目位 |
| `kvtier` | 实验污染控制、上游阅读和 KV offload 方法论有价值 | 缺真实 GPU 结果、token oracle、核心实现改动和闭环优化 | 研究辅助，完成结果后再升级 |
| 总仓 | 证据治理与跨仓导航 | 本身没有新增数据面实现 | 只作入口 |

## 4. 当前能力矩阵

| 能力 | 当前强度 | 已有证据 | 下一层关键缺口 |
|------|---------:|----------|----------------|
| CUDA / Triton Kernel | 7/10 | online softmax、WMMA、Split-KV、W8A16、融合算子、custom op | 架构特化、direct paged attention、持续 GPU 门禁 |
| GPU 性能分析 | 5/10 | CUDA Event、warmup、A/B、raw JSONL、负结果 | ncu/nsys、roofline、stall/occupancy/带宽归因 |
| LLM Runtime | 7/10 | GGUF → Transformer → KV → sampling → Graph → C ABI | 通用模型、真正 layer batching、多 GPU、成熟 workspace 合约 |
| KV Cache | 6/10 | 连续/分页 KV、block pool/table、资源回收 | kernel 直接读页、prefix cache、preemption、并发压力 |
| Serving | 6/10 | 调度、准入、429、取消、SSE、metrics、loadgen | chunked prefill、稳定 SLO、多副本、完整观测 |
| 分布式系统 | 1/10 | 主要为理论和接口边界 | NCCL、TP/PP、路由、故障恢复 |
| 测试与工程 | 6/10 | reference、属性/边界测试、资源不变量、CI | GPU 强制门禁、sanitizer/fuzz、兼容矩阵 |
| 可复现 benchmark | 5/10 | 部分项目有环境、commit、模型 hash 和 raw data | 统一 schema、公平 baseline、跨架构与统计收敛 |

## 5. 开发原则

后续开发不再以“增加仓库、代码量或技术名词”为目标。每个改动必须形成：

```text
问题 → 正确性基线 → 性能/系统假设 → 单变量改动 → 原始结果 → 结论与边界
```

任何性能任务必须先回答：

1. 基线是谁？
2. 输入、dtype、layout、模型和硬件是什么？
3. 什么结果会推翻假设？
4. 如何确认输出没有错？
5. 原始结果存在哪里？
6. 哪些 shape 或负载更慢？

## 6. P0：先修可信度和自动验证

P0 完成前，不应继续扩大功能面。

### P0.1 统一术语、源码和性能口径

重点：

- 将 `cuda-foundations` 中的 TMA、Cluster、FP8、Winograd、PTX MMA
  明确区分为真实实现、兼容 fallback、演示或 roadmap；
- 统一 `trifuse` Gated MLP 的实际 GEMM 数、FLOPs、输出语义和计时统计；
- 禁止把 `tiny-llm` W8A16 写成 Tensor Core INT8；
- 禁止把 online-softmax kernel 直接写成生产 FlashAttention；
- 禁止把 Paged KV block accounting 写成 direct PagedAttention；
- 禁止把调度层 batch 写成完整 fused GPU batch execution。

验收：

- 完成特性表中没有 placeholder/fallback 被标为完成；
- 所有性能数字都有 raw artifact 链接；
- FLOPs/bytes 由代码按真实计算图生成，并有单测；
- README、源码、测试与总仓状态没有相互冲突的声明。

### P0.2 建立真实 GPU correctness 门禁

建议分层：

- PR：单 GPU smoke，控制在可接受时长；
- nightly：真实模型、更多 shape、Compute Sanitizer；
- release：至少两个 GPU 架构的完整矩阵。

验收：

- 缺 GPU、模型或关键依赖时明确失败，不允许“绿色但全部 skip”；
- 连续 20 次主干 GPU workflow 无非预期 skip；
- artifact 包含 commit、dirty 状态、GPU、driver、CUDA、编译器、模型 hash 和日志；
- 连续/分页 KV、Graph on/off、三并发后端路径在固定 token 集上 0 非预期 mismatch。

### P0.3 修复构建与环境契约

验收：

- `paged-serving` 的 MSRV 声明、lockfile 与 CI 一致；
- C++/CUDA、Rust、Python/Torch/Triton 至少固定一套可重建环境；
- clean clone 到 build、test、smoke benchmark 有单一入口；
- 所有可选测试明确说明未运行原因，不将 skip 计为通过。

### P0.4 建立 ownership 证据

精选至少五个提交：

1. correctness bug；
2. 资源生命周期 bug；
3. 可归因性能优化；
4. 负结果；
5. 跨仓接口或契约。

每个提交记录：

- 问题如何发现；
- 原假设和替代方案；
- 修改的关键代码；
- 如何验证；
- 哪些结果不符合预期；
- AI 工具参与了什么，本人做了哪些判断。

## 7. P1：只做一个深改造，再建立性能闭环

### P1.1 默认首选：direct paged attention

目标：attention kernel 直接消费 block table 和页池，不再将完整 K/V gather 到连续
scratch 后计算。

正确性验收：

- 覆盖 batch 1/2/4/8；
- context 128/512/2048；
- block 边界、tail、GQA/MQA、取消和复用；
- 至少 100 个固定 prompt 与参考路径 0 token mismatch；
- Compute Sanitizer `memcheck`、`racecheck`、`initcheck`、`synccheck` 按适用范围通过。

性能验收：

- 在同一 clean commit、硬件、模型和 workload 下与当前 gather 路径配对 A/B；
- 代表性 decode 矩阵 p50 TPOT 改善目标为至少 20%；
- 纳入正式报告的主要 shape 回退不超过 5%，否则保留负结果并解释；
- 报告显存访问、workspace、kernel launch 和 context length 的变化。

### P1.2 Kernel 岗替代主线：`cuflash` 可重入 workspace

若主投 CUDA Kernel，可先于 direct paged attention 完成：

- 移除进程级 static decode scratch；
- 使用调用方 workspace 或 stream-ordered allocator；
- 定义容量、对齐、别名、stream 和 Graph capture 契约；
- 双 stream 并发、resize 交错、超大 `B×H` 和错误传播测试。

验收：

- 双 stream 压力测试 10,000 次无错误；
- Compute Sanitizer 通过；
- CUDA Graph capture/replay 通过；
- API 能在 launch 失败时返回可定位错误；
- profiler 证明 allocator/workspace 改造未引入主要 shape >5% 回退。

### P1.3 Nsight 驱动的性能分析

每次先用 Nsight Systems 判断“时间花在哪里”，再用 Nsight Compute 判断
“某个 kernel 为什么慢”。

至少分析五个核心点：

- `tiny-llm` M=1 GEMM / LM head；
- attention；
- paged gather 或 direct paged attention；
- CUDA Graph decode；
- 一个 Triton 或 `cuflash` kernel。

每个报告至少包含：

- timeline、kernel launch 与 memcpy；
- registers/thread、achieved occupancy；
- DRAM/L2 throughput；
- 主要 warp stall；
- SM/Tensor utilization；
- before/after 和一个可证伪假设。

验收：

- 保存可打开的 `.nsys-rep` / `.ncu-rep`；
- 至少一项主要 shape p50 改善 10% 以上；
- 其他主要 shape 回退不超过 5%；
- 若优化无效，保留报告而不是删除实验。

### P1.4 公平外部 baseline

Runtime / Serving：

- `tiny-llm + paged-serving`；
- llama.cpp / llama-server；
- vLLM。

Kernel：

- 自实现；
- PyTorch / cuBLAS / SDPA；
- 条件允许时的官方 FlashAttention。

验收：

- 同 GPU、模型、量化或醒目标注量化差异；
- 同 prompt、输出长度、采样和并发；
- 每配置至少三个独立进程；
- 报告 p50/p95、范围或 IQR、失败数和 CV；
- CV >10% 的结果不写成稳定结论；
- 原始逐请求/逐 iteration 数据可重新生成全部图表。

## 8. P2：按目标岗位补一条系统证据

### Serving 方向

从以下三项中只选一项做完整闭环：

- chunked prefill；
- preemption；
- prefix cache。

验收：

- property test 保持资源不变量；
- 长短请求混合负载下短请求 p95 TTFT 改善目标至少 20%；
- 成功率不下降；
- 取消、超时和 backend failure 后 KV block 泄漏为 0。

### Distributed 方向

在真实硬件上二选一：

1. 双 GPU Tensor Parallel：权重分片、NCCL collective、正确性和通信/计算时间线；
2. 双副本 Serving：router、过载转移、graceful drain 和 kill-one-replica 故障实验。

在完成前，不在简历中使用“distributed inference”。

最低验收：

- TP：至少三类消息规模/并发，报告 bus bandwidth、algorithm bandwidth 和正确性；
- 双副本：1,000 请求中随机终止一个副本，已完成请求零丢失，其余流量在 5 秒内恢复；
- 保存拓扑、日志、原始负载、错误分类和故障时间线。

### 生产可观测性

每个请求关联：

- queue wait；
- prefill；
- decode step / ITL；
- batch size；
- KV utilization；
- TTFT / TPOT；
- 完成、取消或失败原因；
- GPU clock、power、temperature 和 utilization。

所有指标与 trace 使用 `run_id` / `request_id` 关联。

## 9. 推荐执行顺序

### 里程碑 1：可信度清理

- 修正文档、术语、测试数和性能口径；
- 建立两个旗舰项目的 known limitations；
- 建立 selected commits / ownership 页面。

退出条件：0 个 placeholder 被写成完成能力；0 个无法回溯的性能数字。

### 里程碑 2：GPU 门禁

- PR smoke；
- nightly GPU correctness；
- release 跨架构矩阵；
- 固定模型和 artifact schema。

退出条件：至少 20 次连续成功，0 非预期 skip。

### 里程碑 3：首份 profiler 包

- nsys 定位端到端瓶颈；
- ncu 分析五个 kernel；
- 保存原始报告和 counter 解释。

退出条件：能用证据解释瓶颈，而不是只报总耗时。

### 里程碑 4：一个深改造

默认做 direct paged attention；Kernel 岗优先时做 `cuflash` workspace/stream 安全。

退出条件：正确性、Sanitizer、性能、回退和边界全部有记录。

### 里程碑 5：公平 baseline

- Runtime 与 llama-server / vLLM；
- Kernel 与 PyTorch / cuBLAS / SDPA；
- 至少三个独立进程和原始数据。

退出条件：所有图可由脚本从 raw data 重新生成。

### 里程碑 6：按岗位分叉

- Kernel / Runtime：跨架构、ragged layer batch 或更深 profiler；
- Serving：chunked prefill / preemption / prefix cache；
- Distributed：双 GPU TP 或双副本故障实验。

## 10. 项目完成定义

一个任务只有同时满足以下条件才算完成：

1. clean clone 可构建；
2. 有独立正确性 reference；
3. 正常、边界、失败和资源回收路径有测试；
4. 性能测量固定环境、输入、warmup 和统计方法；
5. raw data、summary、图和 profiler 可追溯；
6. README 写出已完成、未完成和适用边界；
7. 能在十分钟内不看 README 解释设计、瓶颈、失败实验和下一步；
8. 简历 bullet 中的每个数字都能定位到 source、test、artifact 和 commit。

## 11. 简历使用门槛

当前可以写：

- 单 GPU、学习型 Runtime / Serving；
- GGUF、W8A16、KV、CUDA Graph、C ABI、调度、SSE；
- CUDA/Triton online softmax、WMMA、Split-KV 和 custom op；
- 在明确硬件、模型和 workload 下的已归档数字。

完成对应证据前不能写：

- production-grade；
- distributed inference；
- direct PagedAttention；
- FA2/FA3；
- Tensor Core INT8；
- Hopper TMA/WGMMA；
- 全面超过 vLLM、llama.cpp、PyTorch 或官方 FlashAttention。

最强的面试表达不是隐藏短板，而是按以下顺序回答：

1. 我实际实现了什么；
2. 我没有实现什么；
3. 当前证据能证明什么；
4. 哪些结果不能外推；
5. 下一步如何用 correctness、profiler 和负载实验验证。
