# `paged-infer` 仓库审计

## 1. 定位判断

`paged-infer` 适合作为 Serving 控制面和资源状态机练习仓库，而不是模型运行时。README 与 ROADMAP 已明确这种边界，保持低优先级是正确决策。

当前实现不是纯 mock：默认构造路径实际创建 `CpuReferenceExecutor`。因此最准确的表述是“具有随机小模型的 CPU reference backend，同时保留测试用 mock”，而不是统一称为 mock 或 GPU executor。

## 2. 当前架构

```text
HTTP / library request
  -> tokenizer
  -> Scheduler
       pending -> prefill -> decode -> completed/failed
       BlockPool + PageTable
  -> BatchExecutionPipeline
  -> CpuReferenceExecutor / Mock executor
  -> generated token
  -> decoder / SSE / final response
```

关键代码：

- 默认 engine 组装：[`src/engine.rs`](../../../../paged-infer/src/engine.rs#L116)
- 默认 backend 工厂：[`src/gpu_executor.rs`](../../../../paged-infer/src/gpu_executor.rs#L58)
- decode-first 调度：[`src/scheduler.rs`](../../../../paged-infer/src/scheduler.rs#L100)
- 批次描述与执行：[`src/execution_pipeline.rs`](../../../../paged-infer/src/execution_pipeline.rs#L90)
- 单 owner async engine loop：[`src/server.rs`](../../../../paged-infer/src/server.rs#L361)

## 3. 值得保留的资产

### 3.1 调度状态机清楚

- decode 优先，降低已进入系统请求的 inter-token latency。
- 受 `max_batch_size` 和 `max_total_tokens` 双重预算约束。
- pending、prefill、decode、completed/failed 状态分离。
- 内存不足和 backend 失败能进入失败回收路径。

这些设计比 CPU 小模型本身更有作品价值。

### 3.2 分页资源管理具有真实不变量

BlockPool/PageTable、扩容、释放、取消和 OOM 不是空接口。现有属性测试证明了部分资源守恒性质，适合后续扩展为 model checking 风格的状态序列测试。

### 3.3 Server 所有权模型合理

后台 loop 是 engine 的唯一 owner；请求通过 channel 提交，handler 断开后取消请求。循环主动 `yield_now`，避免无 await 的生成循环饿死 SSE task。这是一处值得保留并在面试中解释的工程细节。

### 3.4 测试基础较完整

本次执行结果：

- 92 个 unit tests。
- 12 个 integration tests。
- 16 个 server integration tests。
- 17 个 doc tests。
- 共 137 个通过。

这些测试对控制面有价值，但没有覆盖 CPU Transformer 的独立数值正确性。

## 4. 已确认问题

### PINF-001 / P0：KV Cache 缺少 layer 维度

`CpuReferenceExecutor` 使用：

```rust
kv_cache: HashMap<BlockIdx, KvBlock>
```

见 [`src/cpu_executor.rs`](../../../../paged-infer/src/cpu_executor.rs#L157)。

每个 token 在 layer 循环中，用同一个 `BlockIdx + offset` 写入 K/V：[`src/cpu_executor.rs`](../../../../paged-infer/src/cpu_executor.rs#L209)。后一个 layer 会覆盖前一个 layer；下一 token 的 layer 0 随后可能读取上一 token 最后一个 layer 的 K/V。

影响：

- 分页和资源计数仍然正确，所以当前属性测试不会失败。
- CPU reference 不能作为真实 backend 的数值 oracle。
- “多层 Transformer”这一实现声明不成立。

修复方向见 [设计文档](../designs/paged-infer-correctness.md)。

### PINF-101 / P1：采样参数不生效

API 接受并验证 `temperature` 和 `top_p`：[`src/types/request.rs`](../../../../paged-infer/src/types/request.rs#L17)，CPU executor 最终始终 argmax：[`src/cpu_executor.rs`](../../../../paged-infer/src/cpu_executor.rs#L254)。

两种可接受处理：

- 实现确定种子下的 temperature/top-p sampling。
- 在 reference backend 只支持 greedy 时，拒绝非 greedy 配置并返回明确错误。

静默忽略不可接受。

### PINF-102 / P1：流式 decode 无状态

每个生成 token 被单独传给 tokenizer decode：[`src/engine.rs`](../../../../paged-infer/src/engine.rs#L318)。Byte-level BPE、byte fallback、不完整 UTF-8 或跨 token normalization 可能导致 chunk 丢失或乱码。

必须验证：

```text
concat(stream chunks) == decode(all generated tokens)
```

### PINF-103 / P1：OpenAI 兼容范围表述过宽

- Chat message 被拼成 `role: content` 文本，而不是模型 chat template：[`src/server.rs`](../../../../paged-infer/src/server.rs#L723)。
- 非流式响应的 `finish_reason` 固定为 `stop`，不能区分长度上限等状态：[`src/server.rs`](../../../../paged-infer/src/server.rs#L543)。
- 当前模型和 tokenizer 本身也是随机 reference。

建议在真实 backend 之前描述为“OpenAI-shaped endpoints”，或在 README 列出明确兼容子集。

### PINF-201 / P2：backend 命名绑定 GPU

`GPUExecutorTrait` 和 `gpu_executor` 字段实际可以承载 CPU 与 mock。继续沿用会在未来接入真实 backend 时制造错误抽象。

建议后续改为 `ExecutionBackend`/`ModelExecutor`，但不要与 P0 修复同时做大范围重命名。

### PINF-202 / P2：文档漂移

- README 称默认是 CPU reference。
- crate 顶部文档和 CLI about 仍称 mock：[`src/lib.rs`](../../../../paged-infer/src/lib.rs#L1)、[`src/main.rs`](../../../../paged-infer/src/main.rs#L8)。
- ROADMAP 又写 CPU reference + Mock GPU executor。

应给出一个权威状态表，区分默认 backend 和测试 double。

## 5. 有意边界，不作为 bug

- 没有真实 CUDA kernel。
- 没有 chunked prefill、prefix cache、preemption、LoRA、多 GPU。
- 调度策略简单，缺少复杂公平性/SLO。
- CPU model 使用随机小权重，不加载真实模型。

这些边界与仓库当前定位一致。只有当文档声称生产能力，或准备接入真实 backend 时才升级为缺陷。

## 6. 测试缺口

- 两层以上 incremental decode 与 full recompute 的 logits/token 差分。
- 每层 cache isolation。
- 流式 decode 与一次性 decode 等价。
- EOS、length、cancel、backend error 的 finish reason。
- 多请求取消/失败交错下最终资源归还。
- 非 greedy 参数不能被静默忽略。

## 7. 推荐顺序

1. 修复 PINF-001，并补独立 full-recompute oracle。
2. 修正 backend 文档和命名语义。
3. 选择采样参数策略：实现或拒绝。
4. 增加 stateful streaming decoder。
5. 完善 finish reason 与 API 兼容说明。
6. 在 `tiny-llm` L3/L4 成立前，不投入真实后端集成。

## 8. 成熟度判断

| 维度 | 判断 |
|---|---|
| 控制面架构 | 良好学习样例 |
| 资源状态机 | 有实质内容 |
| 计算正确性 | 当前不成立 |
| API 兼容性 | 部分兼容 |
| CI/测试 | CPU 控制面较强 |
| 生产可用性 | 不适用 |

