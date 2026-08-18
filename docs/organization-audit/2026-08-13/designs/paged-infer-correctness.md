# `paged-infer` KV Cache 与 API 语义修复设计

## 1. 范围

本设计修复 CPU reference 的跨层 KV 污染，并收紧 sampling、streaming 和 API 语义。它不接入 CUDA，不实现复杂调度算法，也不把仓库升级成完整 OpenAI server。

## 2. Layer-aware KV Cache

### 2.1 数据模型

最小改动方案：

```rust
HashMap<(usize, BlockIdx), KvBlock>
```

其中 `usize` 是 layer index。也可以定义小型 `LayerBlockKey`，但不需要新的 cache hierarchy 或泛型抽象。

访问必须包含：

```text
(layer_idx, physical_block, offset, kv_head, dim)
```

### 2.2 调用变化

- layer 循环改为 `enumerate()`。
- 写 K/V 时使用 `(layer_idx, block_idx)`。
- `attention` 显式接收 `layer_idx`。
- Debug metrics 区分 physical scheduler blocks 与 materialized layer blocks，避免 `kv_cache.len()` 被误读为 scheduler block 数。

BlockPool 的物理 block ownership 仍只按 sequence/page 管理；layer 是内容维度，不需要让调度器为每层分配独立 logical block。

### 2.3 回归测试

必须先添加会在旧实现失败的测试：

1. 两层使用同一 physical block/offset 写不同 K/V，分别读取不相互覆盖。
2. 两 token、两 layer 的 incremental forward 与独立 dense cache evaluator 比较。
3. block 跨界位置，例如 `block_size-1` 与 `block_size`。
4. 物理 block 被释放复用后，新 sequence 不读到旧的可见位置。

独立 evaluator 使用每层 dense `Vec<Kv>`，不得复用被测 HashMap 访问函数。

## 3. Sampling 语义

仓库的核心不是 sampling。建议 v0.1 CPU reference 只支持 greedy，并让 API 诚实失败：

- `temperature == 0.0`。
- `top_p == 1.0`。
- 默认值同步改为这组参数。
- 收到其他值时返回 `UnsupportedGenerationMode` 或等价明确错误。

如果保留 mock executor 的不同能力，应由 backend capability 在 submit 阶段校验，不要让 CPU executor 在执行到最后才忽略参数。

未来接入真实 runtime 后，再由 backend 声明并实现 sampling capability。

## 4. Streaming decode

### 4.1 目标性质

对任意成功生成的 token 序列：

```text
所有 SSE text chunk 拼接 == 最终一次性 decode 文本
```

### 4.2 接口方向

给 tokenizer 增加“每请求 decoder state”概念，而不是继续调用单 token decode：

```text
TokenizerTrait::create_decoder() -> Box<dyn IncrementalDecoder>
IncrementalDecoder::push(token) -> Result<Option<String>>
IncrementalDecoder::finish() -> Result<Option<String>>
```

- `SimpleTokenizer` 可以直接逐字符输出。
- Hugging Face tokenizer adapter 应使用其 decoder 组件维护 byte/wordpiece 状态；若当前依赖版本不提供安全 streaming API，先缓冲 token 并只在完成时输出，不要伪造增量语义。

不要使用“每次 decode 全前缀然后盲目取字符串后缀”的实现；某些 decoder 会修改尾部空格或 byte 序列，已发送内容无法撤回。

### 4.3 状态归属

decoder state 按 request ID 存在 engine/server event 层，生命周期与 waiter 一致：完成、取消、失败或断开时删除。

## 5. Finish reason 与 Chat

内部 completion 状态应区分：

```text
Eos
Length
Cancelled
BackendError
TokenizerError
```

HTTP 映射至少做到：

- EOS -> `stop`
- max_tokens -> `length`
- error/cancel -> 错误事件或明确终止语义

Chat 在没有真实模型 template 前有两个选择：

1. 将文档改成“API-shaped chat endpoint”，明确简单拼接。
2. 引入配置化 template，并为固定模型添加 golden prompt。

推荐先做选择 1；真实 backend 接入时再做选择 2。

## 6. Backend 命名

P0 单独合入后，再做小型重命名：

- `GPUExecutorTrait` -> `ExecutionBackend` 或 `ModelExecutor`。
- `gpu_executor` 字段 -> `backend`。
- `create_default_gpu_executor` -> `create_default_backend`。

不要借重命名修改 Scheduler、Request 或 HTTP 大量结构。

## 7. 任务拆分

| 任务 | 内容 | 依赖 |
|---|---|---|
| PINF-001A | layer-aware key + isolation unit test | 无 |
| PINF-001B | dense evaluator + incremental equivalence | 001A |
| PINF-101 | greedy-only capability 与默认参数 | 无 |
| PINF-102 | incremental decoder state | 001A |
| PINF-103 | finish reason + chat compatibility wording | 101/102 |
| PINF-201 | backend 中性命名 | 001 完成后 |

## 8. 验收

- 全部现有 137 项测试继续通过。
- 新增两层 cache isolation 和 incremental oracle。
- 非 greedy 请求不再静默执行 greedy。
- streaming 等价性质覆盖 ASCII、中文/UTF-8、特殊 token 和失败。
- cancel/backend failure 后 decoder 与 KV 资源均被清理。
- crate、CLI、README 对默认 backend 的描述一致。

