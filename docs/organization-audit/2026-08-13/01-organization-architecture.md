# 组织架构与项目组合审计

## 1. 现有项目组合

当前五仓不是五个相互竞争的推理引擎，而是五种不同学习角色：

| 层次 | 权威仓库 | 应负责 | 不应负责 |
|---|---|---|---|
| CUDA 基础 | `cuda-kernel-academy` | 优化概念、示例、通用工程入门 | 生产运行时、权威 FlashAttention 性能 |
| Triton 算子 | `triton-fused-ops` | 小型 fused op、独立 reference、Triton 验证方法 | 完整模型加载、Serving |
| CUDA attention | `cuflash-attn` | dense FlashAttention 前后向、数值与性能深挖 | 通用 LLM runtime、调度系统 |
| 模型 runtime | `tiny-llm` | GGUF、tokenizer、Transformer、KV、sampling | 多租户 Serving 控制面 |
| Serving | `paged-infer` | 分页资源、continuous batching、HTTP/SSE | 重复实现模型 loader 和 kernel 教程 |

这个划分比把所有内容塞进单一 monorepo 更适合作品集：每个仓库都有可讲述的中心问题，读者也能按层次进入。

## 2. 现有优点

### 2.1 边界声明总体诚实

- `paged-infer` 明确是控制面脚手架并保持低优先级。
- `tiny-llm` 明确真实生成和性能基准尚未完成。
- `cuflash-attn` 明确是学习、审计和轻量集成，而非框架替代品。
- `triton-fused-ops` 主动删除了名不副实的 FP8 路径。
- Academy 明确 CUDA 13 目录主要是占位和实验实现。

这类边界说明应继续保留。审计中发现的错误不应通过模糊措辞隐藏，而应通过测试和状态表体现。

### 2.2 技术覆盖具有连贯性

现有仓库覆盖了 AI Inference 面试和实践中重要的几条线：

- GPU memory hierarchy、GEMM、online softmax、融合算子。
- GGUF、量化权重、tokenizer、Transformer runtime。
- KV Cache、分页分配、continuous batching、流式 API。
- 数值差分、属性测试、benchmark、CMake packaging。

这些内容组合起来比孤立的 kernel demo 更有价值。

## 3. 当前结构性问题

### 3.1 缺少纵向权威真值

每个仓库都有自己的测试，却没有一个共同样例把同一输入贯穿所有相关层：

```text
GGUF tensor
 -> tiny-llm 单层输出
 -> logits
 -> token
 -> paged-infer 请求生命周期
```

因此存在三类“绿色假象”：

1. kernel 与项目内 reference 使用同一错误公式。
2. CPU runner 编译了 CUDA 代码，但 GPU case 全部 skip。
3. 资源不变量正确，但模型数值错误。

### 3.2 缺少跨仓契约

以下概念在不同仓库中没有共同定义：

- Q/K/V 是 `[B,S,H,D]`、`[B,H,S,D]` 还是展平 token-major。
- GQA 的 KV head 映射公式。
- RoPE 是 interleaved pair 还是 half-split，以及 cache 的物理排列。
- KV Cache 的 layer、sequence、head、position、dimension 顺序。
- prefill 是否包含历史 prefix、decode 时当前 token 何时变为可见。
- benchmark 的 kernel-only 与 end-to-end 边界。

这些契约不要求共享代码，但必须共享语义和测试向量。

### 3.3 学习仓和旗舰仓的证据强度没有明显区分

Academy 的部分模块使用“工业级/生产级”措辞，而测试和 CI 仍是教学级。相反，`tiny-llm` 作为运行时旗舰，端到端正确性证据还没有达到应有强度。

建议用三类标签代替模糊成熟度：

- `tutorial`：用于解释概念，可能固定形状，必须说明简化。
- `reference`：数值正确优先，有独立 oracle，不承诺性能。
- `candidate`：具有真实 workload、GPU CI、package 和性能证据，但仍非生产承诺。

当前不建议任何仓库使用 `production-ready`。

## 4. 目标架构

建议保持仓库物理独立，只增加三类逻辑共享资产：

### 4.1 共享契约文档

由本报告的 [跨仓张量与运行时契约](03-cross-repo-contracts.md) 起步，稳定后可以放到 Academy 的学习路径或未来组织 `.github` 仓库中。它定义语义，不引入跨仓源码依赖。

### 4.2 共享测试向量

使用小而可审计的 JSON/NPZ/safetensors fixture：

- 固定 seed 和小维度张量。
- 标注 layout、dtype、shape、模型配置。
- 由外部 PyTorch/Hugging Face 脚本生成。
- 每个消费仓库只实现自己的加载与比较适配器。

### 4.3 共享结果 schema

所有 GPU 验证和 benchmark 输出机器可读 JSON，至少记录：

- repo commit、dirty 状态、CUDA/driver/GPU。
- dtype、shape、layout、causal/GQA 等语义参数。
- reference 名称和版本。
- warmup、iterations、统计方法和原始样本。
- correctness tolerance、最大误差和通过状态。

## 5. 跨仓复用原则

### 应复用

- 测试向量和数学契约。
- 结果 schema 与环境采集方法。
- 失败分类、验收术语和文档模板。

### 暂不复用

- `cuflash-attn` 不能直接作为 `tiny-llm` attention backend：它缺少 GQA 和 decode 专用契约。
- Academy kernel 不应被运行时直接依赖：教学实现的优化和 API 稳定性目标不同。
- `paged-infer` 不应直接依赖 `tiny-llm` 的内部 C++ 类型；需要窄 backend ABI/IPC/FFI 契约。
- Triton reference 不应成为 CUDA kernel 的唯一 oracle；两者可能复制相同公式错误。

## 6. 组织投资决策

### 继续维护

- `tiny-llm`：组织旗舰，优先建立真实模型正确性。
- `cuflash-attn`：kernel 深度旗舰，优先建立可信 benchmark。

### 维护模式

- `triton-fused-ops`：修完 correctness 和 CI 后冻结扩展。
- `cuda-kernel-academy`：纠错和维护，不新增模块。
- `paged-infer`：修 P0 后保持控制面练习，等待真实 backend。

## 7. 不建议的方向

- 再写一套新推理引擎或新 FlashAttention 仓库。
- 在 `tiny-llm` 正确生成 token 前优化 TTFT/TPOT。
- 为追求代码复用，把所有仓库强行做成互相链接的库。
- 把 CI 编译成功描述成 GPU 数值验证。
- 用自己实现生成的结果验证自己的实现，而没有第三方 oracle。

