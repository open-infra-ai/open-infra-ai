# AICL-Lab 组织级代码审计

> **只读快照（2026-08-13）**。当时教学仓仍叫 `cuda-kernel-academy`，本文档不代表当前仓名或完成度。
> 当前组织地图见仓库根 [`README.md`](../../../README.md)。

> 审计日期：2026-08-13
>
> 审计方式：本地源码、测试、构建、CI、发布配置与文档的只读审计
>
> 审计目标：为 AI Inference 学习路线、项目维护顺序和后续实现任务提供可信依据

## 1. 结论摘要

AICL-Lab 当前保留的五个仓库已经形成一条合理的学习链：

```text
CUDA 基础与系统教学
    -> Triton 算子与验证方法
    -> CUDA FlashAttention 专项
    -> 真实模型运行时
    -> Serving 控制面
```

仓库之间的职责边界总体清楚，不建议继续合并，也不建议在现阶段新增仓库。当前最主要的问题不是项目数量，而是缺少贯穿整条推理链的共同正确性契约和外部 oracle。五个仓库可以分别编译或通过自己的测试，但这些绿灯尚不能共同证明某个真实模型 token 是正确生成的。

组织下一阶段应从“增加功能和算子”切换到“建立可证明的真实推理链”：

1. 先修复 `tiny-llm` 的张量布局、GQA、RoPE 和模型权重契约。
2. 建立单算子、Transformer 层、模型 logits、token 序列四级差分验证。
3. 修复 `paged-infer` 的 layer-aware KV Cache，再考虑真实后端对接。
4. 统一 Triton RoPE 契约，清理共同参考实现造成的共模错误。
5. 修正 Academy 中无法启动或会误导学习者的示例。
6. 正确性成立后再刷新 FlashAttention 与端到端 benchmark。

## 2. 文档导航

### 组织级结论

- [审计范围与方法](00-scope-and-method.md)
- [组织架构与项目组合审计](01-organization-architecture.md)
- [风险登记表](02-risk-register.md)
- [跨仓张量与运行时契约](03-cross-repo-contracts.md)
- [验证与基准体系](04-validation-and-benchmark-strategy.md)
- [工程治理与发布基线](05-engineering-governance.md)
- [分阶段实施路线图](06-execution-roadmap.md)
- [便宜模型执行任务规范](07-implementation-task-specs.md)

### 逐仓审计

- [`paged-infer`](repos/paged-infer.md)
- [`tiny-llm`](repos/tiny-llm.md)
- [`cuflash-attn`](repos/cuflash-attn.md)
- [`triton-fused-ops`](repos/triton-fused-ops.md)
- [`cuda-kernel-academy`](repos/cuda-kernel-academy.md)

### 优先修复设计

- [`tiny-llm` 推理正确性修复设计](designs/tiny-llm-correctness.md)
- [`paged-infer` KV Cache 与 API 语义修复设计](designs/paged-infer-correctness.md)
- [`triton-fused-ops` RoPE 与验证修复设计](designs/triton-fused-ops-correctness.md)
- [`cuda-kernel-academy` 教学正确性修复设计](designs/cuda-kernel-academy-corrections.md)
- [`cuflash-attn` 基准证据重建设计](designs/cuflash-attn-benchmark.md)
- [跨仓真实模型验证设计](designs/end-to-end-validation.md)

## 3. 仓库快照

| 仓库 | 提交 | 分支 | 跟踪文件 | 定位 |
|---|---|---:|---:|---|
| `paged-infer` | `a32c6df6c587e0da89327aeb10329051c60d8804` | `master` | 37 | Serving 控制面练习 |
| `tiny-llm` | `94c8d12942df337eef48d0b45df39fdca27c5b3f` | `master` | 105 | 真实模型运行时旗舰 |
| `cuflash-attn` | `f588314761ea7e9b3c8d194bce90af16b456ff07` | `master` | 101 | CUDA FlashAttention 专项 |
| `triton-fused-ops` | `ebf6c3296db367f23b117261ec0dbd990e741062` | `master` | 48 | Triton 算子与验证练习 |
| `cuda-kernel-academy` | `57c225faf059a0c96aa93ab5aa51f1b8eb47cdc8` | `master` | 327 | CUDA 学习总入口 |

文档中的代码位置和结论只对以上快照负责。后续仓库发生修改后，应重跑相关验收项，而不是继续把本报告视为当前事实。

## 4. 风险总览

### P0：阻断真实推理正确性

| 编号 | 仓库 | 风险 |
|---|---|---|
| TLLM-001 | `tiny-llm` | Q/K/V 生产者与 attention 消费者的张量布局不一致 |
| TLLM-002 | `tiny-llm` | GQA 缺少 query head 到 KV head 的映射 |
| TLLM-003 | `tiny-llm` | RoPE 未进入 Transformer 计算路径 |
| PINF-001 | `paged-infer` | CPU reference KV Cache 不含 layer 维度，跨层覆盖 |
| TRIT-001 | `triton-fused-ops` | RoPE helper、API、kernel 与示例采用不一致的排列契约 |
| CKA-001 | `cuda-kernel-academy` | TensorCraft FlashAttention 启动 2048 threads/block，无法合法执行 |
| CKA-002 | `cuda-kernel-academy` | 04 模块多层网络复用同一中间缓冲区，产生输入输出别名 |

### P1：证据、接口或工程闭环不足

- `tiny-llm` 缺少真实模型 logits/token oracle，CLI 与路线图不一致。
- `cuflash-attn` benchmark 文档、构建参数、基线实现和计时方式不一致。
- `triton-fused-ops` autotuner 未与真实 wrapper 打通，benchmark reference 默认走 CPU。
- `cuda-kernel-academy` CI 不编译 CUDA，部分教学叙述与实际内存访问不符。
- 全组织没有统一张量契约、GPU 验证矩阵和 benchmark 结果格式。

## 5. 推荐的证据阶梯

```text
L1  单算子：与独立 PyTorch/NumPy reference 差分
L2  单层：与 Hugging Face 或小型明确公式实现差分
L3  模型：固定 GGUF、固定 prompt、逐位置 logits 差分
L4  生成：固定采样策略，与 llama.cpp 逐 token 对齐
L5  Serving：并发、取消、失败和资源守恒
L6  性能：固定硬件、版本、预热、统计与 profiler 证据
```

不得用较高层的“程序没有崩溃”替代较低层的数值正确性，也不得在 L1–L4 未成立前将性能作为主要产出。

## 6. 当前推荐投入比例

| 方向 | 建议投入 | 原因 |
|---|---:|---|
| `tiny-llm` 正确性与 oracle | 50% | 这是组织真实模型故事的主轴 |
| `cuflash-attn` 数值与 benchmark 证据 | 20% | 已有较成熟工程外壳，适合形成 kernel 深度作品 |
| 跨仓验证资产 | 15% | 防止各仓 reference 共同犯错 |
| Academy 教学纠错 | 10% | 避免错误知识固化并改善作品可信度 |
| `paged-infer` 控制面维护 | 5% | 在真实计算后端可用前保持低优先级 |

## 7. 文档使用约定

后续实现模型应严格按任务文档操作：

- 一次只处理一个具有独立验收标准的任务。
- 不顺手重构，不扩大公开 API，不引入第二套抽象。
- 先补能失败的外部 oracle，再修改实现。
- 每项变更都记录实际运行过的命令、通过/跳过数量和硬件环境。
- GPU 不可用时只能声明“编译通过”或“测试跳过”，不能声明数值正确。
