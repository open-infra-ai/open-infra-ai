# open-infra-ai · AI Infra 系统作品集

> 从可验证的 CUDA/Triton kernel 到真实权重推理与 serving 控制面。
> 每个性能结论都应绑定代码、环境、原始结果和适用边界。本仓是组织的
> landing 页、状态注册表、学习路径、跨仓契约和历史档案。

仓库职责与私人求职材料的边界见
[`docs/repository-boundaries.md`](docs/repository-boundaries.md)。

## 90 秒入口

| 面试主题 | 从哪里开始 | 重点 |
|----------|--------------|------|
| 端到端 LLM Serving | [tiny-llm](https://github.com/open-infra-ai/tiny-llm) + [paged-serving](https://github.com/open-infra-ai/paged-serving) | 真实权重、W8A16、Paged KV、continuous batching、C ABI、HTTP/SSE |
| CUDA kernel 深挖 | [cuflash](https://github.com/open-infra-ai/cuflash) | online softmax、Tensor Core、FlashDecoding、数值与性能边界 |
| CUDA/Triton 基础与对照 | [cuda-foundations](https://github.com/open-infra-ai/cuda-foundations) + [trifuse](https://github.com/open-infra-ai/trifuse) | 优化阶梯、参考实现、差分测试、`torch.library` |
| 可追溯证据 | [`docs/evidence-index.md`](docs/evidence-index.md) | 结果包、复现入口、已知限制和尚待补齐的证据 |

## 作品集架构

```text
旗舰系统：HTTP/SSE → paged-serving（调度、Paged KV）
                              ⇅ 受测试的同进程 C ABI
                         tiny-llm（真实权重、CUDA decode）

Kernel 深挖：cuflash（独立作品，不接入 tiny-llm generate）
基础对照：cuda-foundations · trifuse
```

`tiny-llm` 和 `paged-serving` 是同一旗舰系统的数据面与控制面；
`cuflash` 证明 CUDA kernel 深度，但不是旗舰请求路径的依赖。
可交互地查看请求、C ABI 与批量执行边界，见
[`docs/serving-control-data-plane-p2-batch-postprocess.html`](docs/serving-control-data-plane-p2-batch-postprocess.html)
（[当前源规格](docs/serving-control-data-plane-p2-batch-postprocess.architecture.json)；
[已冻结的 device-greedy 快照](docs/serving-control-data-plane-p2-greedy.html)）。

## 状态注册表

| 角色 | 仓库 | 唯一主责 | 状态 |
|------|------|----------|------|
| 基础 | [cuda-foundations](https://github.com/open-infra-ai/cuda-foundations) | 从 SGEMM 到可复用推理组件的 CUDA 工程学习路径 | stable |
| 对照 | [trifuse](https://github.com/open-infra-ai/trifuse) | Triton 算子与 `torch.library` 集成 | stable |
| 深挖 | [cuflash](https://github.com/open-infra-ai/cuflash) | CUDA C++ FlashAttention 前后向与 FlashDecoding | stable |
| 旗舰数据面 | [tiny-llm](https://github.com/open-infra-ai/tiny-llm) | 真实权重加载、量化、decode、KV 与 C ABI | active |
| 旗舰控制面 | [paged-serving](https://github.com/open-infra-ai/paged-serving) | Paged KV、continuous batching、HTTP/SSE 与 serving 评测 | active |

**状态语义**：`active` = 学习/演进中；`stable` = 作品完成，只修正确性 bug 与文档；
`archived` = 不再维护。状态以本表为唯一权威注册表，与各仓 README 状态行、
GitHub topics 三处同步。

## 按目标阅读

1. **准备系统/Serving 面试**：`paged-serving → C ABI → tiny-llm`，按一条请求生命周期阅读。
2. **准备 CUDA kernel 面试**：先看 `cuflash`，再用 `trifuse` 解释 Triton/CUDA 取舍。
3. **从基础完整学习**：按 `cuda-foundations → trifuse → cuflash → tiny-llm → paged-serving`。

完整方法论（优化循环、不变量测试、阶段完成标准）见本仓
[`LEARNING_PATH.md`](LEARNING_PATH.md)——组织级导航的唯一权威入口。

## 跨仓契约

- **ABI 契约（代码双源）**：[`tiny-llm/include/tiny_llm/ffi.h`](https://github.com/open-infra-ai/tiny-llm/blob/master/include/tiny_llm/ffi.h)
  ⇄ [`paged-serving/src/tiny_llm_ffi.rs`](https://github.com/open-infra-ai/paged-serving/blob/master/src/tiny_llm_ffi.rs)
  （repr(C) 布局守卫测试即一致性检查）。
- **语义契约（12 条）**：维度命名 / 布局 / GQA / RoPE / KV 事务语义 / 采样顺序等，
  live 版见 [`docs/cross-repo-contracts.md`](docs/cross-repo-contracts.md)。

## 完成证据摘要

跨仓复现入口、证据层级与当前缺口见
[`docs/evidence-index.md`](docs/evidence-index.md)。

- **tiny-llm**：W8A16 推理端到端可用；clean commit `565da79` 的 schema v2
  五组配对 CUDA Graph A/B 中，TPOT 8.322→**5.225 ms/token**（-37.2%），decode
  吞吐 120.168→**191.384 tok/s**（+59.3%）；10 个进程原始 JSONL、模型哈希和
  [完整限制](https://github.com/open-infra-ai/tiny-llm/blob/master/docs/performance/results/2026-08-23-cuda-graphs-ab.md)
  已归档，TTFT 不作改善声明。当前 **193 项测试通过**；分页 KV（策略 1）与连续 KV
  逐 token 差分一致。
- **paged-serving**：**3 并发分页请求 e2e 与 llama.cpp greedy 对齐**（请求 1 全序列
  严格一致；请求 2 的 `equals`/`is` 为 W8A16 vs Q4_K_M 量化 argmax 边界翻转，
  已诚实记录为"前缀一致 + EOS 终止 + 分歧注释"，不伪造全序列一致）。真实 CUDA
  closed-loop/Poisson 已有 [P1 历史基线](https://github.com/open-infra-ai/paged-serving/tree/master/benchmarks/serving/results/2026-09-04-RTX3060Laptop-paged-serving)
  与 [P2 当前流式矩阵](https://github.com/open-infra-ai/paged-serving/tree/master/benchmarks/serving/results/2026-09-04-RTX3060Laptop-paged-serving-p2-streaming)：
  后者绑定双仓 commit、模型 SHA-256、1344 条原始请求与固定 Poisson seed，但 P2 多档
  TTFT p95 未通过 10% 收敛门槛，不能写成稳定 SLO、通用容量或生产成熟度。
- **cuflash**：FlashAttention 前后向 FP32/FP16/BF16，FP16/BF16 前向接 WMMA；
  修复 grid.y 65535 越界（B*H>65535 回归测试）并加入 causal 边界块跳过优化；
  RTX 3060 Laptop 当前 **81/81 项测试通过**（可选 PyTorch 集成 1 项跳过）。
- **trifuse**：Triton SGEMM + `torch.library`（`torch.ops.trifuse.*`）
  注册三个自定义算子；CPU-only **57 passed / 66 skipped**，RTX 3060 Laptop
  **123/123 passed**。
- **cuda-foundations**：SGEMM 与推理组件教学阶梯；RTX 3060 Laptop 当前
  **261/261 项测试通过**。旧名审计快照见 `docs/organization-audit/`。

> 除明确更新的 paged-serving 2026-09-04 结果外，以上是 **2026-08-23 本地验证快照**。
> 性能数字仍以各技术仓的结果文件、硬件、commit 与复现命令为准；测试数量只表示当前
> 验证面，不直接等价于项目质量。

## 面试展示优先级

1. **旗舰系统：tiny-llm + paged-serving** —— 沿“请求 → 调度 → block table →
   C ABI → CUDA decode → SSE token”讲清数据面/控制面分工，并区分正确性证据与
   已归档但带收敛限制的真实 GPU Serving 证据。
2. **专项深挖：cuflash** —— 用来证明 CUDA kernel、online softmax、
   Tensor Core、数值正确性和 profiling 深度。
3. **基础与对照：cuda-foundations + trifuse** —— 证明优化方法、参考实现和
   CUDA/Triton 工程取舍，不与旗舰系统争夺叙事中心。

## 求职与面试执行

活跃的 12 周计划、简历草稿、岗位清单、投递模板与上游贡献练习已迁到个人执行仓
[`holtwood/ai-infra-interview-prep`](https://github.com/holtwood/ai-infra-interview-prep)。
本组织只承载可复现的公开技术作品和跨仓契约，不再混入随求职进程频繁变化的私人材料。

## 档案区

以下内容是历史记录，**只读存档，不再更新**；文中旧组织名（AICL-Lab / aicl-lab）
与旧工作区路径是当时事实的忠实记录，不改写。

- [`LEARNING_PATH.md`](LEARNING_PATH.md) 之外的计划文档（`MASTER_PLAN.md` /
  `PHASE2_*.md` / `PHASE3_PLAN.md` / `PLAN_v3.md` / `PLAN_I.md`）——
  Phase 1–3 与面试执行期的历史执行计划。
- [`interview/`](interview/) —— 面试证据包与排练材料（Phase 3 / Phase I 产物）：
  证据矩阵、数字卡、讲述稿、QA 库、模拟面试、简历条目等。
- [`docs/organization-audit/2026-08-13/`](docs/organization-audit/2026-08-13/) ——
  组织审计只读快照（当时教学仓还叫 `cuda-kernel-academy`，不是当前事实）。
- [`changelog/`](changelog/) —— 工作区治理变更记录。

## License

MIT（各子仓库 LICENSE 为准）。
