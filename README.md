# open-infra-ai · AI Infra 学习作品集

> **AI Infra 工程学习作品集**：从 CUDA 内核到推理 serving 的完整能力链，
> 每个作品都有独立参考实现与差分验证。本仓是组织的 meta 仓：
> landing 页 + 状态注册表 + 学习路径 + 跨仓契约 + 历史档案。

仓库职责与私人求职材料的边界见
[`docs/repository-boundaries.md`](docs/repository-boundaries.md)。

## 仓库地图（四层能力）

| 层 | 仓库 | 一句话定位 | 状态 |
|----|------|-----------|------|
| L1 CUDA 基础 | [cuda-foundations](https://github.com/open-infra-ai/cuda-foundations) | 从 SGEMM 到可复用推理组件的系统性 CUDA 算子工程学习路径 | active |
| L1 Triton 算子 | [trifuse](https://github.com/open-infra-ai/trifuse) | 精简 Triton 算子库（RMSNorm+RoPE / SwiGLU / FlashAttention / SGEMM）+ torch.library 注册 | stable |
| L1 Attention | [cuflash](https://github.com/open-infra-ai/cuflash) | 从零实现的 CUDA C++ FlashAttention 前后向（FP16/BF16 WMMA + FlashDecoding） | stable |
| L2 推理引擎 | [tiny-llm](https://github.com/open-infra-ai/tiny-llm) | CUDA 原生 C++ 推理引擎（GGUF / W8A16 / 分页 KV 策略 1），导出 C ABI | active |
| L3 控制面 | [paged-serving](https://github.com/open-infra-ai/paged-serving) | PagedAttention 分页 KV + Continuous Batching 的推理控制面（Rust），经 C ABI 接 tiny-llm | active |

**状态语义**：`active` = 学习/演进中；`stable` = 作品完成，只修正确性 bug 与文档；
`archived` = 不再维护。状态以本表为唯一权威注册表，与各仓 README 状态行、
GitHub topics 三处同步。

## 阅读顺序

1. **cuda-foundations**（基础）→ 2. **trifuse**（Triton 表达同一批算子）→
   3. **cuflash**（FlashAttention 前后向深挖）→
   4. **tiny-llm**（模型加载 + 推理内核 + 分页 KV 策略 1）→
   5. **paged-serving**（分页调度 / continuous batching / HTTP 控制面，接 tiny-llm 真实后端）。

完整方法论（优化循环、不变量测试、阶段完成标准）见本仓
[`LEARNING_PATH.md`](LEARNING_PATH.md)——组织级导航的唯一权威入口。

## 跨仓契约

- **ABI 契约（代码双源）**：[`tiny-llm/include/tiny_llm/ffi.h`](https://github.com/open-infra-ai/tiny-llm/blob/master/include/tiny_llm/ffi.h)
  ⇄ [`paged-serving/src/tiny_llm_ffi.rs`](https://github.com/open-infra-ai/paged-serving/blob/master/src/tiny_llm_ffi.rs)
  （repr(C) 布局守卫测试即一致性检查）。
- **语义契约（12 条）**：维度命名 / 布局 / GQA / RoPE / KV 事务语义 / 采样顺序等，
  live 版见 [`docs/cross-repo-contracts.md`](docs/cross-repo-contracts.md)。

## 完成证据摘要

- **tiny-llm**：W8A16 推理端到端可用；clean commit `565da79` 的 schema v2
  五组配对 CUDA Graph A/B 中，TPOT 8.322→**5.225 ms/token**（-37.2%），decode
  吞吐 120.168→**191.384 tok/s**（+59.3%）；10 个进程原始 JSONL、模型哈希和
  [完整限制](https://github.com/open-infra-ai/tiny-llm/blob/master/docs/performance/results/2026-08-23-cuda-graphs-ab.md)
  已归档，TTFT 不作改善声明。当前 **193 项测试通过**；分页 KV（策略 1）与连续 KV
  逐 token 差分一致。
- **paged-serving**：**3 并发分页请求 e2e 与 llama.cpp greedy 对齐**（请求 1 全序列
  严格一致；请求 2 的 `equals`/`is` 为 W8A16 vs Q4_K_M 量化 argmax 边界翻转，
  已诚实记录为"前缀一致 + EOS 终止 + 分歧注释"，不伪造全序列一致）；
  默认测试当前 **232 项通过**。真实 GPU serving 吞吐报告仍是待补证据，不用 CPU
  参考后端数字冒充 GPU 性能。
- **cuflash**：FlashAttention 前后向 FP32/FP16/BF16，FP16/BF16 前向接 WMMA；
  修复 grid.y 65535 越界（B*H>65535 回归测试）并加入 causal 边界块跳过优化；
  RTX 3060 Laptop 当前 **81/81 项测试通过**（可选 PyTorch 集成 1 项跳过）。
- **trifuse**：Triton SGEMM + `torch.library`（`torch.ops.trifuse.*`）
  注册三个自定义算子；CPU-only **57 passed / 66 skipped**，RTX 3060 Laptop
  **123/123 passed**。
- **cuda-foundations**：SGEMM 与推理组件教学阶梯；RTX 3060 Laptop 当前
  **261/261 项测试通过**。旧名审计快照见 `docs/organization-audit/`。

> 以上是 **2026-08-23 本地验证快照**。性能数字仍以各技术仓的结果文件、硬件、
> commit 与复现命令为准；测试数量只表示当前验证面，不直接等价于项目质量。

## 面试展示优先级

1. **主项目：tiny-llm** —— 推理加速岗位先讲真实模型链路、W8A16、decode、
   M==1 GEMM、CUDA Graphs 与可复现的端到端指标。
2. **专项深挖：cuflash** —— 用来证明 CUDA kernel、online softmax、
   Tensor Core、数值正确性和 profiling 深度。
3. **系统扩展：paged-serving** —— 用来证明 Paged KV、continuous batching、
   调度不变量、HTTP/SSE 与服务评测方法；不把它包装成低层 kernel 加速项目。
4. `cuda-foundations` 与 `trifuse` 是基础与横向对照证据，不与主项目争夺叙事中心。

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
