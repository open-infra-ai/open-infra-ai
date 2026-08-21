# open-infra-ai · AI Infra 学习作品集

> **AI Infra 工程学习作品集**：从 CUDA 内核到推理 serving 的完整能力链，
> 每个作品都有独立参考实现与差分验证。本仓是组织的 meta 仓：
> landing 页 + 学习路径 + 跨仓契约 + 计划档案 + 面试材料归档。

## 仓库地图（四层能力）

| 层 | 仓库 | 一句话定位 | 状态 |
|----|------|-----------|------|
| L1 CUDA 基础 | [cuda-foundations](https://github.com/open-infra-ai/cuda-foundations) | 从 SGEMM 到可复用推理组件的系统性 CUDA 算子工程学习路径 | active |
| L1 Triton 算子 | [triton-fused-ops](https://github.com/open-infra-ai/triton-fused-ops) | 精简 Triton 算子库（RMSNorm+RoPE / SwiGLU / FlashAttention / SGEMM）+ torch.library 注册 | stable |
| L1 Attention | [cuflash-attn](https://github.com/open-infra-ai/cuflash-attn) | 从零实现的 CUDA C++ FlashAttention 前后向（FP16/BF16 WMMA + FlashDecoding） | stable |
| L2 推理引擎 | [tiny-llm](https://github.com/open-infra-ai/tiny-llm) | CUDA 原生 C++ 推理引擎（GGUF / W8A16 / 分页 KV 策略 1），导出 C ABI | active |
| L3 控制面 | [paged-infer](https://github.com/open-infra-ai/paged-infer) | PagedAttention 分页 KV + Continuous Batching 的推理控制面（Rust），经 C ABI 接 tiny-llm | active |

**状态语义**：`active` = 学习/演进中；`stable` = 作品完成，只修正确性 bug 与文档；
`archived` = 不再维护。状态以本表为唯一权威注册表，与各仓 README 状态行、
GitHub topics 三处同步。

## 阅读顺序

1. **cuda-foundations**（基础）→ 2. **triton-fused-ops**（Triton 表达同一批算子）→
   3. **cuflash-attn**（FlashAttention 前后向深挖）→
   4. **tiny-llm**（模型加载 + 推理内核 + 分页 KV 策略 1）→
   5. **paged-infer**（分页调度 / continuous batching / HTTP 控制面，接 tiny-llm 真实后端）。

完整方法论（优化循环、不变量测试、阶段完成标准）见本仓
[`LEARNING_PATH.md`](LEARNING_PATH.md)——组织级导航的唯一权威入口。

## 跨仓契约

- **ABI 契约（代码双源）**：[`tiny-llm/include/tiny_llm/ffi.h`](https://github.com/open-infra-ai/tiny-llm/blob/master/include/tiny_llm/ffi.h)
  ⇄ [`paged-infer/src/tiny_llm_ffi.rs`](https://github.com/open-infra-ai/paged-infer/blob/master/src/tiny_llm_ffi.rs)
  （repr(C) 布局守卫测试即一致性检查）。
- **语义契约（12 条）**：维度命名 / 布局 / GQA / RoPE / KV 事务语义 / 采样顺序等，
  live 版见 [`docs/cross-repo-contracts.md`](docs/cross-repo-contracts.md)。

## 完成证据摘要

- **tiny-llm**：W8A16 推理端到端可用；**TPOT ≈ 6.1 ms/token**（本机实测，
  `tiny_llm_bench`）；170 tests；分页 KV（策略 1）与连续 KV 逐 token 差分一致。
- **paged-infer**：**3 并发分页请求 e2e 与 llama.cpp greedy 对齐**（请求 1 全序列
  严格一致；请求 2 的 `equals`/`is` 为 W8A16 vs Q4_K_M 量化 argmax 边界翻转，
  已诚实记录为"前缀一致 + EOS 终止 + 分歧注释"，不伪造全序列一致）。
- **cuflash-attn**：FlashAttention 前后向 FP32/FP16/BF16，FP16/BF16 前向接 WMMA；
  修复 grid.y 65535 越界（B*H>65535 回归测试）并加入 causal 边界块跳过优化。
- **triton-fused-ops**：Triton SGEMM + `torch.library`（`torch.ops.triton_ops.*`）
  注册三个自定义算子，与 vLLM/SGLang 的 custom op 接入模式一致。
- **cuda-foundations**：仓库由旧名 `cuda-kernel-academy` 改名为 `cuda-foundations`，
  审计归档见 `docs/organization-audit/`。

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
