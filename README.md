# AICL-Lab · AI Infra Learning Portfolio

> **五仓作品集 landing 页**：一份把「从 CUDA 算子到推理引擎」的学习路径收口到
> 五个可独立验证仓库的元仓库。Phase 2（E 阶段）完成后五仓进入"面试就绪冻结"态。

## 五仓地图（四层能力）

| 层 | 仓库 | 一句话定位 | 状态 | 链接 |
|----|------|-----------|------|------|
| L1 CUDA 基础 | [cuda-foundations](https://github.com/aicl-lab/cuda-foundations) | 从 SGEMM 到可复用推理组件的系统性 CUDA 算子工程学习路径（原 cuda-kernel-academy，已改名） | ✅ 冻结 | [README](https://github.com/aicl-lab/cuda-foundations) |
| L2 Triton 算子 | [triton-fused-ops](https://github.com/aicl-lab/triton-fused-ops) | 精简 Triton 算子库（RMSNorm+RoPE / SwiGLU / FlashAttention / SGEMM）+ torch.library 注册 | ✅ 冻结 | [README](https://github.com/aicl-lab/triton-fused-ops) |
| L3 Attention | [cuflash-attn](https://github.com/aicl-lab/cuflash-attn) | 从零实现的 CUDA C++ FlashAttention 前后向（FP16/BF16 WMMA） | ✅ 冻结 | [README](https://github.com/aicl-lab/cuflash-attn) |
| L4 推理引擎 | [tiny-llm](https://github.com/aicl-lab/tiny-llm) | CUDA 原生 C++ 推理引擎（GGUF / W8A16 / 分页 KV 策略 1） | ✅ 冻结 | [README](https://github.com/aicl-lab/tiny-llm) |
| L4 控制面 | [paged-infer](https://github.com/aicl-lab/paged-infer) | PagedAttention 分页 KV + Continuous Batching 的推理控制面（Rust） | ✅ 冻结 | [README](https://github.com/aicl-lab/paged-infer) |

## 阅读顺序

1. **cuda-foundations**（基础）→ 2. **triton-fused-ops**（Triton 表达同一批算子）→
   3. **cuflash-attn**（FlashAttention 前后向深挖）→
   4. **tiny-llm**（模型加载 + 推理内核 + 分页 KV 策略 1）→
   5. **paged-infer**（分页调度 / continuous batching / HTTP 控制面，接 tiny-llm 真实后端）。

推荐路径：`cuda-foundations → triton-fused-ops → cuflash-attn → tiny-llm → paged-infer`。

## Phase 2 完成证据摘要

- **tiny-llm**：W8A16 推理端到端可用；**TPOT ≈ 6.1 ms/token**（本机实测，
  `tiny_llm_bench`）；170 tests；分页 KV（策略 1）与连续 KV 逐 token 差分一致。
- **paged-infer**：**3 并发分页请求 e2e 与 llama.cpp greedy 对齐**（请求 1 全序列
  严格一致；请求 2 的 `equals`/`is` 为 W8A16 vs Q4_K_M 量化 argmax 边界翻转，
  已诚实记录为"前缀一致 + EOS 终止 + 分歧注释"，不伪造全序列一致）。
- **cuflash-attn**：FlashAttention 前后向 FP32/FP16/BF16，FP16/BF16 前向接 WMMA；
  E 阶段修复 grid.y 65535 越界（B*H>65535 回归测试）并加入 causal 边界块跳过优化。
- **triton-fused-ops**：Triton SGEMM + `torch.library`（`torch.ops.triton_ops.*`）
  注册三个自定义算子，与 vLLM/SGLang 的 custom op 接入模式一致。
- **cuda-foundations**：仓库由旧名 `cuda-kernel-academy` 正式改名为
  `cuda-foundations`，审计归档见 `docs/organization-audit/`。

## 执行计划归档

本 meta 仓库同时归档各阶段执行计划（只读副本，权威版本在各仓库/根目录）：

- [`MASTER_PLAN.md`](MASTER_PLAN.md) — 总计划
- [`PHASE2_PLAN.md`](PHASE2_PLAN.md) — Phase 2 总计划
- [`PHASE2_NEXT.md`](PHASE2_NEXT.md) / [`PHASE2_NEXT_C.md`](PHASE2_NEXT_C.md) /
  [`PHASE2_NEXT_D.md`](PHASE2_NEXT_D.md) / [`PHASE2_NEXT_E.md`](PHASE2_NEXT_E.md) —
  各批次可执行细化（A→E）
- [`docs/organization-audit/`](docs/organization-audit/) — 组织审计归档
  （范围/架构/风险/跨仓契约/验证策略/工程治理/执行路线）

## 面试冻结声明

E 阶段完成后，五仓进入"**面试就绪冻结**"：不再扩新功能；只修正确性 bug 与文档
漂移；新想法记入各仓 ROADMAP 的"不做什么"清单。这是作品集交付纪律，不是降低标准。

## License

MIT（各子仓库 LICENSE 为准）。
