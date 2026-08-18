# triton-fused-ops · 10 分钟讲述稿

数字见 [`../NUMBERS_CARD.md`](../NUMBERS_CARD.md) §5。

## 0. 一句话定位

可差分验证的 Triton 融合算子练习仓。

## 1. 2 分钟：做什么、边界、为什么这样切

三条路径：`fused_rmsnorm_rope`、`fused_gated_mlp`（SwiGLU）、FlashAttention **前向**参考；外加 Triton SGEMM 与 `torch.ops.triton_ops.*`。

FlashAttention 本仓只做 cuflash-attn 的独立参考，不讲前后向优化（README 降级声明）。

为什么存在：和 cuda-foundations 做 **同题异构**（同一 GEMM，CUDA C++ vs Triton）；和 cuflash 做 **同题深浅**（Triton 参考 vs CUDA 深挖）。验证方法（独立 NumPy/PyTorch reference、失败路径、GPU skip 不报 pass）是这个仓的资产。

本次 freeze：pytest **116 passed, 1 skipped**（`torch.compile` smoke）。

## 2. 3 分钟：最难的实现细节

**TRIT-001：RoPE 是 half-split 还是 interleaved。**

Llama/Qwen 常用 `rotate_half`：把最后一维切成两半，`[-x_second, x_first]`，再 `x*cos + rotate_half(x)*sin`。错误实现会用 `repeat_interleave(freqs, 2)`，那是 pair-wise 交错，数值能「看起来像 RoPE」但和 HF 对不上。

修复：`triton_ops/reference/rmsnorm_rope.py` 用 `np.concatenate([cos, cos], axis=-1)` 做 full cache（`b1bcdcb`）。kernel、reference、example 必须同一契约。只拿 kernel 和自己的 reference 比会共模出错，所以要有外部 half-split 约定。

融合收益一句话：RMSNorm+RoPE 从多次 HBM 往返收成一次（README 0.104 ms @ (1,128,4096)）。

## 3. 2 分钟：优化/调试故事

故事选 **torch.library 注册**，不是再吹一个 kernel。

- Before：只能 `from triton_ops.kernels import sgemm`。推理框架（vLLM/SGLang）走 `torch.library` 才能进 `torch.compile` / 图。
- 改动：`triton_ops/ops.py` 注册 `triton_ops::sgemm` / `fused_rmsnorm_rope` / `fused_gated_mlp`。优先 `torch.library.triton_op`，否则 `custom_op + register_fake`（`1bbf5c8`）。内部不复制 kernel。
- After：`import triton_ops; torch.ops.triton_ops.sgemm(a,b)`。差分测试在 `tests/test_torch_library.py`。
- 代价：本次 freeze `test_torch_compile_smoke` **skip**（compile 失败不伪造通过）。只接受 CUDA 张量。

SGEMM 本身：`tests/test_sgemm.py` **24** 项 vs `torch.mm`，含非 2 幂与失败路径。

## 4. 2 分钟：验证方法

- CPU：reference 契约、输入校验。
- GPU：kernel vs reference，rtol/atol=1e-2。
- 无 GPU：明确 skip，不把 skip 当 pass。
- 负资产：删掉假 FP8 E4M3（其实是 uint8 线性量化）。
- 实测表：gated_mlp **3.45 ms**、rmsnorm_rope **0.104 ms**（commit `ebf6c32+`，README 原文栈）。

## 5. 1 分钟：短板与下一步

Autotuner 基础设施在，和 wrapper 的生产级打通不是旗舰。Triton FA 不是 cuflash 竞品。下一步（冻结外）：Triton FA vs CUDA FA 的 10 分钟对比讲述（本 Phase 3 稿补上）。不新开 INT8/FP8 融合。

## 6. 追问清单

1. Triton 的 program 对应 CUDA 的什么？ → 通常一个 program = 一个 block。
2. 为什么 load/store 必须 mask？ → tile 会越过 M/N/K。
3. `tl.dot` 的累加精度？ → 常用 fp32 acc。
4. TRIT-001 怎么发现的？ → helper/API/example 排列不一致；审计点名 half-split。
5. concat cache 长什么样？ → `[c0..c_{D/2-1}, c0..c_{D/2-1}]`。
6. 何时 Triton、何时 CUDA？ → 见 `cross-cutting.md`。
7. torch.library 和直接调 kernel 的差别？ → 图可见、fake/meta、与框架 custom op 同一接入模式。
8. 为什么 compile smoke skip 也可以交？ → 协议禁止伪造通过。
9. 3.45 ms 算快吗？ → 约 10 TFLOPS vs 卡理论 ~46 TFLOPS FP16，练习实现。
10. 为什么删 FP8？ → 名称撒谎；uint8 线性量化不是 E4M3。
