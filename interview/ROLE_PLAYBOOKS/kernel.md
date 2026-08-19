# Kernel 岗位角色手册

> Phase I6。目标岗位偏 **kernel / CUDA / Triton** 时的讲述顺序与重点。
> 同一份材料重排：主推 cuflash-attn + triton-fused-ops + cuda-foundations；tiny-llm 只讲转置优化与 microbench；paged-infer 一分钟带过。

## 一句话定位

“我是写 kernel 的：从 SGEMM 阶梯到 FlashAttention 前向/反向，同一个问题用 CUDA 和 Triton 各写一遍，并诚实记录负结果。”

## 20 分钟讲述时间轴

| 分钟 | 讲什么 | 用到的 QA_BANK / 数字 |
|------|--------|------------------------|
| 0–2 | 定位 + 能力链一句话：“我不是迷你 vLLM，是从算子层级长出来的” | Q10 |
| 2–6 | CUDA 基础深挖：GEMM 阶梯动机、bank conflict、shared memory、roofline/occupancy 口径 | Q1/Q4/Q5/Q8；0.58→0.92→0.66→0.68→1.09→5.58 TFLOPS |
| 6–11 | Triton：block 抽象、RoPE half-split 契约、TRIT-001 怎么发现、什么时候 Triton 什么时候 CUDA | Q11/Q16/Q17/Q19 |
| 11–17 | FlashAttention：online softmax、grid.y 65535、causal skip 负结果、FlashDecoding、LogicalHBM 口径 | Q21/Q25/Q26/Q27/Q30；±2% 表格 |
| 17–18 | tiny-llm 只讲转置优化与 microbench：为什么 M==1 GEMM 卡在访存 | Q41；lm_head 10.0002→0.9794 ms |
| 18–19 | paged-infer 一分钟带过：只给“控制面在另一师傅手上，我不管调度” | Q57（一句话，不展开） |
| 19–20 | 反问 + 边界声明：“FA 没接到 runtime generate；Triton 是参考不是生产选型” | Q23/Q28 的 OUT |

## 推荐的 QA_BANK 题号子集（28 题）

`Q1 – Q28`（CUDA 基础 Q1–Q10、Triton Q11–Q20、FlashAttention Q21–Q28）。理由：kernel 岗把这三个主题讲深就够了；runtime/serving 的题目按边界一句话带过。

## 简历条目重排顺序

1. `cuflash-attn › 1`：FA 前后向多精度差分 + grid.y 展平回归（E7）——先证明 kernel 正确性纪律。
2. `cuflash-attn › 2`：causal ±2% 负结果留文档（E8）——证明诚实。
3. `cuflash-attn › 3`：FlashDecoding Split-KV（E9）——证明 decode 方向的理解。
4. `triton-fused-ops › 2`：TRIT-001 RoPE half-split 契约修复（E4）——证明能被审计抓到约定 bug。
5. `triton-fused-ops › 3`：torch.library 三 op 注册、compile smoke 如实 skip（E5）——证明接入模式。
6. `cuda-foundations › 1`：SGEMM 阶梯留负优化（E1）——证明教学也讲数字。
7. `总览 › 2`：转置 M==1 GEMM 的 microbench（E15）——只作为 kernel 微观视角收尾，不展开 runtime。

## 反问问题清单（≥3）

1. 你们 decode 路径的性能结论是 ncu 还是 end-to-end 实测？我这边 ncu 报 `ERR_NVGPUCTRPERM`，用的是 kernel microbench。
2. 生产 kernel 遇到“优化低于噪声”时你们回滚还是留文档？我是按负结果留文档的。
3. WMMA/TMA/warp specialization 在你们目标卡（A100/H100/Blackwell）上分别卡在哪一层？我想知道要补哪块。
4. 如果入职先接 Triton 还是 CUDA kernel？我可以两边各写一份同一算子。
