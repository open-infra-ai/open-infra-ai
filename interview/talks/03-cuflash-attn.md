# cuflash-attn · 10 分钟讲述稿

数字见 [`../NUMBERS_CARD.md`](../NUMBERS_CARD.md) §4。因果跳过必须当负结果讲。

## 0. 一句话定位

从零实现的 CUDA FlashAttention 前后向参考。

## 1. 2 分钟：做什么、边界、为什么这样切

IN：FA 前向+反向；FP32/FP16/BF16；FP16/BF16 前向 WMMA；causal；FlashDecoding（query_len=1 Split-KV）；head_dim 32/64/128。

OUT：GEMM 教程、Triton 教学、推理运行时、Serving。tiny-llm **没有**链接本仓 kernel。

为什么独立成仓：FA 是面试高频深挖，和「能跑 generate」不是同一件事。教学代码可读，对标 FA2/FA3 的 warp specialization / TMA 明确不做。

本次 freeze：ctest **71 collected / 0 failed / 1 skip**（pytorch comparison）。

## 2. 3 分钟：最难的实现细节

**Online softmax + 不物化 S。**

标准 attention：`S = QK^T` 是 `[S,S]`，再 softmax、乘 V。O(N²) 写回 HBM。

Online softmax：按 KV tile 维护 `m`（行最大）和 `l`（softmax 分母）。来一块新 K/V 时：

1. 算局部 `m_new = max(m, tile_max)`
2. 用 `exp(m - m_new)` 缩放已有分子（未归一化 O）和 `l`
3. 累加本 tile 的 `exp(s - m_new) * V`

最后 `O /= l`。辅助内存 O(N)，不是 O(N²)。logsumexp `L` 在 FP16/BF16 API 里保持 **FP32**（v0.5.0 breaking），否则反向 `exp(S-L)` 会坏。

第二细节（30 秒）：`grid.y = B*H` 在 B*H>65535 时 launch 非法。已展平到 `grid.x`（`d144765`，`ForwardTest.GridYOverflowSmoke`，512×128）。

## 3. 2 分钟：优化故事（causal ±2%，负结果）

选的优化：causal 路径跳过整块「未来」KV（`e1735b3`）。`q_last = min(q_start+BLOCK_M-1, seq_len-1)`，`kv_start > q_last` 则 `break`。

Before/after（FP32 causal，hd=64，文档原表）：

| seq_len | before | after | 变化 |
|--------:|-------:|------:|-----:|
| 256 | 0.518 | 0.524 | +1.2% |
| 1024 | 4.63 | 4.59 | −0.9% |
| 4096 | 58.4 | 57.7 | −1.2% |

**结论必须说**：变化在 ±2% 内，低于 10% 阈值，**增益低于噪声**。D 阶段已经有粗 break；这次主要是 clamp `q_last`、避免最后一块 padding 多留 KV。保留是因为语义清楚且数值不回归，**不是因为加速成功**。

本机非 causal FP16 seq=1024：**1.76 ms**；seq=4096 hd=128：**84.1 ms**（`6860cbc`）。对标 FA2 我们慢一档，原因是没有 warp 特化/TMA。

## 4. 2 分钟：验证方法

- 多精度 fwd/bwd 与 CPU/参考差分。
- 非整 tile `seq_len=257`：`ForwardTest.CausalNonTileAlignedSeqLen`。
- FlashDecoding：`ChunkCountInvariant`（不同 chunk 数结果不变）。
- 负结果写入 `docs/performance/causal-boundary-skip.md`，不改写成加速。
- pytorch 对比本次 skip，不声称「和 SDPA 全矩阵对过」。

## 5. 1 分钟：短板与下一步

短板：教学吞吐；反向长序列误差压测未做；双缓冲/`cp.async`、warp softmax 未做。下一步冻结外：真要提速再开 TMA/warp-specialization，而不是再堆一个 ±2% 的 skip。

## 6. 追问清单

1. 为什么不用物化 S？ → HBM 与 O(N²) 内存。
2. online softmax 维护什么？ → 行 max `m` 与分母 `l`。
3. 为什么 L 用 FP32？ → 反向重建 softmax。
4. grid.y 65535 是什么 bug？ → CUDA 网格上限；B*H 展平到 x。
5. causal skip 为什么不快？ → 旧 break 已覆盖主路径；±2% 噪声。
6. 和 FA2 差在哪？ → warp 特化、流水、TMA；我们没有。
7. FlashDecoding 解决什么？ → decode 时 Q 短、KV 长，按 KV 分块并行再 reduce。
8. 为什么没接到 tiny-llm？ → owner 分离；runtime 用自己的 decode attention。
9. head_dim=128 为什么更慢？ → 更大 smem，tile 缩小（文档 84.1 ms @4096）。
10. 你会不会说「做了因果优化所以快了」？ → 不会；那是减分句。
