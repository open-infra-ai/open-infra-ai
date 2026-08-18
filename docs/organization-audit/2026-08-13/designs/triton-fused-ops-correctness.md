# `triton-fused-ops` RoPE 与验证修复设计

## 1. 目标

统一 RMSNorm+RoPE 的数学、cache 形状、API、reference、example 和 test，并让 benchmark/autotuner 至少对一个真实 kernel 闭合。

## 2. 唯一 RoPE 契约

公共 API 采用完整 cache：

```text
x:      [B,S,C]
weight: [C]
cos:    [S,D]
sin:    [S,D]
H = C / D
```

数学采用 half-split：

```python
x1 = x_head[..., :D//2]
x2 = x_head[..., D//2:]
y1 = x1 * cos[..., :D//2] - x2 * sin[..., :D//2]
y2 = x1 * sin[..., :D//2] + x2 * cos[..., :D//2]
```

频率 helper 必须生成：

```python
freqs = outer(position, inv_freq)       # [S,D/2]
emb = concat([freqs, freqs], dim=-1)    # [S,D]
cos, sin = emb.cos(), emb.sin()
```

禁止 `repeat_interleave(2)`。

如果未来需要支持 interleaved-pair，必须作为不同明确 convention 参数和独立 kernel，而不是自动猜测输入。

## 3. Validation

补充：

- `D` 必须为正偶数。
- `C % D == 0`。
- 若显式传 `num_heads`，必须满足 `num_heads * D == C`。
- cos/sin shape、dtype、device、contiguous contract 一致。
- 4D cache 只接受文档声明的一种形状；squeeze 后再次验证。

Empty tensor 二选一：

- 推荐支持 `B==0` 或 `S==0`，在 positive dimension validator 前处理，并返回正确 shape。
- `C==0`/`D==0` 仍拒绝，因为无法定义 head。

不要保留不可达 empty branch。

## 4. 外部 golden fixture

新增一个不 import `triton_ops.reference` 的生成脚本，使用纯 PyTorch 明确公式生成：

- `B=2,S=5,H=3,D=8`。
- position 包含 0 和非零。
- weight 非全 1。
- FP32 golden，同时保存 FP16 input。

测试层次：

1. `compute_rope_frequencies` 对 golden cache。
2. CPU NumPy reference 对 golden output。
3. CUDA PyTorch reference 对 golden output。
4. Triton kernel 对同一 golden output。

任何一层都不以相邻被测层作为唯一 expected。

## 5. Example 修正

示例必须：

- 使用完整 `[S,D]` cos/sin。
- 调用与 kernel 相同的 half-split reference。
- 明确 `hidden_dim = num_heads * head_dim`。
- 比较最大误差，而不是只打印 shape。
- 不重复维护第三份 RoPE 公式；示例可使用公开 external-style helper，但测试仍依赖 fixture。

## 6. Benchmark 修正

给 CUDA reference 显式 wrapper：

```python
def reference_cuda(...):
    return reference_fused_rmsnorm_rope(..., backend="cuda")
```

Gated MLP 同理。`benchmark_kernel` 的双方必须返回同 device 的 torch tensor。加入一个真实 GPU smoke，形状保持很小。

FlashAttention 也应加入 benchmark family，或把 README 中“综合 benchmark”限制为实际覆盖的 RMSNorm/RoPE 与 MLP。

## 7. Autotuner 接入

推荐增加内部 launch adapter，而不是让公共 wrapper 暴露大量调优参数：

```text
_launch_rmsnorm_rope(..., block_size, num_warps, num_stages)
_launch_gated_mlp(..., block_m, block_n, block_k, num_warps, num_stages)
```

- 公共 wrapper 调用一组稳定默认值。
- `TritonAutoTuner` 调用内部 adapter。
- 配置先经过 problem-shape filter。
- 测试必须使用真实算子和至少两个可区分配置，不再只用 dummy kernel。

如果这项成本过高，替代方案是删除 autotuner 的对外宣传和未使用 config spaces。保留不可调用的基础设施不是中间完成状态。

## 8. Property test 分层

PR lane：

- 小 shape，最多 10–20 examples。
- `S<=128,C<=1024`。
- 重点覆盖 tail、奇数 S、多个 H/D。

Nightly/manual：

- 模型真实 shape。
- 大 sequence 和显存压力。
- 编译变体数量受固定矩阵约束。

不要让 Hypothesis 随机生成数十 GB 级中间矩阵。

## 9. CI 设计

CPU required lane：

- ruff format/check。
- mypy。
- CPU reference/validation tests。
- package build + import smoke。

GPU lane：

- 三个 kernel 的小型 external-oracle test。
- 可选 nightly property 和 benchmark。
- 无 GPU 时不能把 GPU check 标为成功。

## 10. 任务拆分与验收

| 任务 | 验收 |
|---|---|
| TRIT-001A | frequency helper 改为 concat；golden cache 通过 |
| TRIT-001B | reference/kernel/example 统一；外部 output fixture 通过 |
| TRIT-105/106 | 显式 heads 和 empty 契约测试通过 |
| TRIT-102 | benchmark CUDA reference smoke 通过 |
| TRIT-101 | 真实算子 autotune 两配置可运行，或删除未兑现表面 |
| TRIT-103 | CPU CI required；GPU 状态独立 |

完成后 README 才能继续声称三条算子具有独立参考实现与差分测试。

