# `triton-fused-ops` 仓库审计

## 1. 定位判断

仓库收敛到三条 Transformer 算子路径是正确选择：

- fused RMSNorm + RoPE。
- fused gated MLP。
- FlashAttention forward。

删除名不副实的 FP8 路径，强调 reference 和验证方法，也符合学习仓定位。当前不需要新增算子；先修复 RoPE 契约并让 CI、benchmark、autotuner 与现有三条路径真正闭合。

## 2. 当前结构

```text
triton_ops/
  kernels/       Triton kernels + Python wrappers
  reference/     NumPy/PyTorch reference
  validation.py  shape/device/dtype contracts
  autotuner/     generic search/cache/metrics
  benchmark/     generic correctness/performance report
tests/
examples/
```

### 已有合理部分

- FlashAttention reference 在 CPU 上与 PyTorch SDPA 对比：[`tests/test_flash_attention.py`](../../../../triton-fused-ops/tests/test_flash_attention.py#L9)。
- kernel 使用显式 strides，tail block 有 `S=11/65` 覆盖。
- Gated MLP wrapper 和实现采用标准 `activation(gate) * up` 公式：[`triton_ops/kernels/gated_mlp.py`](../../../../triton-fused-ops/triton_ops/kernels/gated_mlp.py#L153)。
- validation 和自定义异常比直接依赖 Triton crash 更适合作为公共 API。
- README 清楚说明 GPU case 会 skip，不把未运行说成通过。

## 3. P0：RoPE 契约不一致

### 3.1 Kernel 使用 half-split

kernel 把每个 head 切成前后两半：

```text
x1 = x[..., :D/2]
x2 = x[..., D/2:]
out1 = x1*cos - x2*sin
out2 = x1*sin + x2*cos
```

见 [`triton_ops/kernels/rmsnorm_rope.py`](../../../../triton-fused-ops/triton_ops/kernels/rmsnorm_rope.py#L205)。Reference 使用同样逻辑：[`triton_ops/reference/rmsnorm_rope.py`](../../../../triton-fused-ops/triton_ops/reference/rmsnorm_rope.py#L152)。

### 3.2 Helper 生成 interleaved 排列

`compute_rope_frequencies` 从 `D/2` 个频率出发，用 `repeat`/`repeat_interleave(2)` 扩展为 D：[`triton_ops/reference/rmsnorm_rope.py`](../../../../triton-fused-ops/triton_ops/reference/rmsnorm_rope.py#L286)。

对于 half-split `rotate_half`，完整 cache 应是：

```text
concat(freqs, freqs)
```

而不是：

```text
repeat_interleave(freqs, 2)
```

后一种排列对应相邻偶奇 pair 的旋转约定。

### 3.3 示例又采用第三种接口

公共 wrapper 声明 cos/sin 是 `[seq_len, head_dim]`，并据此推导 head 数：[`triton_ops/kernels/rmsnorm_rope.py`](../../../../triton-fused-ops/triton_ops/kernels/rmsnorm_rope.py#L242)。示例创建的却是 `[seq_len, head_dim/2]`：[`examples/rmsnorm_rope_example.py`](../../../../triton-fused-ops/examples/rmsnorm_rope_example.py#L74)，示例 reference 又使用 interleaved pair。

因此 API、helper、kernel、reference 和 example 不能同时正确。

### 3.4 为什么现有测试发现不了

GPU 测试使用随机 cos/sin，并比较 kernel 与项目内相同 half-split reference：[`tests/test_rmsnorm_rope.py`](../../../../triton-fused-ops/tests/test_rmsnorm_rope.py#L21)。两者共享同一约定，因此会共同通过。

需要由外部 PyTorch `rotate_half` 公式生成固定 golden vector，而不是继续增加同类随机测试。

## 4. P1 工程问题

### TRIT-101：autotuner 没有接入真实算子

`TritonAutoTuner` 把 config 字典作为关键字参数传给 `kernel_fn`：[`triton_ops/autotuner/tuner.py`](../../../../triton-fused-ops/triton_ops/autotuner/tuner.py#L43)。配置包含 `BLOCK_SIZE/BLOCK_M/BLOCK_N/BLOCK_K/num_warps/num_stages`。

但公共 `fused_rmsnorm_rope` 和 `fused_gated_mlp` wrapper 不接受这些参数，而是在内部硬编码 block size。集成测试只传 dummy kernel：[`tests/test_autotuner.py`](../../../../triton-fused-ops/tests/test_autotuner.py#L131)。

当前“autotuner 基础设施已存在，只缺一次真实执行”的 ROADMAP 判断不准确；它还缺真实 kernel 适配层。

### TRIT-102：benchmark reference 默认走 CPU

`BenchmarkSuite.benchmark_rmsnorm_rope` 把 CUDA tensor 直接传给 reference，却没有指定 `backend="cuda"`：[`triton_ops/benchmark/suite.py`](../../../../triton-fused-ops/triton_ops/benchmark/suite.py#L285)。Reference 默认 backend 是 CPU，会把 tensor 搬到 CPU/NumPy；correctness verifier 又预期两个 `torch.Tensor`。

Gated MLP 的 closure 同样没有显式 CUDA backend。当前 benchmark suite 不能视为已接通。

### TRIT-103：没有 CI

仓库没有 `.github/workflows`。因此 README 中列出的 ruff、mypy、pytest 和 build 没有自动守护。最小 CPU CI 就能覆盖 reference、validation、格式、类型和 packaging；GPU lane 再单独处理。

### TRIT-104：property test 规模不受控

- RMSNorm/RoPE 随机到 `S=8192,C=8192`，100 例。
- Gated MLP dimension test 可达到 batch 64、intermediate 22528、hidden 4096，50 例。

在 GPU CI 上这会造成巨大显存、编译变体和运行时间，不适合作为每次 PR 的 property 范围。应把小形状 property 与少量真实形状 smoke 分离。

### TRIT-105：显式 `num_heads` 缺少一致性验证

未传时验证 `hidden_dim % head_dim == 0`；显式传入后没有验证 `num_heads * head_dim == hidden_dim`：[`triton_ops/validation.py`](../../../../triton-fused-ops/triton_ops/validation.py#L203)。错误配置可能只写输出的一部分或越过逻辑边界。

### TRIT-106：empty fast path 不可达

wrapper 先调用 positive-dimension validation，再检查 `batch/seq/hidden == 0` 返回 empty。若 validator 拒绝 0，则后续分支只是死代码。应选择“支持 empty”或“拒绝 empty”一种契约并测试。

## 5. FlashAttention 边界

当前 FlashAttention：

- 输入是同形状 `[B,H,S,D]`。
- 支持 causal/non-causal forward。
- 没有 backward、GQA、varlen、cross attention 或 dropout。
- kernel GPU 测试只覆盖 FP16、D=16、S=11/65：[`tests/test_flash_attention.py`](../../../../triton-fused-ops/tests/test_flash_attention.py#L31)。

这对于精简教学仓可以接受，但不能从两组 shape 推导广泛 dtype/head_dim 支持。

## 6. 依赖与版本

- `pyproject.toml` 仅给 Torch/Triton 宽下界，没有已验证组合 lock。
- 当前源码版本 `2.0.0`，本地最新 tag 为 `v1.0.0`。
- metadata 中仍使用示例邮箱。

建议 CI 锁定一套已验证 Torch/Triton/CUDA 组合，同时保留 package 的合理版本范围。

## 7. 本次验证

- `python3 -m compileall` 对 `triton_ops`、tests、examples 通过。
- 当前环境未安装 Torch、Triton、pytest、Hypothesis、ruff、mypy。
- GPU 被系统隔离。

因此本次不能确认任何 Triton kernel 的编译或数值结果。

## 8. 推荐顺序

1. 选择 half-split RoPE 唯一契约并引入外部 golden fixture。
2. 同步 helper、reference、wrapper、validation、example 和文档。
3. 建立 CPU CI。
4. 修 benchmark 的 CUDA reference 路径。
5. 让一个真实 kernel 能被 autotuner 调参，或删除未兑现的公共能力。
6. 缩小 PR property 范围，真实大 shape 放 nightly/manual。
7. 再跑一次真实 GPU correctness；不新增第四个算子。

详细方案见 [RoPE 与验证修复设计](../designs/triton-fused-ops-correctness.md)。

## 9. 成熟度判断

| 维度 | 判断 |
|---|---|
| 范围控制 | 良好 |
| Gated MLP/Flash reference 思路 | 合理 |
| RoPE 正确性 | 当前不成立 |
| Autotuner/benchmark | 尚未接通真实路径 |
| CI | 缺失 |
| GPU 证据 | 当前未验证 |
| 生产可用性 | 不适用 |
