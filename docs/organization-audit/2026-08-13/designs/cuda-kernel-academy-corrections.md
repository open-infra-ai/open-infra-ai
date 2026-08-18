# `cuda-kernel-academy` 教学正确性修复设计

## 1. 目标

让 Academy 的核心教学示例满足三条底线：

1. 示例能够合法启动并产生正确结果。
2. 优化名称与代码实际使用的机制一致。
3. CI 至少能编译公开 CUDA 表面，真实 GPU 数值状态单独报告。

本轮不新增模块，不把 placeholder 补成完整 Hopper/Blackwell library，也不追求性能纪录。

## 2. CKA-001：TensorCraft FlashAttention

### 2.1 最小正确实现方向

当前 64×32 threads 必须重构，不能只把 2048 截断为 1024 而忽略 load/index 依赖。

推荐教学配置：

```text
BLOCK_M = 8
BLOCK_N = 32
HEAD_DIM = 64
block = dim3(32, 8)     # 256 threads
```

共享内存 load 改成基于线性 `tid` 的协作循环：

```text
for linear = tid; linear < rows * HEAD_DIM; linear += block_threads
```

不能继续用 `ty` 作为 K/V row，因为 `BLOCK_M` 与 `BLOCK_N` 不相等。

分工：

- 每个 `(ty,tx)` 计算一个 QK score。
- 每个 `ty` 对应一个 query row。
- 当前简化版可让同一 row 的 warp 冗余维护完整 output accumulator，以换取易读正确性；文档必须说明冗余。
- 后续优化再把 softmax 和 P×V 做 warp 协作，不能在本任务中同时追求。

### 2.2 输入契约

- 第一版只支持 `head_dim==64`，遇到其他值明确失败。
- 只支持 dense、同 head 数、non-causal；文档与函数名明确。
- batch/heads/seq 必须为正，空输入行为明确。
- launch 后立即检查 `cudaGetLastError`；测试阶段同步检查 runtime error。

### 2.3 测试

把 attention test 加入 `02-tensorcraft-core/tests/CMakeLists.txt`。覆盖：

- `S=1,7,31,32,33,65`。
- batch 1/2、heads 1/3。
- float 与项目实际支持的其他 dtype。
- 与独立 CPU stable-softmax reference 比较。
- 非 64 head dim 被拒绝。
- `cudaFuncGetAttributes`/launch 不超过线程和资源限制。

禁止只断言输出 vector 的 size。

## 3. CKA-002：04 inference ping-pong buffer

### 3.1 数据结构

将单一 `temp_buffer_` 改成两个同容量的中间 buffer：

```text
temp_a
temp_b
```

容量取所有非最终层最大 `batch_size * out_features * sizeof(float)`。如果为了实现简单，两块都按全层最大输入/输出容量分配也可以，但不增加 buffer pool 抽象。

### 3.2 路由规则

```text
current = input
for each layer i:
    destination = output                    if last
                  temp_a                    if first intermediate
                  opposite(previous temp)   otherwise
    assert destination != current
    launch(current, destination)
    current = destination
```

`forward` 与 `forward_with_timing` 必须共享同一 buffer 选择规则。可以抽取一个很小的内部 helper，但不要重构整个 engine。

### 3.3 额外校验

- `layer[i].out_features == layer[i+1].in_features`。
- input/output pointer alias 是否支持应明确；推荐第一版拒绝 caller 传同一 pointer。
- batch size 必须为正。

### 3.4 测试

- 1、2、3、4、5 层。
- 宽度扩大和缩小交替，例如 `7→13→5→11→3`。
- batch 1 和非 2 的幂 batch。
- `forward` 与 `forward_with_timing` 输出完全一致。
- CPU reference 数值比较。
- compute-sanitizer memcheck/racecheck（环境支持时）。

## 4. SGEMM 教学修订

### 4.1 Bank conflict

修订文档应画出“warp 同一时刻”的 bank 映射，而不是沿单个线程的循环方向判断 column access。

对当前 GEMM：

- `As[ty][k]` 是 warp broadcast。
- `Bs[k][tx]` 是连续 bank。
- padding 不是该内层访问正确性的必要条件。

处理方案：

1. 将当前 kernel 改名/描述为 `padded_sgemm`，不声称消除了实测冲突。
2. 另加一个极小 transpose/bank-conflict microbenchmark，真实展示 `[tx][ty]` 的冲突与 `[tx][ty+padding]` 的改善。
3. 用 Nsight Compute shared bank conflict metric 记录证据；没有 profiler 数据时不写 speedup 数字。

### 4.2 Double buffering

当前实现应描述为“ping-pong shared buffers 的结构示例”，不是 load/compute overlap。

二选一：

- 低成本方案：重命名和纠正文档，不声称延迟隐藏。
- 高成本方案：在 Ampere 专属后续任务中实现 `cp.async` pipeline，并保留同步 fallback。

本轮推荐低成本方案。

### 4.3 Tensor Core benchmark 边界

如果 wrapper 内包含 allocation、FP32↔FP16 conversion 和同步，结果必须标为 end-to-end wrapper latency。要比较 kernel，allocation/conversion 应移到 benchmark loop 外。

## 5. 03 高级模块收口

### 5.1 SoftwarePipeline GEMM

在重新设计线程 tile 前，先从 README 性能阶梯和默认 benchmark 中移除该选项，标记为 experimental/known incorrect。不要静默 fallback 到另一个实现并继续显示 `SoftwarePipeline` 名称。

后续独立设计至少写清：

- CTA tile 与 warp tile。
- 每线程 accumulator tile。
- cooperative global-to-shared load。
- pipeline stage 的生产/消费同步。
- shared/register 索引上界证明。

验收必须包含非对称小矩阵和 compute-sanitizer。

### 5.2 TopK

第一阶段明确限制 `n<=1024` 并在 host launcher 拒绝更大输入。不要悄悄只处理前 1024 项。真正 large-N TopK 作为独立多 block merge 任务。

### 5.3 03 FlashAttention

- 固定 `head_dim=64` 时必须拒绝其他配置。
- 测试比较数值，不只比较 size。
- CUDA13 placeholders 继续保持明确标记，不纳入性能表。

## 6. CI 最小闭环

### 6.1 CPU lane

- pre-commit/format。
- docs build 和链接检查。
- CMake preset schema。

### 6.2 CUDA compile lane

使用单架构 sm_80：

- 配置不需要 GPU driver，只需要 toolkit。
- 编译 public header smoke，确保 attention 等模板被真正实例化。
- 编译 02、03、04 最小 targets；允许关闭 benchmark/Python。
- 网络依赖使用固定 commit/cache。

仅编译 header-only interface target 不够，因为不会实例化出错模板。

### 6.3 GPU lane

- 01 SGEMM 数值。
- 02 attention 数值和 launch。
- 03 选定稳定路径。
- 04 多层 oracle。
- compute-sanitizer。

没有 GPU runner 时独立显示未验证。

## 7. 文档与状态标签

模块采用：

- 01：`tutorial`。
- 02：`educational reference API`。
- 03：`experimental`，逐路径标注 verified/placeholder。
- 04：`tutorial system integration`。

删除“工业级/生产级”。性能表必须标注 commit、GPU、来源和验证日期；无法追溯的数字移到 historical/unverified 或删除。

## 8. 任务拆分

| 任务 | 内容 | 验收重点 |
|---|---|---|
| CKA-001A | 合法 FlashAttention launch/load | CUDA launch + 小 shape oracle |
| CKA-001B | attention 测试接入 CMake | CI target 真正实例化 |
| CKA-002 | 04 ping-pong buffers | 3–5 层 CPU 差分 |
| CKA-103A | bank conflict errata | warp 映射准确，无虚构数字 |
| CKA-103B | double buffer 更名/说明 | 不再声称同步代码 overlap |
| CKA-104 | 禁用错误 software pipeline 表面 | 默认路径不可选错误实现 |
| CKA-105 | TopK n 上界 | n>1024 明确失败 |
| CKA-CI | sm_80 compile lane | 真实实例化 public kernels |

每项独立提交，不能在教学文档任务中顺手重写 kernel。

