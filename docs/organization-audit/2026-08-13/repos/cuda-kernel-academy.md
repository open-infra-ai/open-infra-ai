# `cuda-kernel-academy` 仓库审计

> 归档说明：本审计完成于仓库改名 cuda-foundations 之前，文件名保留历史名称。

## 1. 定位判断

Academy 作为五仓总入口和 CUDA 系统教学仓的方向是合理的。四个模块形成了从单 kernel 到小型系统的学习阶梯，README 对历史重复仓库的收敛也降低了维护面。

但教学仓的第一质量标准不是“功能多”，而是概念与实现不能误导。当前存在非法 kernel launch、多层 buffer alias，以及 bank conflict/double buffering 叙述与实际代码不符。应优先发布一轮 correctness/errata 修订，而不是新增模块。

## 2. 模块判断

| 模块 | 合理角色 | 当前判断 |
|---|---|---|
| `01-sgemm-tutorial` | SGEMM 优化阶梯 | kernel 有实质，但部分优化解释和 benchmark 边界不准确 |
| `02-tensorcraft-core` | reusable API 组织方式 | “生产级”措辞过强，attention 有致命 launch 配置 |
| `03-hpc-advanced` | 高级/实验主题 | 范围很宽，部分实现是 placeholder 或缺少数值验证 |
| `04-inference-engine` | kernel/内存/流组装示例 | 定位诚实，但三层以上 forward 有 buffer alias |

## 3. 值得保留的资产

### 3.1 学习入口和边界说明清楚

根 README 明确四模块职责，LEARNING_PATH 负责五仓导航。03 README 也明确 CUDA 13 的 TMA、cluster、FP8 主要是教学占位，而非真实 Hopper/Blackwell 完整封装。

### 3.2 01 模块有连续优化实现

naive、tiled、padding、double buffer、WMMA 与 cuBLAS reference 形成了可阅读的演进结构。核心问题在解释和测量，而不是所有实现都是空壳。

### 3.3 04 模块边界比名称更诚实

README 明确它是多层线性网络和基础 GPU plumbing，不声称真实 LLM。Tensor、MemoryPool、StreamManager、配置、日志和 benchmark 能展示 kernel 如何进入小系统。

### 3.4 根构建图已经收敛

02、03、04、common 和 examples 进入根 CMake；01 保持独立 Makefile。04 对 TensorCraft target 的依赖也由根构建显式保证。这比多个历史仓库各自漂移更好维护。

## 4. P0 正确性问题

### CKA-001：TensorCraft FlashAttention launch 非法

launcher 固定：

```cpp
HEAD_DIM = 64
BLOCK_M = 32
dim3 block(HEAD_DIM, BLOCK_M)
```

即 64×32=2048 threads/block，超过 CUDA 每 block 1024 threads 的上限：[`attention.hpp`](../../../../cuda-kernel-academy/02-tensorcraft-core/include/tensorcraft/kernels/attention.hpp#L387)。

此外传入的 `head_dim` 被忽略并固定为 64。02 的测试目标只包含 elementwise、softmax、normalization、GEMM，没有 attention：[`tests/CMakeLists.txt`](../../../../cuda-kernel-academy/02-tensorcraft-core/tests/CMakeLists.txt#L1)，所以这个问题没有进入自动验证。

### CKA-002：04 多层 forward 缓冲区别名

InferenceEngine 只分配一个 `temp_buffer_`。第一层写入 temp 后，如果总层数至少为 3，下一层的 `current_input` 与 `current_output` 都会指向 temp：[`inference_engine.cpp`](../../../../cuda-kernel-academy/04-inference-engine/src/inference_engine.cpp#L151)。

GEMM 不是可安全原地执行的逐元素算子，线程写输出时会覆盖其他线程尚未读取的输入。`forward_with_timing` 有同类路径。

仓库已有三层 CPU oracle 测试：[`test_inference.cpp`](../../../../cuda-kernel-academy/04-inference-engine/tests/test_inference.cpp#L98)，但组织 CI 不执行 GPU 测试，因此没有形成保护。

## 5. 教学正确性问题

### CKA-103：bank conflict 示例解释了不存在于该 warp 访问中的冲突

示例把 `Bs[k][tx]` 称为 column access，并说 padding 消除冲突：[`bank_conflict_free_sgemm.cuh`](../../../../cuda-kernel-academy/01-sgemm-tutorial/src/kernels/bank_conflict_free_sgemm.cuh#L84)。

对 32×32 thread block，warp 内通常 `ty` 固定、`tx=0..31`：

- `As[ty][k]`：所有线程读取同一地址，是 shared-memory broadcast。
- `Bs[k][tx]`：同一行连续 32 个 float，落在不同 bank。

因此这段 GEMM 内层访问本来就没有注释描述的 32-way conflict。Padding 对一般转置/column-by-warp 场景有价值，但这里没有展示那个场景。

### CKA-103：double buffering 没有真正 overlap

代码先由线程同步执行普通 global load，再执行 compute：[`double_buffer_sgemm.cuh`](../../../../cuda-kernel-academy/01-sgemm-tutorial/src/kernels/double_buffer_sgemm.cuh#L94)。两段指令在同一线程顺序执行，没有 `cp.async`、异步 copy pipeline 或专门 producer warp。

双 shared buffer 只改变了存储轮换，不足以证明“同时加载与计算”。文档应称为 ping-pong buffering 的结构准备，或真正实现异步 pipeline 后再声称 overlap。

### CKA-104：03 software pipeline 线程映射错误

256 threads 下：

```text
thread_row = tid / 64    -> 0..3
thread_col = tid % 64    -> 0..63
```

每线程又处理 4×4，导致：

- M 方向只覆盖 16 行，而 tile 声称 64 行。
- N shared index 可达到 255，但 shared tile 只有约 64 列。
- block 可能写到相邻 N tile，产生越界读取和跨 block 竞争。

见 [`gemm.cu`](../../../../cuda-kernel-academy/03-hpc-advanced/src/03_gemm/gemm.cu#L523)。该优化路径缺少定向数值和 sanitizer 覆盖。

## 6. 其他 P1/P2 问题

### 6.1 CI 不编译 CUDA

根 CI 只有 pre-commit、docs build 和 `cmake --list-presets`：[`ci.yml`](../../../../cuda-kernel-academy/.github/workflows/ci.yml#L71)。它不能发现：

- 缺失源文件。
- 非法 template/launch 配置。
- 链接问题。
- 数值错误和 sanitizer 错误。

### 6.2 02 “工业级/生产级”措辞不成立

02 README 直接称工业级、生产级：[`README.md`](../../../../cuda-kernel-academy/02-tensorcraft-core/README.md#L1)。但 attention 无测试且不能合法启动，Python binding 还会因预期源文件不存在而跳过。

建议改为“面向 reusable kernel API 的教学 library”。

### 6.3 CUDA feature detection 只看 toolkit，不看 architecture

02 中 TMA/WGMMA/FP8 等宏由 CUDA 版本触发，不能证明目标设备具备对应架构能力。应同时依据 `__CUDA_ARCH__` 或 target capability，并将 unsupported path 明确失败。

### 6.4 03 attention 测试只检查输出长度

FlashAttention test 运行后只断言 host vector size：[`test_flash_attention.cpp`](../../../../cuda-kernel-academy/03-hpc-advanced/tests/attention/test_flash_attention.cpp#L7)，没有任何数值比较。实现又固定 `HEAD_DIM=64`，尽管 config 接受任意 `head_dim`：[`flash_attention.cu`](../../../../cuda-kernel-academy/03-hpc-advanced/src/05_attention/flash_attention.cu#L115)。

### 6.5 03 TopK 对 `n>1024` 不正确

launcher 只启动 `min(n,1024)` 个线程，kernel 也只扫描到 `blockDim.x`，但 public API 没有拒绝更大的 n：[`topk.cu`](../../../../cuda-kernel-academy/03-hpc-advanced/src/05_attention/topk.cu#L63)。

### 6.6 高级实现与名称不匹配

- TensorCoreMMA 当前明确回退到 WMMA。
- Winograd 路径是 implicit GEMM fallback。
- CUDA 13 FP8 GEMM 使用 float naive 计算。

03 README 对 CUDA13 placeholder 已经诚实，但更细的 API/benchmark 文档仍应保持相同表述。

### 6.7 性能数字缺少自测证据

根 ROADMAP 已要求重测 SGEMM/HPC 的 TFLOPS 数字，这是正确措施。在完成前，表格应标记来源或 historical/unverified，不能作为当前硬件实测。

### 6.8 构建与安装边界不完整

- Root preset 默认会拉取 GoogleTest、benchmark、fmt、RapidCheck、CUTLASS 等多个网络依赖。
- `native` preset 实际仍固定 `70;80;86`，名称不准确。
- 根 install 主要安装 common header/docs，却生成 package version 信息；并没有形成四模块可消费 package。

教学仓可以选择不发布 SDK，但应删掉容易误解的 package 暗示。

## 7. 本次验证

- `cmake --list-presets` 成功列出 10 个 configure presets。
- 未执行根 CUDA configure/build/test：默认配置需要多个网络依赖，且当前 GPU 被隔离。
- 没有对任何 Academy kernel 给出现场数值通过结论。

## 8. 推荐顺序

1. 修 CKA-001 attention launch，并加入外部数值 oracle。
2. 修 CKA-002 为 ping-pong buffer，覆盖 3/4/5 层。
3. 禁用或修复 03 software pipeline 与 `n>1024` TopK。
4. 发布教学 errata，重写 bank conflict/double buffering。
5. 将 02 的生产措辞降级为教学/reference。
6. 增加最小 sm_80 CUDA compile lane；真实 GPU numerical 单独运行。
7. 清理未验证性能表和 package/preset 漂移。

详细方案见 [Academy 教学正确性修复设计](../designs/cuda-kernel-academy-corrections.md)。

## 9. 成熟度判断

| 维度 | 判断 |
|---|---|
| 学习路径组织 | 良好 |
| 基础 kernel 内容 | 有实质 |
| 教学准确性 | 需要一轮集中纠错 |
| 高级模块正确性 | 参差，多个路径未验证 |
| CI | 文档/格式有，CUDA 验证不足 |
| 生产可用性 | 不适用 |

