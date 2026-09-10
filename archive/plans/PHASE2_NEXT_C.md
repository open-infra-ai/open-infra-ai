# PHASE 2 下一批任务（Batch 3：C0 → C3，tiny-llm decode 性能攻坚）

> **生成时间**：2026-08-18（A1–B5 完成后）
> **执行方**：DeepSeek Flash / 其他低成本模型
> **上游文档**：`PHASE2_PLAN.md` 第 6 节（本批是其 C 阶段的细化与证据更新）
> **状态**：A1–B5 ✅ 全部完成，五仓 clean、ahead 0、cuda-foundations 改名完成。
>
> 本批唯一目标：把 tiny-llm decode 的深度锚点做实——**TPOT 从 20.9ms 降到 ≤8ms（保守 ≤12ms）**，并且 greedy 输出与优化前逐 token 一致。

---

## 0. 本批前置事实（人工已实测，执行模型不要重复验证这些环境结论）

1. **硬件**：RTX 3060 Laptop 6GB；CUDA 12.x；驱动 591.44。
2. **`ncu` 不可用**：`ERR_NVGPUCTRPERM`（WSL2 无性能计数器权限）。**不要尝试跑 ncu**。
3. **`nsys stats` 不可用**：importer 缺失，qdstrm 无法转报告。**不要依赖 nsys 做瓶颈分析**。
4. **性能分析工具改用仓库内微基准**（本批 C0 会新建 `tiny_llm_kernel_bench`）。
5. 当前基线（本机实测，2026-08-18）：
   - `tiny_llm_bench --graphs`（CUDA Graphs ON）：**TPOT 20.9 ms**（47.9 tok/s）；
   - `tiny_llm_bench`（graphs OFF）：**TPOT 91.8 ms**（10.9 tok/s，32 token 序列）；
   - llama.cpp 对比基线：**TPOT 3.7 ms**（272 tok/s）。
6. 已定位瓶颈（人工微基准，数值供验收参考，允许 ±50% 波动）：
   - `fp16_matmul` lm_head（N=151936, K=896, M=1）：**≈9.5 ms/step**（约 45%）
   - 24 层 W8A16 GEMM：w1/w3（N=4864）≈0.15 ms ×2，w2（N=896,K=4864）≈0.15 ms，wq/wo ≈0.046 ms ×2，wk/wv ≈0.019 ms ×2 —— 合计 **≈10.9 ms**（约 50%）
   - attention_decode、rmsnorm、RoPE、add、silu_mul 等合计 **≈2 ms**
7. **根因**：`w8a16_matmul_m1_kernel` / `fp16_matmul_m1_kernel` 中 lane=k、weight 按 `weight[k*N+col]` 读取，跨 lane 地址 stride=N×2B，完全不 coalesced。
8. **人工原型已验证**：weight 转置为 `[N,K]` 后同样 warp-per-column kernel：
   - lm_head：9.5 → **1.06 ms**
   - w1/w3（N=4864）：0.15 → **0.054 ms**
   - wq/wo（N=896）：0.046 → **0.019 ms**
   - wk/wv（N=128）：0.019 → **0.014 ms**

---

## 任务 C0：新建 `tiny_llm_kernel_bench` 微基准（P0，证据工具）

**目标**：让"每个 kernel 花多少时间"可重复测量，替代不可用的 ncu/nsys stats。

**改动文件**：新增 `src/kernel_bench.cpp`（或 `src/kernel_bench.cu`，以能 include `kernels/*.cuh` 为准）；修改 `CMakeLists.txt`。

**实现规格**：

1. 可执行文件：`tiny_llm_kernel_bench`，链接 `tiny_llm` 静态库。
2. 测量对象（shape 与真实 Qwen2.5-0.5B decode 一致）：
   | 项 | 调用 | 参数 |
   |---|---|---|
   | W8A16 GEMM | `w8a16_matmul` | M=1, K=896, N ∈ {128, 896, 4864} |
   | W8A16 GEMM down | `w8a16_matmul` | M=1, K=4864, N=896 |
   | FP16 lm_head | `fp16_matmul` | M=1, K=896, N=151936 |
   | attention_decode | `attention_decode` | S ∈ {8, 32, 64, 128}, Hq=14, Hkv=2, D=64 |
   | rmsnorm | `rmsnorm` | batch=1, hidden=896 |
   | RoPE | `apply_rope_inplace` | num_tokens=1, Hq=14, Hkv=2, D=64, pos=0 |
   | add | `add_inplace` | n=896 |
   | silu_mul | `silu_mul_inplace` | n=4864 |
3. 计时方法：
   - 每个项先 warmup 20 次；再测 200 次（lm_head 可 100 次）；
   - 循环前后各一次 `cudaDeviceSynchronize()`，用 `std::chrono::steady_clock` 计算均值 ms；
   - 输出 CSV 行：`<name>,<shape>,<ms>`，便于复制到文档。
4. 数据分配：全部 `cudaMalloc` + `cudaMemset` 即可（数值不必有意义）；`scales` 填 half 0.5（`0x3800`）。
5. **不要**在 kernel_bench 里做任何优化实验代码，只测现有公开 kernel 接口。

**验收**：
```bash
cmake --build build -j$(nproc)
./build/tiny_llm_kernel_bench
# 期望：正常输出上表所有行；数值与本批第 0.6 条参考同数量级（允许 ±50%）
```

**提交**：`perf(bench): add kernel microbenchmark for decode-path evidence`

**完成后必须把输出表格复制到汇报里。**

---

## 任务 C1：M==1 GEMM 转置权重快路径（P0，本批核心）

**目标**：为 decode 路径的 W8A16 与 FP16 GEMM 增加 `[N,K]` 转置权重布局，用 coalesced 访问把 GEMM 总时间降 3–9 倍。

**改动文件**：
- `include/tiny_llm/types.h`（QuantizedWeight 增加转置指针）
- `kernels/transpose_weights.cu` / `kernels/transpose_weights.cuh`（新增）
- `kernels/w8a16_matmul.cu` / `kernels/w8a16_matmul.cuh`
- `src/model_loader.cpp`（上传后构建转置副本 + 释放逻辑）
- `tests/test_w8a16_matmul.cu`、`tests/test_model_loader.cpp`（新增测试）
- `CMakeLists.txt`（新 kernels 文件会被 GLOB 收录，显式 append 更稳）

**实施步骤（严格按顺序）**：

### Step 1：QuantizedWeight 扩展（types.h）

在 `struct QuantizedWeight` 末尾新增两个字段：
```cpp
// 转置布局（[cols, rows]）供 M==1 decode 快路径使用；prefill 仍用 data/scales。
int8_t *data_t = nullptr;   // [cols, rows]
half   *scales_t = nullptr; // [cols, scaleRows()]
```
新增方法：
```cpp
bool hasTransposed() const {
    return data_t != nullptr && scales_t != nullptr;
}
```
并在 `totalBytes()` 中**不要**计入转置副本（保持原语义，注释说明转置副本单独统计）。

### Step 2：转置 kernel（新增文件）

`kernels/transpose_weights.cuh` 声明、`.cu` 实现：
```cpp
// data: [rows, cols] -> data_t: [cols, rows]，按元素并行。
void transpose_int8(const int8_t *src, int8_t *dst, int rows, int cols, cudaStream_t stream);
// scales: [scale_rows, cols] -> scales_t: [cols, scale_rows]
void transpose_scales(const half *src, half *dst, int scale_rows, int cols, cudaStream_t stream);
// fp16 权重转置：src [rows, cols] -> dst [cols, rows]
void transpose_fp16(const half *src, half *dst, int rows, int cols, cudaStream_t stream);
```
kernel 索引：`idx = blockIdx.x*blockDim.x + threadIdx.x;`，`src_row = idx / cols; src_col = idx % cols; dst[src_col*rows + src_row] = src[idx];`（scales 同理）。

### Step 3：model_loader 构建转置副本

在 `src/model_loader.cpp` 中，每处完成 `QuantizedWeight` 上传后（`load_quantized` lambda、`load_qweight` lambda、`loadQuantizedTensor`），立即：
```cpp
CUDA_CHECK(cudaMalloc(&qw.data_t, qw.cols * qw.rows * sizeof(int8_t)));
CUDA_CHECK(cudaMalloc(&qw.scales_t, qw.scaleCols() * qw.scaleRows() * sizeof(half)));
tiny_llm::kernels::transpose_int8(qw.data, qw.data_t, qw.rows, qw.cols, stream);
tiny_llm::kernels::transpose_scales(qw.scales, qw.scales_t, qw.scaleRows(), qw.cols, stream);
```
（`stream` 若该处没有，用默认 stream `0`。）

对 `weights.lm_head_fp16`（FP16，`[hidden_dim, vocab_size]`），新增成员或局部全局指针存储转置副本。**推荐做法**：在 `ModelWeights` 增加 `half *lm_head_fp16_t = nullptr;`（`[vocab_size, hidden_dim]`），model_loader 加载 lm_head_fp16 后分配并 `transpose_fp16(...)`；清理函数同步 `cudaFree`。若不想改 `ModelWeights`，可以把转置指针存在 `ModelLoader` 返回结构之外的全局/成员，**但禁止**用静态局部变量（会泄漏且多实例不安全），最终必须有明确的 free 路径。

free 逻辑：在 `model_loader.cpp` 现有的 `freeWeights`/清理 lambda 中，对每个 `QuantizedWeight` 增加：
```cpp
if (qw.data_t) { cudaFree(qw.data_t); qw.data_t = nullptr; }
if (qw.scales_t) { cudaFree(qw.scales_t); qw.scales_t = nullptr; }
```

### Step 4：新 M==1 转置 kernel 与 dispatch

在 `w8a16_matmul.cu` 新增（可直接采用以下原型，注意保持 C++17 / `__half2float`）：

```cuda
__global__ void w8a16_matmul_m1_transposed_kernel(
    const half *__restrict__ input, const int8_t *__restrict__ weight_t,
    const half *__restrict__ scales_t, half *__restrict__ output,
    int N, int K, int group_size) {
    const int warps_per_block = blockDim.x / 32;
    const int col = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    if (col >= N) return;
    const int lane = threadIdx.x & 31;
    float sum = 0.0f;
    const int scale_rows = (K + group_size - 1) / group_size;
    for (int k = lane; k < K; k += 32) {
        float a = __half2float(input[k]);
        float w = static_cast<float>(weight_t[(size_t)col * K + k]);
        float s = __half2float(scales_t[(size_t)col * scale_rows + (k / group_size)]);
        sum += a * w * s;
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) output[col] = __float2half(sum);
}

__global__ void fp16_matmul_m1_transposed_kernel(
    const half *__restrict__ input, const half *__restrict__ weight_t,
    half *__restrict__ output, int N, int K) {
    const int warps_per_block = blockDim.x / 32;
    const int col = blockIdx.x * warps_per_block + (threadIdx.x / 32);
    if (col >= N) return;
    const int lane = threadIdx.x & 31;
    float sum = 0.0f;
    for (int k = lane; k < K; k += 32) {
        float a = __half2float(input[k]);
        float w = __half2float(weight_t[(size_t)col * K + k]);
        sum += a * w;
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) output[col] = __float2half(sum);
}
```

Dispatch 修改（只改 M==1 分支）：
```cpp
if (M == 1 && weight_t != nullptr && scales_t != nullptr) { ... 转置 kernel，block 128，grid=(N+3)/4 ... }
// 否则保留现有 m1 kernel 作为 fallback
```
`fp16_matmul` 同理，新增参数/重载接收 `weight_t`。**保持旧签名可用**（无转置指针时走旧 kernel），这样测试与调用方可以渐进迁移。

调用点接线：
- `src/inference_engine.cpp` 的 `computeLogits`（W8A16 后备分支）与 `src/execution_common.cpp` 的 helper，传入 `weights.lm_head.data_t / scales_t`；
- `src/transformer.cpp` 的 `attention` / `feedForward`，传入各 `QuantizedWeight.data_t / scales_t`；
- `src/ffi.cpp` 与 `execution_common.cpp` 的 lm_head fp16 分支，传入 `weights.lm_head_fp16_t`。

### Step 5：测试

`tests/test_w8a16_matmul.cu`：
1. `TransposedFastPathMatchesCpuReference`：随机 M=1、K=128、N=1024（以及 N=896、K=4864 一个 down-proj 形状），`data_t = transpose(data)`，分别调旧 kernel 与新 kernel，两者与 CPU 参考误差均 `≤1e-1`，且彼此 `≤1e-2`。
2. `TransposedFastPathFallsBackWithoutBuffers`：不传 `data_t` 时行为与旧路径一致。

`tests/test_model_loader.cpp`：
3. 加载最小 GGUF fixture 后断言每个 `QuantizedWeight.hasTransposed()` 为 true，且抽查 `data_t[i*rows+j] == data[j*cols+i]`（D2H 比较一个 8×4 子块即可）。

### Step 6：端到端正确性 + 性能

```bash
cmake --build build -j$(nproc)
./build/tiny_llm_tests                      # 期望 158+ 新测试全过，skip 数不变
./build/tiny_llm_kernel_bench               # 记录转置前后对比表
./build/tiny_llm_demo ../models/qwen2.5-0.5b-instruct-q4_k_m.gguf --prompt "你好" --max-tokens 32 --show-tokens > /tmp/c1_after.txt
TLLM_CUDA_GRAPHS=0 ./build/tiny_llm_demo ../models/qwen2.5-0.5b-instruct-q4_k_m.gguf --prompt "你好" --max-tokens 32 --show-tokens > /tmp/c1_after0.txt
# 与优化前的输出做 diff：逐 token 必须完全一致（先取优化前基线再改代码）
./build/tiny_llm_bench ../models/qwen2.5-0.5b-instruct-q4_k_m.gguf --prompt "你好" --max-tokens 64 --warmup 3 --iters 5 --graphs
```

**验收（量化门槛）**：
- 所有测试全绿；
- `tiny_llm_kernel_bench` 中 `N=4864` W8A16 项 **≤0.08 ms**；`lm_head` FP16 项 **≤2.0 ms**（人工原型为 0.054 / 1.06 ms）；
- `--graphs` TPOT **≤12 ms**；
- graphs 开/关两次 `tiny_llm_demo` 输出逐 token 一致，且与 C1 改动前输出一致。

**提交**：`perf(kernel): transposed-weight M==1 GEMM fast path for decode`

**若 TPOT 未 ≤12ms**：保留改动（只要单项 kernel 达标），把 microbench 表和 bench 输出放进报告，等待下一步指令，不要自己再叠优化。

---

## 任务 C2：按 C1 结果决定（分支任务）

### 分支 A（C1 TPOT ≤ 12ms）：CUDA Graphs 改为默认开启

1. `src/inference_engine.cpp`：`TLLM_CUDA_GRAPHS` 环境变量语义改为 **默认开启、`=0` 关闭**（现在的判断逻辑反转；日志仍打印当前状态）。
2. `src/benchmark.cpp`：`--graphs` 参数改为“显式禁用诊断”或保留但输出里注明默认已开启；`tiny_llm_bench` 默认跑 graphs 路径。
3. 更新测试/文档中的环境变量说明；跑差分：
   ```bash
   TLLM_CUDA_GRAPHS=0 ./build/tiny_llm_demo ... > /tmp/g0.txt
   ./build/tiny_llm_demo ... > /tmp/g1.txt      # 默认即 graphs
   diff /tmp/g0.txt /tmp/g1.txt && echo IDENTICAL
   ```
4. 若 graphs 捕获失败仍保留 fallback 逻辑。

**验收**：默认（不设环境变量）与 `TLLM_CUDA_GRAPHS=0` 输出一致；`tiny_llm_bench` 默认 TPOT 与 C1 记录一致。

**提交**：`perf(runtime): enable CUDA Graphs decode by default with opt-out`

### 分支 B（C1 TPOT 仍 >12ms）：按 microbench 数据优化下一个最大项

只允许做以下三选一（其余不动）：
1. 对转置 kernel 做 `half2` 向量化输入加载（`k = lane*2`，一次 `__half2` 读两个输入，每 warp 覆盖 64 个 k）；
2. 若 microbench 显示 attention 投影仍占 >15%，给转置 W8A16 kernel 增加 `__ldg` / `__restrict__` 并合并 wk+wv 为一次 launch（N=256）；
3. 若 lm_head 仍 >3ms，把 grid 中 `WARPS_PER_BLOCK` 从 4 调到 8 并做 2 的幂 sweep（4/8/16），选最优，记录 sweep 数据。

**验收**：每项改动前后 microbench + bench 表格；TPOT 进一步下降且输出不回归。

**提交**：`perf(kernel): <具体优化名>`

---

## 任务 C3：性能文档与面试叙事收口

**改动文件**：
- `docs/performance/results/2026-08-18-decode-optimization.md`（新增）
- `README.md`（基准快照表更新）
- `docs/performance/benchmark-methodology.md`（工具说明：ncu/nsys 在本机不可用的替代方案）
- `ROADMAP.md`（阶段 2/3 勾选与数字更新）

**文档必须包含**：
1. before/after 表：TPOT、tok/s、microbench 各 kernel 表、llama.cpp 比值；
2. 根因图（一行即可）：`M==1 kernel 的 lane=k 映射 × [K,N] 布局 → stride-N 访存 → 转置 [N,K] 后 coalesced`；
3. 为什么不做 ncu/nsys：`ERR_NVGPUCTRPERM` + importer 缺失，用仓库内 microbench 替代；
4. CUDA Graphs 默认开启后的对比；
5. "下一步"（诚实写）：Tensor Core WMMA、KV paged、FlashDecoding 集成等未做项。

**验收**：README 新表数字与 C1/C2 实测一致，且有复现命令。

**提交**：`docs(perf): decode optimization report and updated benchmark snapshot`

---

## 本批完成后的下一步

C0–C3 完成后停下汇报：microbench 前后表、TPOT 前后数字、commit hash、greedy 输出 diff 结论、遗留 NOTE。

下一步将进入 `PHASE2_PLAN.md` 第 7 节 **D 阶段（分页 KV 端到端）**，任务明细已在该文档 D1–D5，届时按 ABI v2 顺序下发。
