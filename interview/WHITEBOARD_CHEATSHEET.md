# 白板公式卡

> Phase I2。面试被要求手推/画图时的速查：公式 → 3–5 行推导 → 与本仓库数字的对应 → 常见错误。
> 数字以 [`NUMBERS_CARD.md`](NUMBERS_CARD.md) 为准；证据编号见 [`EVIDENCE_MATRIX.md`](EVIDENCE_MATRIX.md)。

---

## 1. Online softmax 递推（含 m/l/O 三行递推）

**公式**（对第 t 个 tile）：
- `m_new = max(m, rowmax(S_t))`
- `l_new = l * exp(m − m_new) + rowsum(exp(S_t − m_new))`
- `O_new = O * exp(m − m_new) + exp(S_t − m_new) · V_t`

**推导（3–5 行）**：
1. 标准 softmax 分母 `l = Σ exp(s_i − m)`，分子 `O = Σ exp(s_i − m) v_i`，输出 `O / l`。
2. 若用旧全局 max `m` 归一化，新 tile 的 `s` 可能更大，旧项必须先乘 `exp(m − m_new)` 才能换到新归一化尺度。
3. 所以三行递推里因子 `exp(m − m_new)` 把旧统计搬到新 max 下，`exp(S_t − m_new)` 归一化新 tile。
4. 全程只流式读 KV，不物化 `S`。

**与本仓库数字对应**：这正是 `cuflash` 前向 kernel 的 online 更新；FP16/BF16 API 把 logsumexp 保持 FP32（v0.5.0 breaking，E6/Q23）。

**常见错误**：忘记给旧分子/分母乘 `exp(m − m_new)`，导致新旧 tile 尺度不一致；或把 `m_new` 只用于当前 tile 而不回溯旧项。

**可在现场打开**：`cuflash/src/forward/flash_attention_forward_typed.cu`；追问见 `interview/QA_BANK.md` Q21。

## 2. FlashAttention 为什么是 O(N) 辅助内存

**公式**：辅助状态规模 = 每行 `O(d)`（O tile）+ 每行 `2`（m、l 标量）× N 行 = `O(N·d)`；完整 `S`（N×N）不出全局内存。

**推导（3–5 行）**：
1. 朴素 attention 要写 `S = QKᵀ` 再 softmax，S 是 N×N，O(N²) 内存。
2. Flash 把 Q/K/V 切成块，`S` 的每块在片上算出、更新 `m/l/O` 后直接丢弃。
3. 需要留在外面的只有每行的 running `m`、`l` 和累计 `O`，都是 N 线性级。
4. 计算量仍是 O(N²·d)，但内存工作集不是 O(N²)。

**与本仓库数字对应**：`cuflash` 不物化 S；`NUMBERS_CARD.md` §8 明确 3030/5118 MiB 这对数字无归档、不可当 FA 显存证据（E22）。

**常见错误**：把“O(N) 辅助内存”说成“O(N) 计算”；或把“O(N) 内存”当成本机实测显存表。

**可在现场打开**：`cuflash` 算法说明与 `benchmarks/bench_flash_attention.cu` 的 LogicalHBM 计数器；追问见 Q22/Q30。

## 3. Attention 标准 vs Flash 的 HBM 读写量

**公式**（单头、seq=N、head_dim=d，精度 b 字节，不计掩码/数据复用常数）：
- 标准：`Q/K/V` 读各 `N·d·b`；`O` 写 `N·d·b`；写 `S`（N²·b）再读回 `S`（N²·b）→ 总 HBM ≈ `4·N·d·b + 2·N²·b`。
- Flash：`Q/K/V` 读各 `N·d·b`；`O` 写 `N·d·b`；辅助 `N` 级；中间 `S` tile 只在片上 → 总 HBM ≈ `4·N·d·b + O(N)`。

**推导（3–5 行）**：
1. 每个量按“元素数 × 字节数”加总。
2. 标准版 `S` 必须写一次进 HBM 再读回做 softmax 归一化与加权，贡献 2·N²。
3. Flash 把 S 的生成、softmax、`×V` 都在 tile 内完成，只有在线 rescale，删掉 2·N² 项。
4. 长序列时 N² 项主导，所以 Flash 的收益随 N 增大而变大。

**与本仓库数字对应**：`cuflash` benchmark 报的是 **LogicalHBM** 模型流量（Q30），本机 FP16 seq=1024 1.76 ms / 4096 84.1 ms（`6860cbc`），对 SDPA 约 0.42×–0.67× 是教学预期差距。

**常见错误**：把 LogicalHBM 说成物理 DRAM 带宽；忘记 Flash 仍要读 K/V 每块多次（这里的 N·d 已按总流量算，不指向任何实测计数器）。

**可在现场打开**：`cuflash/benchmarks/bench_flash_attention.cu`（counter 名 `LogicalHBM GB/s`）；追问见 Q30。

## 4. Roofline 与 arithmetic intensity

**公式**：`intensity I = FLOPs / bytes`；理论峰值 `P = peak_FLOPs`；带宽顶 `β`（GB/s）；若 `I < P / β` 则为 memory-bound。

**推导（3–5 行）**：
1. 画两条斜线：水平线 = 算力顶 `P`，斜线 = 带宽顶 `β·I`。
2. 交叉点 `I* = P / β`；kernel 落在交点左侧（I < I*）→ 受带宽限制。
3. decode 的 M==1 GEMM 每列输出只复用一次权重，FLOPs/bytes 低，倾向带宽墙。
4. 提高 I 的办法是复用（转置权重让 K 向连续读、tile 复用）。

**与本仓库数字对应**：decode 主 GEMM 转置前 `10.0002 → 0.9794 ms`（E15）就是访存形状主导的证据；本机 ncu 不可用，没有正式 roofline 图，只能定性讲（Q8/E28）。

**常见错误**：报“我们打满带宽”；把没有 ncu 的 roofline 说成实测点。

**可在现场打开**：`tiny-llm/docs/performance/results/2026-08-18-decode-optimization.md` 与 `cuda-foundations/docs/en/guides/profiling.md`；追问见 Q8。

## 5. GEMM 分块算术：tile 大小、shared 容量、bank conflict

**公式**：tile `BM×BN` 输出、`BK` 累加维度；shared memory 需求 ≈ `(BM·BK + BN·BK)·b` 字节；一个 warp 打同一 bank 若 stride 为 32 的倍数则冲突。

**推导（3–5 行）**：
1. 每块要 A 的 `BM×BK` 与 B 的 `BN×BK` 两个 tile，放 smem 供整块复用。
2. smem 容量固定（如 48–100 KB），`BM×BK + BN×BK` 越大可复用越多，但 block 数/occupancy 变少。
3. 输出 tile 按 32 线程的整数倍分给 warp；bank 总数 32，访问 stride 32×4B 会打同 bank。
4. 取舍：加 padding 换无冲突，可能换更多指令与占用（本仓的负结果）。

**与本仓库数字对应**：`cuda-foundations` 阶梯 0.58 → 0.92 → 0.66（bank-conflict-free）→ 0.68（double-buffer）→ 1.09（WMMA）→ 5.58（cuBLAS）TFLOPS（E1）。

**常见错误**：宣称“bank-conflict-free 一定更快”；把 double-buffer 说成用了 `cp.async`/TMA（Q7）。

**可在现场打开**：`cuda-foundations/01-sgemm-tutorial/` 各 kernel 与 `docs/en/benchmarks/index.md`；追问见 Q3–Q7。

## 6. Occupancy 粗算

**公式**：`occupancy = active_warps / max_warps_per_SM`；受 min(每 SM 线程上限 / block 线程数, 寄存器上限 / block 用寄存器, smem 上限 / block 用 smem, block 上限) 约束。

**推导（3–5 行）**：
1. 每 SM 同时容纳的 warp 数 = 各资源维度算出的 block 数下限 × block 内 warp 数。
2. 增加 block 内 warp 数（更多线程）会提高每 block 资源占用，可能降低可并行 block 数。
3. 寄存器用得多会 spill；smem 用得多会限制 block 数。
4. 高 occupancy 利于藏延迟，但挤占资源导致 spill 时反而更慢。

**与本仓库数字对应**：本机 ncu 报 `ERR_NVGPUCTRPERM` 没有 occupancy 计数器；替代证据是 `tiny_llm_kernel_bench` 与阶梯相对 TFLOPS（E28/Q2）。

**常见错误**：面试说“occupancy 不够所以慢”却拿不出数字；把高 occupancy 无条件当目标。

**可在现场打开**：`tiny-llm/src/kernel_bench.cpp` 与 `cuda-foundations/docs/en/guides/profiling.md`；追问见 Q2。

## 7. W8A16：量化 / 反量化公式、per-group scale 数量

**公式**：`w = q · scale`（q 为 INT8，scale 为 FP16）；group=128 → scale 数量 = `(K/g) × N`（对 K×N 权重矩阵）。

**推导（3–5 行）**：
1. W8A16 = 权重 INT8 + FP16 scale，激活 FP16。
2. 每 128 个权重共享一个 scale，反量化先乘 scale 再与激活乘。
3. FFMA 在 FP16 激活与（dequant 后）权重上做；INT8 存的是量化后的码。
4. group 越大 scale 越少但量化越粗：省 scale 流量 vs 精度。

**与本仓库数字对应**：tiny-llm 推理走 W8A16、group=128（`tests/test_w8a16_matmul.cu`，Q35）；差分测试 `W8A16MatMulTest.*` 与 `WeightW8A16RoundTripPreservesValues`（E13）。

**常见错误**：说成 GPTQ/AWQ 生产量化栈；把 scale 数记成每行一个；把 W8A16 与 Q4_K_M 混为一谈（Q2/D2）。

**可在现场打开**：`tiny-llm/tests/test_w8a16_matmul.cu` 与 `tests/test_quantization.cpp`；追问见 Q35/Q36。

## 8. GQA：kv_head 公式与 KV cache 节省比

**公式**：`kv_head = q_head / (q_head_per_kv_group)`，记 `Hq/Hkv = group_size`；KV cache 字节按 Hkv 线性，故节省比 = 1/group_size（相对等量 q_head 的 MHA）。

**推导（3–5 行）**：
1. MHA 每 q_head 一份 K/V；GQA 让一组 q_head 共享一份 K/V。
2. 每组共享 K/V，KV 总量降到 Hkv = Hq / group_size。
3. 显存（K+V）与每 token 写入量都按 Hkv 计，所以比例是 1/group_size。
4. 输出头数仍是 Hq，attention 计算不变，只是 K/V 广播。

**与本仓库数字对应**：Qwen2.5-0.5B 是 14 Q head / 2 KV head（14→2，E14），group_size=7；`GQAMappingDecodeMatchesCpuReference`（`fdbabcc`）锁映射。

**常见错误**：说“节省了 group_size 倍的延迟”；把 KV head 数当成 q_head 数去套 KV 公式；说任意 HF 结构都支持（Q37）。

**可在现场打开**：`tiny-llm/tests/test_kernels.cu`（`GQAMappingDecodeMatchesCpuReference`）；追问见 Q37。

## 9. RoPE half-split：rotate_half 公式与 interleaved 的区别

**公式**（half-split / rotate_half，d=head_dim）：
- `new[ i ]               = x[ i ]·cos_i − x[ i + d/2 ]·sin_i`
- `new[ i + d/2 ]          = x[ i ]·sin_i + x[ i + d/2 ]·cos_i`，对 `i ∈ [0, d/2)`
- interleaved 则是 `new[2i] = x[2i]·cos_i − x[2i+1]·sin_i`；`new[2i+1] = x[2i]·sin_i + x[2i+1]·cos_i`（pairwise 交错，与 HF 对不上）。

**推导（3–5 行）**：
1. 频率向量 `freqs` 长度 d/2 要配成 d 维，两种约定决定怎么摆。
2. half-split 把 `freqs` 作为 `[c0..c_{d/2-1}, c0..c_{d/2-1}]`（concat 重复）。
3. interleaved 把 freqs 按 pairwise 插进相邻位置（repeat_interleave）。
4. 两种“看起来都像 RoPE”，但位置编码的值不同，与 HF/HF 系列模型约定必须一致。TRIT-001 抓的正是排列不一致（E4）。

**与本仓库数字对应**：`triton-fused-ops/triton_ops/reference/rmsnorm_rope.py:320-324`（concat 而非 repeat_interleave），修复 commit `b1bcdcb`；测试 `test_compute_rope.py`、`test_rmsnorm_rope.py`。

**常见错误**：把 half-split 说成 interleaved；只和 kernel 内部 reference 比（共模，Q16）；在长上下文里位置 id 与 KV 槽错位。

**可在现场打开**：`triton-fused-ops/triton_ops/reference/rmsnorm_rope.py` 与 `tests/test_compute_rope.py`；追问见 Q16/Q17。

## 10. KV cache 字节数公式

**公式**：`bytes = 2 × L × Hkv × D × S × b`（K 一份 + V 一份；L 层数，Hkv KV head 数，D head_dim，S 序列长度，b 每元素字节）。

**推导（3–5 行）**：
1. 每层每 KV head 存 K 与 V 两个张量，所以 ×2。
2. 每个 head 的每 token 是 D 维向量，token 数 S。
3. 两层循环相乘：L 层 × Hkv 头 × D 维 × S token × 2（K/V）→ 公式。
4. 分页只是把这段连续区切成块并映射，总字节数公式不变（实际读写的还是这些量）。

**与本仓库数字对应**：`NUMBERS_CARD.md` §3 明确 3030 vs 5118 MiB 这对数字**无归档**、不可当实测；分页 vs 连续的正确性证据是逐 token 差分（E20），不是字节表（E22）。

**常见错误**：把这对未归档数字背出来；漏 ×2（K/V）；把 3368 MB（转置副本）说成分页节省。

**可在现场打开**：`tiny-llm/tests/test_ffi.cpp`（`FFITest.PagedKVStrategyMatchesContiguous`）与 `EVIDENCE_MATRIX.md` E22；追问见 Q47。

## 11. TTFT / TPOT / decode tok/s 口径与互相换算

**公式**：
- `TPOT = decode 每 token 平均耗时`；`decode tok/s = 1000 / TPOT(ms)`。
- `TTFT = prefill + 采样出首 token 的时间`；两者相加口径：`总墙钟 ≈ TTFT + (num_tokens−1)·TPOT`。

**推导（3–5 行）**：
1. TTFT 看第一批 prefill 与首 token；TPOT 稳态看后续每 token。
2. 两套工具口径不同：tiny-llm TTFT 含首 token 采样；llama-bench `pp1` 不含（E17）。
3. 直接相除会得到不公平比值，所以要明确工具与“含/不含采样”。
4. tok/s 与 TPOT 互为倒数（单位换算）。

**与本仓库数字对应**：TPOT 6.087 ms / 164.283 tok/s / TTFT 10.567 ms（`f897084`）；llama.cpp tg64 3.7 ms；比值 1.65×（非同量化，D2）。

**常见错误**：把 README 早期 ~4.7× TTFT 当公平比（§8）；把 `--iters 5` 与 `--iters 10` 的报告命令混用。

**可在现场打开**：`tiny-llm/docs/performance/results/2026-08-18-decode-optimization.md` 与 `NUMBERS_CARD.md` §1；追问见 Q43。

## 12. PagedAttention 碎片与 `<5%` 浪费说法的适用条件

**公式**：总浪费 = 外部碎片（空闲但不可分配给该请求的块）+ 内部碎片（最后一块未用槽 ≤ block_size−1 per request）；平均尾部浪费率 = `E[tail] / seq_len`。

**推导（3–5 行）**：
1. 连续预留为 max_seq 分配，小请求浪费 “max_seq − len”。
2. 分页按块分配，浪费被限制在最后一块的内部空槽（< block_size）。
3. “浪费 < 5%”只在某类假设下成立：请求长度分布接近块大小、块数较多、尾部占比小时。
4. 对单条短请求或块很少的场景，尾部浪费占比可以远大于 5%，所以措辞要带适用条件。

**与本仓库数字对应**：paged-serving 用属性测试锁块计数不变量（`used+free==total`，E23），不锁“浪费 <5%”宣传句（Q47）；分页正确性证据是分页 vs 连续 KV 的逐 token 差分（E20）。

**常见错误**：无条件说“分页把显存浪费降到 <5%”；把 3030/5118 当证据（E22）；把内部碎片与外部碎片混为一谈。

**可在现场打开**：`paged-serving/src/kv_cache.rs`（`prop_block_count_invariant`）与 `paged-serving/src/scheduler.rs`；追问见 Q47/Q50。
