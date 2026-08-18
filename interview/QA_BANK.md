# 面试问答库（60 问）

数字只从 [`NUMBERS_CARD.md`](NUMBERS_CARD.md) 取。声明指针见 [`EVIDENCE_MATRIX.md`](EVIDENCE_MATRIX.md)。禁止把 3030 vs 5118 MiB、causal ±2%「优化成功」、3 并发「生产能力」说出口。

分布：CUDA 基础 Q1–Q10 · Triton Q11–Q20 · FlashAttention Q21–Q32 · tiny-llm Q33–Q46 · serving Q47–Q60。

---

## CUDA 基础（Q1–Q10）

### Q1. CUDA 里 thread、block、warp 各是什么调度单位？
- 一句话答案：thread 是执行单元，block 是可同步的协作组，warp 是 32 线程的 SIMT 调度量子。
- 展开（3–5 点）：
  - SM 以 warp 发射指令；同一 warp 走同一 PC，分歧会串行。
  - block 内可用 `__syncthreads` 和 shared memory；block 之间不能假设顺序。
  - grid 是 block 的集合；`gridDim.y` 有硬件上限（见 Q25）。
  - 教学仓 SGEMM 用 block 对应输出 tile，warp 在 WMMA 层变成矩阵片段。
- 证据：`cuda-foundations/01-sgemm-tutorial/`；`cuflash-attn@d144765`（grid 上限）
- 追问 1：一个 block 必须是 warp 的整数倍吗？ → 硬件按 warp 调度；非 32 倍数的尾部仍占一个 warp，多余 lane 闲置。

### Q2. occupancy 是什么，为什么不是越高越好？
- 一句话答案：occupancy 是 SM 上活跃 warp 相对上限的比例；高 occupancy 能藏延迟，也可能挤掉寄存器/shared 让编译器spill。
- 展开（3–5 点）：
  - 限制因素：寄存器/线程、shared/block、硬件 warp/SM 上限。
  - 本机 `ncu` 报 `ERR_NVGPUCTRPERM`，没有 occupancy 计数器（E28）。
  - 替代证据是阶梯相对 TFLOPS 和 `tiny_llm_kernel_bench`，不是 roofline 实测点。
  - 面试说「我们 occupancy 不够所以慢」而拿不出 ncu，是减分。
- 证据：`tiny-llm/docs/performance/results/2026-08-18-decode-optimization.md`；`cuda-foundations/docs/en/guides/profiling.md`
- 追问 1：没有 ncu 怎么证明 GEMM 是瓶颈？ → microbench：lm_head 10.0002 ms，attention 在 µs 级（E15/E28）。

### Q3. shared memory bank conflict 是什么？你们的「bank-conflict-free」为什么更慢？
- 一句话答案：32 个 bank，同拍多 lane 打同一 bank 会串行；本机 padding 版 0.66 TFLOPS，低于 tiled 的 0.92。
- 展开（3–5 点）：
  - padding 换冲突，也可能换占用、指令数和对齐。
  - 教学仓必须把负结果留在表里，否则学习者以为 padding 总赢。
  - 不要把「写了 bank-conflict-free」说成「一定更快」。
- 证据：E1；`cuda-foundations/docs/en/benchmarks/index.md`（0.66 vs 0.92）
- 追问 1：什么形状冲突最明显？ → 列主或 stride 为 32 倍数的向量化访问；面试以本机表为准，不编 occupancy。

### Q4. shared memory 在 GEMM 里干什么？
- 一句话答案：把 A/B 的 K 向 tile 搬进 SM，让输出 tile 复用，降低 HBM 往返。
- 展开（3–5 点）：
  - naive 每次乘加都打 global，0.58 TFLOPS。
  - tiled 用 smem 后 0.92（E1）。
  - smem 容量限制 tile 大小，也限制 occupancy。
  - FA 的 Q/K/V tile 同样靠 smem；head_dim=128 更慢部分来自更大 smem、更小 tile。
- 证据：E1；cuflash `docs/performance/benchmarks.md`（4096×hd128 FP16 84.1 ms）
- 追问 1：smem 和 L1 是不是同一块？ → Ampere 上可配置；本仓没有测过拆分比例，不报数。

### Q5. SGEMM 阶梯每一步的动机是什么？
- 一句话答案：naive 证正确性 → tiled 证复用 → padding 证 bank → double-buffer 证重叠意图 → WMMA 证 Tensor Core 接口 → cuBLAS 当天花板。
- 展开（3–5 点）：
  - 数字：0.58 / 0.92 / 0.66 / 0.68 / 1.09 / 5.58 TFLOPS（1024³ FP32）。
  - 中间两步相对 tiled **更慢**，这是教材，不是失败隐瞒。
  - 相对 cuBLAS 仍约 0.20×（WMMA 1.09 / 5.58）。
- 证据：E1
- 追问 1：为什么不停在 tiled？ → 要让面试官能追问 Tensor Core 对齐与负优化；阶梯是学习产物。

### Q6. WMMA 对形状有什么对齐要求？
- 一句话答案：本教学路径 `launch_gemm_wmma` 拒绝非 16 倍数，抛 `std::invalid_argument`。
- 展开（3–5 点）：
  - WMMA 片段常见 16×16×16；不对齐就不能走这条 kernel。
  - 默认 CUDA arch 曾静默落到 `sm_52`，WMMA 直接不可用，已修。
  - WMMA 1.09 远低于 cuBLAS：缺算法选择、epilogue、启发式。
- 证据：`cuda-foundations/CHANGELOG.md`（非 16 倍数拒绝；arch 默认修复）
- 追问 1：tiny-llm 的 decode GEMM 用 WMMA 吗？ → 不用。M==1 走转置 INT8 快路径，瓶颈是 coalescing 不是 Tensor Core 教学 API。

### Q7. 你们的 double buffering 用了 `cp.async` 吗？
- 一句话答案：没有。同一线程先同步 global load 再计算，不是异步 copy pipeline。
- 展开（3–5 点）：
  - `cp.async` 是 Ampere 的异步拷贝，需要 producer/consumer 或 pipeline API。
  - 本教程 double-buffer 0.68 TFLOPS，未超过 tiled 0.92。
  - cuflash ROADMAP 仍把双缓冲/`cp.async` 列为未做。
  - 把教学 double-buffer 说成 TMA/warp-specialization，是红灯。
- 证据：`cuda-foundations` double_buffer kernel；`cuflash-attn/ROADMAP.md`；组织审计对同步 load 的说明
- 追问 1：那为什么还留这一级？ → 展示「写了软件流水意图但没有硬件异步」时数字会怎样；负结果是课。

### Q8. 什么是 roofline？你们有实测点吗？
- 一句话答案：把 kernel 放在「算力顶」和「带宽顶」之间看受哪边限；本机没有 ncu 带宽点。
- 展开（3–5 点）：
  - arithmetic intensity = FLOPs / 字节。
  - decode M==1 GEMM 强度低，更像带宽墙；转置是为 coalescing 服务。
  - cuflash 报的是 **LogicalHBM** 模型，不是物理带宽（Q30）。
  - 没有把阶梯 TFLOPS 画成正式 roofline 图，不假装有。
- 证据：E28；cuflash `benchmarks/bench_flash_attention.cu` counter 名 `LogicalHBM GB/s`
- 追问 1：转置之后还在带宽墙上吗？ → microbench 显示仍远慢于理论峰值；端到端还剩 1.65× vs llama.cpp，量化也不同（E15/E17）。

### Q9. kernel launch 开销为什么在 decode 里重要？
- 一句话答案：decode 每 token 一次前向，很多小 kernel；launch 和 host 同步会吃掉 µs 级算子。
- 展开（3–5 点）：
  - tiny-llm 用 CUDA Graphs 默认捕获 decode device 路径（E16）。
  - 小 kernel（rmsnorm/rope/attention_decode）在 10–50 µs，文档标噪声 ±50%，不当主瓶颈叙事。
  - 主时间在 lm_head / W8A16 GEMM（10.0002 ms → 0.9794 ms）。
  - Graphs 不 capture `advanceSeqLen`、logits D2H、采样。
- 证据：E15、E16；`tiny-llm/docs/performance/cuda-graphs.md`
- 追问 1：Graphs 为什么要 device `write_pos`？ → 捕获会固化主机读到的地址；KV 槽必须从 device int 读（Q42）。

### Q10. cuda-foundations 和 tiny-llm 的边界是什么？
- 一句话答案：前者是教学阶梯；后者是真实 GGUF 运行时。runtime 禁止 include 教学仓头文件。
- 展开（3–5 点）：
  - `04-inference-engine` 已降级为教学预览（E26）。
  - 旧 slug `cuda-kernel-academy` 在五仓源码 0 命中（E27）；教学品牌名仍可用。
  - freeze：0 failed / 209 collected / **78 skipped**，不能说 209 全执行（数字卡 §6）。
- 证据：E26、E27；`cuda-foundations/04-inference-engine/README.md`
- 追问 1：面试只讲教学仓够不够？ → 不够。旗舰是 tiny-llm + paged-infer。

---

## Triton（Q11–Q20）

### Q11. Triton 的 program/block 抽象和 CUDA block 什么关系？
- 一句话答案：一个 Triton program 通常对应一个 CUDA block；你写的是 tile 索引，不是 threadIdx。
- 展开（3–5 点）：
  - `tl.arange` 构造 tile；`num_warps`/`num_stages` 是粗调参。
  - 看不到 bank 索引和精确 occupancy。
  - 同题 GEMM 在 cuda-foundations 用手写 smem，在本仓用 Triton（E2）。
- 证据：`triton-fused-ops/triton_ops/kernels/sgemm.py`；E2
- 追问 1：program 能不能小于一个 warp？ → 启动器仍按 block/warp 跑；过小 tile 浪费 lane。本仓不以 occupancy 表说话。

### Q12. 为什么 Triton load/store 必须 mask？
- 一句话答案：tile 会越过 M/N/K；无 mask 会读脏或写越界。
- 展开（3–5 点）：
  - `tests/test_sgemm.py` 含非 2 幂 17×33×65（E2/数字卡 §5）。
  - 失败路径也要测，不只 happy path。
  - FA/RoPE 同理：seq 和 head_dim 尾部。
- 证据：`triton-fused-ops/tests/test_sgemm.py`（24 项，rtol/atol=1e-2）
- 追问 1：mask 会不会让编译变慢？ → 会增加分支；正确性优先。没有单独 mask 微秒表，不报。

### Q13. `tl.dot` 做什么，累加用什么精度？
- 一句话答案：tile 级矩阵乘；练习实现常用 FP32 accumulator，再写回输出 dtype。
- 展开（3–5 点）：
  - 这是 Triton 相对手写 `mma` 的速度来源之一。
  - 差分对 `torch.mm`，容差 1e-2，不是 bitwise。
  - 没有与教学 CUDA SGEMM 的同刻 head-to-head 表（E2）。
- 证据：E2、E3；`tests/test_sgemm.py`
- 追问 1：能当生产 GEMM 吗？ → 不能。练习实现；tiny-llm 推理 GEMM 是 CUDA W8A16。

### Q14. autotuner 在本仓扮演什么角色？
- 一句话答案：基础设施在，没有当成接到每个 wrapper 上的旗舰调参系统。
- 展开（3–5 点）：
  - 面试不把它说成 vLLM 级别 kernel 选择器。
  - 真正的资产是 reference + 差分 + torch.library。
  - 融合也不是默认正确：lm_head 的时间在访存形状（cross-cutting §2）。
- 证据：`triton-fused-ops` README；`interview/cross-cutting.md` §2
- 追问 1：为什么不把每个 op 都 autotune？ → 作品集要讲契约和负结果，不堆配置空间。

### Q15. RMSNorm+RoPE 融合的收益怎么讲？
- 一句话答案：两者都是逐元素、同一行，中间结果不必回 HBM；README **0.104 ms** @ (1,128,4096)。
- 展开（3–5 点）：
  - gated_mlp silu **3.45 ms** @ (1,128,4096,11264)，约 10 TFLOPS vs 卡理论 ~46 TFLOPS FP16。
  - 数字来自 commit `ebf6c32+` 原文栈，面试跟表走。
  - 不要把 3.45 ms 说成打榜。
- 证据：E3；数字卡 §5
- 追问 1：为什么不把 lm_head 也融进 Triton？ → runtime 已是 C++；再套 Python 启动器破坏最小运行时叙事。

### Q16. RoPE half-split 和 interleaved 差在哪？
- 一句话答案：Llama/Qwen 用 `rotate_half` 切最后一维两半；interleaved 是 pairwise 交错，数值「像 RoPE」但和 HF 对不上。
- 展开（3–5 点）：
  - 正确 full cache：`concat([freqs, freqs])`，即 `[c0..c_{D/2-1}, c0..c_{D/2-1}]`。
  - 错误：`repeat_interleave(freqs, 2)`。
  - kernel、reference、example 必须同一契约。
- 证据：E4；`triton_ops/reference/rmsnorm_rope.py`
- 追问 1：只拿 kernel 和自己的 reference 比行不行？ → 不行，会共模；要有外部 half-split 约定。

### Q17. TRIT-001 是怎么发现的？
- 一句话答案：helper / API / example 的 RoPE 排列不一致，审计点名 half-split，不是性能回归。
- 展开（3–5 点）：
  - 修复 commit `triton-fused-ops@b1bcdcb`。
  - 同时处理 Triton 3.x 兼容。
  - 测试：`tests/test_compute_rope.py`、`test_rmsnorm_rope.py`。
- 证据：E4
- 追问 1：这算 production incident 吗？ → 不算。是教学仓契约 bug；价值是「名字像 RoPE ≠ 和 HF 一致」。

### Q18. torch.library 注册了什么？为什么要注册？
- 一句话答案：`triton_ops::sgemm` / `fused_rmsnorm_rope` / `fused_gated_mlp`；为了进 PyTorch 图，而不是只 `import kernel`。
- 展开（3–5 点）：
  - 优先 `torch.library.triton_op`，否则 `custom_op + register_fake`（`1bbf5c8`）。
  - 内部不复制 kernel 逻辑。
  - 只接受 CUDA 张量。
  - 本次 freeze `test_torch_compile_smoke` **skip**，不伪造通过（E5）。
- 证据：E5；`triton_ops/ops.py`
- 追问 1：和 vLLM `torch.ops.vllm.*` 是同一套机制吗？ → 接入模式同类；不是同一仓库、同一 dispatcher 实现。

### Q19. 什么时候用 Triton，什么时候用 CUDA C++？
- 一句话答案：公式还在变、要对着 PyTorch、要进 compile 图 → Triton；访存/网格/反向稳定性已用 microbench 钉死 → CUDA C++。
- 展开（3–5 点）：
  - 同题：SGEMM 两套；FA 是 Triton 参考 + CUDA 深挖。
  - 禁止两套都自称生产最优。
  - 详见 `cross-cutting.md` §2。
- 证据：E2、E6、E9；`interview/cross-cutting.md`
- 追问 1：会不会用 Triton 重写 tiny-llm GEMM？ → 不会。

### Q20. Triton 3.x dtype / compile 兼容怎么讲？
- 一句话答案：版本会变；本 freeze compile smoke 是 skip；假 FP8 E4M3 已删。
- 展开（3–5 点）：
  - 选 Triton 就要把框架版本写进测试矩阵。
  - uint8 线性量化不是 E4M3，名称不能先于实现。
  - pytest **116 passed, 1 skipped**。
- 证据：E5；数字卡 §6；triton README 负资产说明
- 追问 1：skip 能当交付吗？ → 能，前提是文档写明 skip 原因；把 skip 改成 xfail 伪装绿不行。

---

## FlashAttention（Q21–Q32）

### Q21. online softmax 在维护什么？
- 一句话答案：每行的 running max `m` 和分母 `l`（以及未归一化分子 O），tile 到来时用 `exp(m-m_new)` 缩放旧值再累加。
- 展开（3–5 点）：
  - `m_new = max(m, tile_max)`。
  - 本 tile：`exp(s - m_new) * V` 累进分子。
  - 结束 `O /= l`。
  - 不必物化完整 `S = QK^T`。
- 证据：`cuflash-attn` 前向 kernel；讲述稿 `03-cuflash-attn.md`
- 追问 1：为什么不先全局 max 再 softmax？ → 那要两遍扫 KV，失去单遍 streaming 和 O(N) 工作集。

### Q22. 为什么说 FlashAttention 是 O(N) 内存？
- 一句话答案：不存 N×N 的 attention 矩阵；辅助状态随序列长度线性，tile 在 smem 里滚动。
- 展开（3–5 点）：
  - 计算仍是 O(N² d)，内存工作集不是 O(N²)。
  - 这是算法声明，不是「我们比 FA2 更省显存」的实测表。
  - 本仓没有把 3030/5118 当 FA 显存证据（那对数字无归档，E22）。
- 证据：cuflash algorithm 文档；E6
- 追问 1：反向是不是也 O(N)？ → 反向要重算或重读 tile，并保存 logsumexp；仍避免物化 S，但实现更重（Q23/Q32）。

### Q23. 前向和反向如何衔接？
- 一句话答案：前向写出 O 和行统计（logsumexp `L`）；反向用它们重建 softmax 权重，再反传到 Q/K/V。
- 展开（3–5 点）：
  - `L` 在 FP16/BF16 API 里保持 **FP32**（v0.5.0 breaking），否则 `exp(S-L)` 坏。
  - 多精度差分在 unit tests（E6）。
  - 反向长序列误差压测未做，短板要主动说。
- 证据：E6；cuflash CHANGELOG `[0.5.0]`
- 追问 1：tiny-llm 用这套反向吗？ → 不用。runtime 只推理；cuflash 是专项深挖，未链进 generate。

### Q24. causal mask 在 tile 算法里怎么处理？
- 一句话答案：未来位置不进 softmax；实现上对 tile 内非法 QK 置 `-inf`，并可跳过整块全未来的 KV tile。
- 展开（3–5 点）：
  - 旧路径已有粗 `break`。
  - E2b 再 skip 全未来块，实测 ±2%，当负结果（Q26）。
  - causal FP32 seq=1024：4.55 ms；4096：56.7 ms（数字卡 §4）。
- 证据：E8；`docs/performance/benchmarks.md`
- 追问 1：-inf 会不会在 FP16 溢出？ → 要和 online max 一起设计；本仓多精度测试锁数值，不靠口头保证。

### Q25. grid.y > 65535 是什么 bug？
- 一句话答案：CUDA 网格 y 维上限 65535；`grid.y = B*H` 在 B=512,H=128 时为 65536，launch 失败。
- 展开（3–5 点）：
  - 修复：把 batch×heads 展平到 x（`d144765`）。
  - 回归：`ForwardTest.GridYOverflowSmoke`，FP32 scalar、seq_len=1。
  - 不覆盖 WMMA tile 约束。
- 证据：E7
- 追问 1：为什么测 seq_len=1？ → 用最小形状打 launch 配置，不和数值精度缠在一起。

### Q26. causal skip 为什么只有 ±2%？
- 一句话答案：主路径已被旧 `break` 覆盖；新 skip 的增益低于 10% 阈值，是噪声级负结果。
- 展开（3–5 点）：
  - 表：256 +1.2%；512 −0.7%；1024 −0.9%；2048 −1.9%；4096 −1.2%（`e1735b3`）。
  - 保留是因为语义清楚、数值不回归，**不是因为加速成功**。
  - 红灯句：「我们做了因果优化所以更快」。
- 证据：E8；`docs/performance/causal-boundary-skip.md`
- 追问 1：那为什么不回滚？ → 无效访存意图可文档化；回滚省不下可测延迟，还会丢掉回归锁。

### Q27. FlashDecoding / Split-KV 解决什么？
- 一句话答案：decode 时 Q 短、KV 长；按 KV 分块并行算局部 online softmax，再跨块 reduce。
- 展开（3–5 点）：
  - query_len=1 路径（`9f65df7`）。
  - 测试：`MatchesCpuReferenceF32`、`ChunkCountInvariant`。
  - 不是把训练前向整段换成 Split-KV。
- 证据：E9
- 追问 1：tiny-llm decode attention 用它了吗？ → 没有。owner 分离。

### Q28. 和 FA2/FA3 差在哪？
- 一句话答案：没有 warp specialization、TMA/WGMMA、完整 pipelining；教学吞吐，ROADMAP 未声称追上。
- 展开（3–5 点）：
  - 本机非 causal FP16 seq=1024：**1.76 ms**；4096 hd=128：**84.1 ms**（`6860cbc`）。
  - 对 SDPA 大约 0.42×–0.67× 是预期差距（数字卡边界 §9）。
  - `cp.async` 双缓冲未做（ROADMAP）。
- 证据：数字卡 §4、§8；`cuflash-attn/ROADMAP.md`
- 追问 1：面试官说太慢怎么办？ → 承认教学实现；能讲 online softmax、grid 上限、负结果 skip、Split-KV 契约。

### Q29. FP16/BF16 数值要注意什么？
- 一句话答案：累加和 logsumexp 用更高精度；API 输出可以是 FP16/BF16。
- 展开（3–5 点）：
  - head_dim 支持 32/64/128。
  - 差分有容差；不是 bitwise vs SDPA。
  - freeze 里 pytorch comparison **skip**（E6）。
- 证据：E6；数字卡 §4 表
- 追问 1：BF16 和 FP16 谁更快？ → 本机表里两者接近（如 1024：1.76 vs 1.79 ms）；不把噪声说成架构结论。

### Q30. LogicalHBM 是什么？为什么不能说成物理带宽？
- 一句话答案：按算法模型估计的字节流量（Q/O 各一次 + K/V 按 Q block 重载），counter 故意叫 `LogicalHBM GB/s`。
- 展开（3–5 点）：
  - 不是 ncu 测到的 DRAM 计数器。
  - 反向不乱用同一模型（文档：避免模型继续出错）。
  - 红灯：把 LogicalHBM 说成「我们打满了 HBM」。
- 证据：`cuflash-attn/benchmarks/bench_flash_attention.cu`；README 带宽口径
- 追问 1：为什么还要报它？ → 让不同 seq/head 的时间可对照流量模型；同时防止自己撒谎。

### Q31. cuflash 的 ctypes 绑定和 triton 的 torch.library 有何不同？
- 一句话答案：ctypes 是教学/C++ API 的 Python 包装；torch.library 是框架图可见的 custom op。
- 展开（3–5 点）：
  - 不要把两套说成同一接入机制。
  - tiny-llm 走自己的 C ABI，不经过这两套。
  - FA owner 是 cuflash；Triton FA 只是参考（E3/E6）。
- 证据：E5、E6；各仓 README 边界
- 追问 1：为什么不把 cuflash 注册进 torch.ops？ → 作品集要展示 C++ 深度；接入层已经在 Triton 仓演示过。

### Q32. 反向数值稳定性你怎么守？
- 一句话答案：保存 FP32 logsumexp，online 缩放避免溢出，用多精度差分而不是单次肉眼看 loss。
- 展开（3–5 点）：
  - v0.5.0 把 `L` 提到 FP32 是 breaking，说明旧路径有坑。
  - 长序列误差压测未做 → 短板。
  - 训练核未接到 tiny-llm，避免「推理仓有完整 bwd」的暗示。
- 证据：E6；CHANGELOG `[0.5.0]`
- 追问 1：差分过了是否等于训练可用？ → 不等于。缺大规模 bwd 压测和优化器闭环。

---

## tiny-llm（Q33–Q46）

### Q33. GGUF 在本引擎里扮演什么角色？
- 一句话答案：权重容器；加载后反量化再重量化为 W8A16，不是运行时按 GGUF 块直接 matmul。
- 展开（3–5 点）：
  - 一条命令：`tiny_llm_demo ...q4_k_m.gguf --prompt "你好"`（E10）。
  - 模型：Qwen2.5-0.5B Instruct。
  - llama.cpp 对比用同一 GGUF 文件，但对方原生 Q4_K_M（E17）。
- 证据：E10、E12
- 追问 1：能直接跑 Q4_K kernel 吗？ → 当前推理路径是 W8A16；Q4_K 用于加载反量化测试。

### Q34. Q4_K / Q5_0 反量化怎么验证？
- 一句话答案：合成块对 Python `gguf.quants`；真实模型首块门控于本机 Qwen GGUF。
- 展开（3–5 点）：
  - 覆盖 Q4_0 / Q5_0 / Q8_0 / Q4_K / Q6_K（E12）。
  - 测试：`DequantizeQ5_0`、`DequantizeQ4_K`、`DequantizeQ6_K`、`GGUFRealModelTest`。
- 证据：E12；`tiny-llm/tests/test_quantization.cpp`
- 追问 1：反量化误差会传到 token 吗？ → 会。请求 2 的 is/equals 就是重量化后的 argmax 边界（Q44）。

### Q35. W8A16 是什么意思？
- 一句话答案：权重 INT8 + FP16 scale，激活 FP16；本仓 group 大小 128。
- 展开（3–5 点）：
  - 不是 GPTQ/AWQ 的生产量化栈。
  - 有 CPU 参考差分：`W8A16MatMulTest`、`WeightW8A16RoundTripPreservesValues`（E13）。
  - 转置快路径必须对同一参考（`TransposedFastPathMatchesCpuReference`）。
- 证据：E13；`tests/test_w8a16_matmul.cu`
- 追问 1：为什么不保持 Q4_K 推理？ → 要可控地看 M==1 GEMM 访存；格式与 kernel 绑死就讲不清转置故事。

### Q36. per-group scale 怎么用？
- 一句话答案：每 128 个权重一个 scale；反量化 `w = q * scale` 后再与 FP16 激活乘。
- 展开（3–5 点）：
  - 真实形状 1×896×896、group 128 必须与 reference 对齐。
  - group 越大越省 scale 流量、越粗量化。
  - 没有单独报「group=64 会快多少」，不编。
- 证据：`tiny-llm/tests/test_w8a16_matmul.cu` 注释；E13
- 追问 1：scale 存在转置副本里吗？ → 转置的是权重布局以利 K 向连续读；scale 仍按 group 对齐。细节以 kernel 为准，不背未归档的字节表。

### Q37. GQA 14→2 如何映射？
- 一句话答案：`kv_head = q_head / group_size`；Qwen2.5-0.5B 是 14 个 Q head、2 个 KV head。
- 展开（3–5 点）：
  - 测试：`GQAMappingDecodeMatchesCpuReference`（`fdbabcc`）。
  - 必须进真实 `transformer.cpp` 路径，不能只在单测里存在。
  - 第二真实模型 `TLLM_GGUF_TEST_MODEL_2` 本次 skip。
- 证据：E14
- 追问 1：MHA 能不能跑？ → 规格测试覆盖映射；旗舰模型是 GQA。不要说「任意 HuggingFace 结构都支持」。

### Q38. RoPE 加在前向的哪一步？
- 一句话答案：attention 之前 `apply_rope_inplace`（`transformer.cpp`）。
- 展开（3–5 点）：
  - 测试：`RoPETest.ApplyInplaceMatchesReference`（`1038639`）。
  - 位置 id 必须与 KV 写入位置一致，否则长上下文会错。
  - 和 Triton 仓 half-split 契约一致（Q16），runtime 是 CUDA。
- 证据：E14
- 追问 1：Graphs 捕获 RoPE 吗？ → decode device 路径包含小 kernel；变化的位置在 device 状态里。捕获范围见 Q42。

### Q39. KV append 和 advance 为什么是两段式？
- 一句话答案：各层先按当前 `current_len` 写入，所有层写完后 **一次** `advanceSeqLen`。
- 展开（3–5 点）：
  - 若每层都 advance，后面的层会写到错误槽。
  - `advanceSeqLen` 溢出从静默 clamp 改为返回错误。
  - Graphs **不 capture** host `advanceSeqLen`。
- 证据：`tiny-llm/src/kv_cache.cpp` 注释；`DEVELOPMENT_PLAN.md` Graphs 范围
- 追问 1：prefill 和 decode 都走两段式吗？ → 是。差别在 `num_tokens` 是 S 还是 1（Q40）。

### Q40. `append_kv_at` 的 bug 是什么？怎么发现的？
- 一句话答案：为 Graphs 写的 kernel 曾按 decode 的 1 token 落盘；prefill 多 token 只写 1 行。真模型策略 1 vs 2 差分抓住。
- 展开（3–5 点）：
  - 修复：`total = num_tokens * per_token`，`offset = (write_pos + t) * per_token + j`。
  - 注释：「D2e 修复：prefill 多 token 时逐行写入」。
  - 发现：`FFITest.PagedKVStrategyMatchesContiguous`（`7b456cd`）。
  - 教训：为 decode 特化的 kernel 必须用 S>1 锁 prefill。
- 证据：`tiny-llm/kernels/elementwise.cu`；E20；讲述稿故事 ③
- 追问 1：连续 KV 为什么没爆？ → 它不走这条 paged scatter；差分的对照面是对的。

### Q41. 转置权重为什么能加速 M==1 GEMM？
- 一句话答案：旧核读 `weight[k*N+col]`，lane stride=N，不 coalesced；转置后读 `weight_t[col*K+k]`，stride=1。
- 展开（3–5 点）：
  - Before TPOT **24.348 ms**；C1 **6.560**；C2 **6.087**。
  - lm_head **10.0002 → 0.9794 ms**；N=4864 **0.1631 → 0.0486 ms**。
  - 峰值显存 2494 → **3368 MB**（转置副本）。
  - ncu 不可用，microbench 是证据（E15/E28）。
- 证据：E15；数字卡 §1–3
- 追问 1：这是 Tensor Core 优化吗？ → 不是。是访存形状。M==1 时教学 WMMA 帮不上这篇故事。

### Q42. CUDA Graphs 捕获什么？为什么要 device 参数化？
- 一句话答案：捕获 decode 的 device kernel 序列；会变的索引必须在 device 上，否则图把地址写死。
- 展开（3–5 点）：
  - 默认 ON，`TLLM_CUDA_GRAPHS=0` opt-out（`f897084`）。
  - 测试：`InferenceEngineTest.CudaGraphsGenerateMatchesNonGraph`。
  - 不 capture：`advanceSeqLen`、logits D2H、采样。
  - `append_kv_at` 从 device `write_pos` 读（Q40）。
- 证据：E16；`tiny-llm/docs/performance/cuda-graphs.md`
- 追问 1：paged 路径接 Graphs 了吗？ → 没有。Phase 4 候选；现在讲边界。

### Q43. TTFT 和 TPOT 口径如何避免被追问打穿？
- 一句话答案：TPOT 用 `tiny_llm_bench` greedy mean；llama.cpp 用 `llama-bench tg64` 3.7 ms；TTFT 两边定义不同，禁止相除当公平比。
- 展开（3–5 点）：
  - 本机 TPOT **6.087 ms**、164.283 tok/s、TTFT **10.567 ms**。
  - 比值 **1.65**（6.09/3.7），非同量化。
  - README 早期 ~4.7× TTFT 不可当公平比（数字卡 §8）。
  - `--iters 5` 与 README `--iters 10` 要报对应命令。
- 证据：E17；数字卡 §1、§8
- 追问 1：1.65× 算接近吗？ → 数量级接近；C1 前约 6.6×。故事在访存，不在「已经追上」。

### Q44. 为什么「2+2」会变成 is vs equals？
- 一句话答案：W8A16 vs Q4_K_M 在 argmax 边界翻转；公共前缀 `[17,10,17]`，然后 374 vs 16819，两边都到 EOS 151645。
- 展开（3–5 点）：
  - 测试只断言前缀 + EOS（`9c974d3`）。
  - 请求 1 的 24 id（含 EOS）全等（E21）。
  - 不是调度写错。
- 证据：E18、E21
- 追问 1：怎样才能全序列对齐？ → 接同一套量化，而不是把引擎说成更懂中文。

### Q45. FFI ABI v2 有哪些字段？
- 一句话答案：`TinyLlmConfig` 9 个 C `int`：hidden、layers、heads、kv_heads、head_dim、vocab、block_size、max_batch、`max_num_blocks`。
- 展开（3–5 点）：
  - `tinyllm_step` 另传扁平 `block_tables` / `num_blocks`。
  - Rust：`sizeof == 9*4`，`tiny_llm_config_layout_is_stable`（`050c80a`）。
  - `max_num_blocks==0` → 策略 2 连续 KV。
- 证据：E19；`tiny-llm/include/tiny_llm/ffi.h`
- 追问 1：ABI 会不会吃掉 6.09 ms？ → 相对 GEMM，几次 C 调用不是主因。没有单独微秒表，不报。

### Q46. 和 llama.cpp 还差什么？
- 一句话答案：量化格式、kernel 成熟度、多后端、采样器生态；本机 TPOT 仍 1.65×，且不是同量化。
- 展开（3–5 点）：
  - 产品我会选 llama.cpp / vLLM（cross-cutting §1）。
  - 作品集剩下的是契约、负结果、差分。
  - 没有多卡、prefix cache、抢占。
- 证据：E17；`interview/cross-cutting.md` §1
- 追问 1：简历为什么还没 llama.cpp PR？ → 还没做。Phase 4 才考虑 good-first-issue。

---

## paged-infer / serving（Q47–Q60）

### Q47. PagedAttention 要解决什么碎片问题？
- 一句话答案：为 max_seq 预留连续 KV 会把显存钉死在峰值；块化按需分配，用 block table 做逻辑连续。
- 展开（3–5 点）：
  - 控制面持有映射；计算面按块读写。
  - 本仓正确性证据是策略 1 vs 2 逐 token 一致（E20），**不是** 3030 vs 5118 MiB（E22）。
  - 策略 1 仍有 scatter/gather 往返代价。
- 证据：E20、E22；paged-infer README
- 追问 1：碎片和内部碎片是一回事吗？ → 块内最后一块会浪费不到 block_size 的槽；本仓用属性测试锁块计数，不锁「浪费 <5%」宣传句。

### Q48. block table 存在哪一侧？
- 一句话答案：paged-infer 控制面持有；策略 1 经 FFI 上传扁平表和每序列 `num_blocks`。
- 展开（3–5 点）：
  - tiny-llm 按表 scatter/gather，不自己做调度。
  - 策略 2：`PAGED_INFER_TINY_LLM_STRATEGY=2`，`max_num_blocks=0`。
  - 默认策略 1（ROADMAP 已勾选）。
- 证据：E19、E20
- 追问 1：扁平表越界怎么办？ → 由 runtime 校验；布局错误会在 Rust 守卫或 GPU 差分爆，不允许静默。

### Q49. continuous batching 和 static batching 差在哪？
- 一句话答案：请求异步到达、完成即退；同一 step 可混 prefill 与 decode，而不是等整批结束。
- 展开（3–5 点）：
  - decode 优先，避免新 prefill 饿死在途 token。
  - 3 并发 e2e 是正确性 fixture，不是 QPS 曲线（E21）。
  - 无 chunked prefill。
- 证据：`paged-infer/src/scheduler.rs`；E21
- 追问 1：和动态 batching 营销词有何不同？ → 这里有状态机 + 块池 + 水位线；不是「凑满 8 条再跑」。

### Q50. 请求状态机有哪些状态？
- 一句话答案：Pending → Prefill → Decode → Completed；失败可在任一段释放 KV。
- 展开（3–5 点）：
  - 结束后 `active_sequences==0`、利用率回 0（e2e 断言）。
  - 取消/失败归还有属性测试（E23）。
- 证据：E23、E21；`scheduler.rs`
- 追问 1：Completed 还占块吗？ → 不应占。属性测试锁 `used+free==total`。

### Q51. 三层准入指什么？
- 一句话答案：先看调度优先级与 batch 槽，再看 KV 块是否够，再看内存水位线（含 decode reserve）。
- 展开（3–5 点）：
  - 水位线：预估占用后利用率 ≤ threshold。
  - 给在途 decode 预留增长，否则不接新 prefill。
  - TinyLlm 适配器还把最大并发 clamp 到 4（6GB 卡）。
- 证据：E24；`paged-infer/src/tiny_llm_executor.rs`；讲述稿 `05-paged-infer.md`
- 追问 1：clamp 4 是算法上限吗？ → 不是。是本机显存保护，写在 Rust 适配器。

### Q52. 内存水位线和 decode reserve 为什么要绑在一起？
- 一句话答案：只看当前 used 会在 prefill 后把 decode 增长空间吃光，在途请求 OOM。
- 展开（3–5 点）：
  - 这是 serving 控制面问题，不必 GPU。
  - 测试在 scheduler 单测里（E24）。
  - 与「峰值 3368 MB」不是同一口径：后者是 tiny-llm 转置副本。
- 证据：E24；数字卡 §3 vs §8
- 追问 1：reserve 512 token 怎么来的？ → 适配器 decode 预留；不是从 ncu 推出来的。面试说「经验保护」，不说「最优」。

### Q53. HOL 阻塞怎么修？
- 一句话答案：队头大 prefill 不能饿死后面可运行的小请求。
- 展开（3–5 点）：
  - 回归：`test_small_pending_request_not_blocked_by_large_one`。
  - 占满 batch 后，1-token 请求须在有限 step 内完成。
  - 无抢占时，HOL 更要命：不能把大请求踢走，只能在调度顺序上让路。
- 证据：E24
- 追问 1：这算公平性还是吞吐？ → 先是活性（liveness）。没测生产公平性指标。

### Q54. 优先级调度规则是什么？
- 一句话答案：先 decode，再高优先级 prefill，再 pending；同级 FCFS。
- 展开（3–5 点）：
  - 测试：`test_priority_higher_prefill_starts_first`。
  - 高优先级来了也不能抢已经在跑的序列（无抢占）。
- 证据：E24；`scheduler.rs` 头注释
- 追问 1：OpenAI API 怎么表达优先级？ → 控制面内部字段；不要暗示完整生产优先级产品。

### Q55. 为什么明确不做抢占？
- 一句话答案：无 swap / preempt-resume；内存不够就拒新请求或失败。
- 展开（3–5 点）：
  - README/ROADMAP「明确不做」。
  - 这是和 vLLM 的边界，主动讲。
  - 高优先级到达不能踢人。
- 证据：`paged-infer/README.md` 无抢占节；E24
- 追问 1：那水位线是不是抢占的替代？ → 是准入控制，不是抢占。已占用的 KV 不会被换出。

### Q56. swap / recompute 你们做到哪一步？
- 一句话答案：生产向的 KV swap 未做；CPU 参考后端有「增量 decode vs 全量 recompute」一致性测试。
- 展开（3–5 点）：
  - `test_multilayer_incremental_vs_full_recompute` 锁的是计算正确性，不是调度换出。
  - 不要把这条测试说成 vLLM recompute 抢占。
- 证据：`paged-infer/src/cpu_executor.rs`；README 无抢占
- 追问 1：OOM 时会 recompute 吗？ → 不会。失败/拒绝。

### Q57. 分页 KV 策略 1 的数据流是什么？
- 一句话答案：调度器分配块 → FFI 上传 block table → tiny-llm scatter 写 KV / gather 读 KV → 采样 token 回控制面。
- 展开（3–5 点）：
  - 与策略 2 真模型逐 token 一致（E20）。
  - 代价：多一次显存往返；未接 Graphs。
  - freeze 默认 `cargo test` **没开** `--features tiny-llm`。
- 证据：E19、E20、E21；数字卡 §6
- 追问 1：gather 能消掉吗？ → Phase 4 候选 P4-1；现在不改源码。

### Q58. 3 并发 e2e 证明了什么、没证明什么？
- 一句话答案：证明块表 + ABI + 生命周期能承载真实 Qwen 与 llama.cpp greedy 对齐（请求 1 全等）；不证明生产并发或吞吐。
- 展开（3–5 点）：
  - 请求 1：24 id，末位 EOS **151645**。
  - 请求 2：前缀 + EOS，诚实分歧（E18）。
  - 跑完 `active_sequences==0`。
  - T1 未重跑该 feature 测试，开口要带这句。
- 证据：E18、E21；`qwen2_three_concurrent_paged_requests_match_llama_cpp`
- 追问 1：为什么是 3？ → fixture 设计；不是容量规划。适配器还 clamp 到 4。

### Q59. SSE token 级流式的边界是什么？
- 一句话答案：`SimpleTokenizer` 可 token 级 SSE；HF tokenizer 是结束时一整块。`data: [DONE]` 有 integration 测试。
- 展开（3–5 点）：
  - 曾经假流式（切 32 字符）已删除。
  - 测的是 CPU 参考后端 HTTP 契约，37 项，不是 GPU QPS（E25）。
  - 引擎循环必须 yield，否则首 token 卡到整批结束。
- 证据：E25；`paged-infer/ROADMAP.md`；CHANGELOG 流式诚实化
- 追问 1：线上 demo 能看流式吗？ → 取决于 tokenizer 后端。HF 路径不要吹 token-level。

### Q60. Rust 后端 trait / capabilities 怎么体现边界？
- 一句话答案：`TinyLlmExecutor` 返回 `GREEDY_ONLY`（无 sampling、`retry_safe=false`）；语言不是重点，边界可测试才是。
- 展开（3–5 点）：
  - 布局守卫 + 属性测试 + 无 GPU 也能测调度。
  - 计算面仍是 C++/CUDA。
  - 把标题说成「Rust 比 C++ 更适合 AI Infra」减分。
- 证据：E19、E23；`paged-infer/src/tiny_llm_executor.rs`；`interview/cross-cutting.md` §3
- 追问 1：会改成 Python 控制面对标 vLLM 吗？ → 能讲对照，但会失去 `repr(C)` 守卫和 cargo 不变量；不是本作品下一步。
