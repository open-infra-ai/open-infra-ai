# 三个必答题

每篇正文约 800–1200 字（汉字）。数字与 commit 不得离开 [`NUMBERS_CARD.md`](NUMBERS_CARD.md) / [`EVIDENCE_MATRIX.md`](EVIDENCE_MATRIX.md)。

---

## 1. 为什么不用 llama.cpp / vLLM？

一句话：它们是产品；我这套是为了把推理栈拆开、测准、讲清。如果明天要上线服务，我会直接用它们，而不是用 tiny-llm 去扛流量。

llama.cpp 已经把 GGUF 容器、多种量化格式、CUDA/Metal/CPU 后端、采样器和命令行做成了别人能一条 `llama-cli` 跑通的东西。vLLM 已经把 PagedAttention、continuous batching、调度器、OpenAI 兼容 API 和分布式做成了行业默认。在这个前提下，再写一个「功能更少的 llama.cpp」或「更慢的 vLLM」没有工程价值。唯一站得住的目标是：**每一层我都能指出文件、测试、失败模式和数字口径**。这是转行作品集，不是创业替代品。

学习目标和生产目标必须分开说，否则面试官会认为你在贬低上游。生产要的是量化覆盖、多卡、抢占、稳定性、生态和有人值班。学习要的是另一组问题：QKV 的生产者布局和 attention 消费者是否一致；GQA 有没有 `kv_head = q_head / group_size`；RoPE 是否真的进了 Transformer 前向；KV 的 append 和 advance 是不是两段式；decode 的时间到底在反量化 GEMM、lm_head 还是 kernel launch。这些问题在 llama.cpp 里都已经有正确答案，但答案埋在很大的代码墙后面。tiny-llm 把路径收成可审计的最小运行时：真实 Qwen2.5-0.5B、W8A16、tokenizer 与 HuggingFace 30 例 417 token 逐 id 对齐。`tiny_llm_bench` 给出 TPOT **6.087 ms/token**、decode **164.283 tok/s**。同卡 llama.cpp `llama-bench` 的 `tg64` 是 **3.7 ms / 272.2 t/s**。比值 **1.65**。方法论文档写明两边量化不同：llama.cpp 原生算 Q4_K_M，tiny-llm 反量化后再重量化成 W8A16（E17）。C1 转置之前这个比值大约是 6.6。故事是「我找到了访存瓶颈」，不是「我更快」。

可控性是第二理由。llama.cpp 的量化格式和 kernel 绑在一起。我想看「重量化成 W8A16 之后，M==1 的 GEMM 在 RTX 3060 上怎么死」，就必须自己持有反量化、转置副本和 `w8a16_matmul_m1_transposed_kernel`。microbench 把 lm_head 从 **10.0002 ms** 打到 **0.9794 ms**，N=4864 的 W8A16 从 **0.1631 ms** 打到 **0.0486 ms**（E15）。这些数字在 llama.cpp 内部很难单独抽出来当教材。vLLM 的分页 KV 和调度绑在一起。我想看「控制面和计算面如何用 9 个 C int 分开」，就必须自己持有 `ffi.h` 和 Rust 的 `tiny_llm_config_layout_is_stable`（E19）。从上游学到的设计要能点名：GGUF 当权重容器、CUDA Graphs 消 launch、PagedAttention 的块表、continuous batching 的 decode 优先。这些是作业对象，不是竞争对象。

第三层是诚实。请求 2 上 llama.cpp 出 `equals`（token 16819）、tiny-llm 出 `is`（374），不是调度写错，是两种量化在 argmax 边界翻转（E18）。测试只断言公共前缀三枚 token 加 EOS 151645。如果目标是「每一个 token 都对齐 llama.cpp」，正确做法是接同一套量化，而不是把引擎吹成更懂中文。峰值显存 **3368 MB** 里含转置副本，C1 前是 **2494 MB**（E15/数字卡 §3）。这是用显存换 coalescing，不是免费午餐。

进组之后我会怎么用这套经验，也要说清楚。读 vLLM 的调度器时，我会先找状态机和内存水位线，而不是先找融合 kernel。读 llama.cpp 的 decode 时，我会先问量化格式和 M==1 GEMM 的访存，而不是先问「有没有 CUDA Graphs」。这是作品集真正迁移到工作里的部分：不是搬运五仓代码，而是搬运提问顺序。若岗位要的是立刻改 vLLM 的 CUDA graph runner，我会承认 tiny-llm 没有多卡、没有抢占、没有 prefix cache；我能贡献的是把问题定义清楚、把差分测试补上、把负结果写进文档。

所以结论必须清醒。五仓证明我读懂了 llama.cpp 和 vLLM 的分层；**产品我会选它们**。继续自建只在两种情况下合理：面试要讲的深度锚点还没讲完，或者上游有一个小到能 merge 的缺口。把作品集说成开源替代品，是减分。

**追问**

1. 简历为什么不写 llama.cpp 贡献？ → 还没做。Phase 4 才考虑 good-first-issue。现在能讲的是对齐方法和差距口径。  
2. 1.65× 算接近吗？ → 数量级接近，量化不同。C1 前约 6.6×，故事在访存。  
3. 会不会把 tiny-llm 接到 vLLM？ → 没有产品动机。价值在契约，不在替换 vLLM 的 runner。

---

## 2. 什么时候用 Triton，什么时候用 CUDA C++？

一句话：Triton 用来快速表达公式并接入 PyTorch 图；CUDA C++ 用来抠访存、launch 配置和架构细节。我两边写了同一道 GEMM 和同一道 Attention，所以这不是空谈。

Triton 的调度单位是 program，通常对应一个 CUDA block。你用 `tl.arange` 构造 tile，用 mask 挡住越界，用 `tl.dot` 做累加，用 `num_warps`/`num_stages` 做很粗的调参。开发速度来自不必手写 shared memory 的双缓冲索引和指令级调度。代价是对 bank conflict、warp 特化、TMA、精确 occupancy 的控制变弱。调试也更依赖「和 `torch.mm` 或 SDPA 差分过了」。本仓库 `tests/test_sgemm.py` 有 24 项对 `torch.mm`，形状包括 64 对齐、M=1、N=1 和非 2 幂 17×33×65，容差 rtol/atol=1e-2（E2）。三条融合算子各有独立 NumPy/PyTorch reference（E3）。这是 Triton 的正确用法：先锁输入契约和参考实现，再谈融合省几次 HBM。`fused_rmsnorm_rope` 在 (1,128,4096) 上是 **0.104 ms**，`fused_gated_mlp` silu 在 (1,128,4096,11264) 上是 **3.45 ms**（README 表，commit `ebf6c32+`）。3.45 ms 对应大约 10 TFLOPS，相对 RTX 3060 Laptop 的 FP16 理论峰值大约 46 TFLOPS，是练习实现，不是打榜数字。

CUDA C++ 的调度单位是线程、warp、shared memory 和 `mma`。cuda-foundations 的 SGEMM 阶梯把 naive **0.58 TFLOPS** 推到 WMMA **1.09**，同时 **bank-conflict-free 掉到 0.66**、double-buffer 是 **0.68**，cuBLAS 是 **5.58**（E1）。这种「写了优化步骤反而变慢」在 Triton 里很难当教材留下来，因为你往往看不到那一次 padding 换来的占用变化。cuflash-attn 的 online softmax、把 `grid.y=B*H` 展平以免超过 65535、FlashDecoding 按 KV 分块再 reduce，都是必须碰 launch 配置和数值稳定性的问题。Triton 的 FlashAttention 前向故意留在 triton-fused-ops 当参考，owner 是 cuflash（E6、E9）。tiny-llm 的 decode GEMM 更是纯 CUDA 访存：旧 kernel 读 `weight[k*N+col]`，lane 之间 stride 是 N；转置后读 `weight_t[col*K+k]`，stride 是 1（E15）。这类故事 Triton 写得出来，但你很难在面试里指着一条 PTX 级理由。

接入形态也不一样。推理框架要的是 `torch.ops.vllm.*` 这种命名空间，不是一个孤立的 `.cu`。所以 Triton 仓做了 `torch.library`：`triton_ops::sgemm`、`fused_rmsnorm_rope`、`fused_gated_mlp`。优先 `torch.library.triton_op`，否则 `custom_op + register_fake`（`1bbf5c8`，E5）。内部只调已有 kernel，不复制逻辑。CUDA 的 FA 走 C++ API 和 ctypes；tiny-llm 走自己的 C ABI。不要把 ctypes 教学绑定说成和 vLLM custom op 同一套机制。

Triton 3.x 的坑要主动提。TRIT-001 不是性能 bug，是 RoPE 排列。`repeat_interleave(freqs,2)` 和 `concat([freqs,freqs])` 都能跑出「像 RoPE 的东西」，只有 half-split 和 Qwen/Llama 的 `rotate_half` 一致（`b1bcdcb`，E4）。dtype 和编译路径随 PyTorch/Triton 版本变。本次 freeze 里 `test_torch_compile_smoke` 是 skip，不是绿（E5）。选 Triton 就要把框架版本写进测试矩阵，不能假设 compile 永远成功。本仓还删过假 FP8 E4M3 路径：名字能写进 README，并不等于数值契约已经锁死。Triton 让错误实现也能很快跑出「看起来合理」的延迟，所以差分测试和契约注释比 autotune 表更重要。

融合也不是默认正确。RMSNorm+RoPE 合成一次，是因为两者都是逐元素、同一行、中间结果不必回 HBM。把 lm_head 这种 N=151936 的 GEMM 和采样融合在一起，Triton 写得出来，但 tiny-llm 的 profiling 已经说明时间在访存形状，不在「少一次 launch」。autotuner 在本仓是基础设施，没有当成旗舰接到每个 wrapper 上；面试不把它说成 vLLM 级别的调参系统。删掉假 FP8 E4M3 也是同一纪律：名字先于实现时，Triton 的速度会把错误量化送进教程。SGEMM 的 24 项差分（含 17×33×65）就是用来挡住「融合了所以一定对」。没有对照表就不报 Triton 比教学 CUDA 更快。

我实际在用的选型规则是三条。算子还在改公式、要对着 PyTorch 周更、需要进 `torch.compile` 图，用 Triton。已经用 microbench 证明瓶颈在 coalescing、图捕获、网格上限或训练反向，用 CUDA C++。同一算法需要两套实现时，Triton 当 oracle 或接入层，CUDA 当深度作品，禁止两套都自称生产最优。cuda-foundations 和 triton-fused-ops 并排存在，就是这条规则的物理形态。

**追问**

1. FA 为什么不 Triton 到底？ → 前后向、WMMA、grid 上限、FlashDecoding 是 CUDA 深挖；Triton 前向只当参考。  
2. Triton SGEMM 比教学 CUDA SGEMM 快吗？ → 没有同一时刻 head-to-head 表，不编。  
3. 会不会用 Triton 重写 tiny-llm 的 GEMM？ → 不会。runtime 已是 C++/W8A16，再套 Python 启动器破坏最小运行时叙事。

---

## 3. Serving 控制面为什么用 Rust？

一句话：不是因为 Rust 更快，而是因为控制面的故障模式是「不变量被破坏」和「ABI 对不齐」。这两种错误在 CUDA 工程里很难测，在 Rust 里可以变成默认 CI。

先划边界。计算面是 tiny-llm：kernel、量化、CUDA Graphs、显存、真实 token。控制面是 paged-infer：请求状态机、BlockPool、continuous batching、水位线、OpenAI 兼容 HTTP。如果把调度写进同一个 C++ 仓库，HOL 修复和 attention 数值会缠在一次 GPU 测试里，失败时你不知道该怀疑哪一层。拆开之后，默认 `cargo test` 就能覆盖调度、属性测试和 server integration。本次 freeze 是 **218 passed**（E25 相关的 server 测试 37 项在内）。真实 GPU 上的 3 并发 e2e 用 `--features tiny-llm` 门控，T1 按计划没开这个 feature，所以面试时要主动说：token 对齐证据在 `9c3700b` 和 `9c974d3`，不是今天这 218。分层的价值正在这里：控制面正确性和计算面正确性可以分开变红。

跨语言边界是 Rust 的第一笔账。`TinyLlmConfig` 必须是 9 个 C `int`，顺序与 `tiny-llm/include/tiny_llm/ffi.h` 一致：hidden、layers、heads、kv_heads、head_dim、vocab、block_size、max_batch、`max_num_blocks`。少一个字段，Rust 就会把分页池大小读成词表，或者把策略 1 读成策略 2。Rust 侧 `#[repr(C)]` 加测试 `tiny_llm_config_layout_is_stable`，断言 `size_of::<TinyLlmConfig>() == 9 * 4`（E19，`050c80a`）。这比微信群里「我们约定一下结构体」可执行：布局漂移在 `cargo test` 红掉，而不是在 GPU 上静默用错 batch。步进函数同样约定扁平 `block_tables` 和每序列 `num_blocks`。策略 1 默认上传真块表；`PAGED_INFER_TINY_LLM_STRATEGY=2` 把 `max_num_blocks` 置 0，退回连续 KV。切换策略不必重编 CUDA。

第二笔账是不变量。分页内存的核心不是「浪费小于 5%」这句宣传，而是任意取消、失败、完成后 `used_blocks + free_blocks == total_blocks`。手写三个例子不够。属性测试 `prop_block_count_invariant` 和 `prop_resources_reclaimed_after_cancel_and_failure` 把随机操作序列锁在这个等式上（E23）。HOL、NaN 水位线、Unicode stop 的字节偏移，全部是控制面回归（E24），不需要 CUDA。这些 bug 的共同形状是：看起来像字符串或浮点校验，实际会让调度器卡死或永不释放块。Rust 的 `Result`、显式 `is_finite` 和测试发现成本，低于在 `.cu` 文件里找同样的逻辑。

第三笔账是端到端仍然发生在边界上，而不是发生在「Rust 更快」。3 并发 e2e 里请求 1 的 24 个 token id（末位 EOS 151645）与 llama.cpp greedy 全等（E21）。请求 2 只断言前缀和 EOS，因为量化方案不同。跑完后 `active_sequences == 0`、利用率回到 0。这证明块表、生命周期钩子和 C ABI 能承载真实模型，也证明控制面不能替计算面撒谎。没有抢占、没有 chunked prefill、没有前缀缓存，文档写在「明确不做」里。Rust 没有让这些缺口消失；它只是让缺口可测试、可陈述。T1 freeze 的 218 项测试刻意没开 `--features tiny-llm`，就是为了把「调度器绿了」和「真模型绿了」分开报，避免用控制面 CI 冒充 GPU e2e。

适配器本身也体现边界。`TinyLlmExecutor` 声明 `GREEDY_ONLY`，decode 预留 512 token，并把最大并发 clamp 到 4，避免 0.5B 模型在 6GB 卡上把连续 KV 打爆。这些限制写在 Rust 侧，而不是假装 GPU 后端无所不能。OpenAI 兼容层测的是 HTTP 契约和 SSE 的 `data: [DONE]`，HF tokenizer 流式还降级成「结束时一整块」。Rust 让这些降级能写成测试和文档，而不是口头「差不多有流式」。

升华只有一句。语言不是重点，边界才是。若控制面用 C++、计算面用 Rust，只要 ABI、生命周期和测试矩阵同样严，故事仍然成立。选 Rust 是因为 `repr(C)` 守卫、属性测试和「无 GPU 也能测调度」刚好落在这条边界上。把标题说成「Rust 比 C++ 更适合 AI Infra」，是错的，也是减分。

**追问**

1. FFI 会不会吃掉 6.09 ms 的 TPOT？ → 相对 decode GEMM，几次 C 调用不是主因。没有单独的 ABI 微秒测量，不报数。  
2. 为什么不把调度写成 CUDA kernel？ → 状态机、HTTP、取消和块表是主机职责；塞进 GPU 就做不了属性测试。  
3. 会不会改成 Python 控制面去对标 vLLM？ → 能讲对照，但会失去布局守卫和 cargo 不变量。不是本作品的下一步。
