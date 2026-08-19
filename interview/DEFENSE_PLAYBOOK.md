# 面试弱项防御手册

> Phase I1。把 PLAN_v3 §1.5 的诚实短板升级为可排练的防御剧本。
> 原则：先承认、不辩解、用仓库文件/commit/命令把弱点变成可查证的口径。
> 所有数字以 [`NUMBERS_CARD.md`](NUMBERS_CARD.md) 为准；证据编号见 [`EVIDENCE_MATRIX.md`](EVIDENCE_MATRIX.md)。

---

### D1. ncu/nsys 在本机不可用（性能证据链是 microbench + CUDA Event）
- 面试官最可能怎么问：你这个性能结论是 ncu 测出来的吗？occupancy 是多少？
- 10 秒承认口径：ncu 在本机报 `ERR_NVGPUCTRPERM`，nsys stats 也缺 importer，所以我没有 profiler 指标；我用 kernel microbench + CUDA Event 搭了替代证据链。
- 证据式回答：`tiny-llm/src/kernel_bench.cpp` 提供 `tiny_llm_kernel_bench`，能逐 kernel 报延迟；文档 `tiny-llm/docs/performance/results/2026-08-18-decode-optimization.md:34-52` 写明替代链；`cuda-foundations/docs/en/guides/profiling.md` 有 WSL2 GPUCTRPERM runbook。关键 commit `tiny-llm@ca70de2` 加入 microbench。它能回答“decode 时间花在哪个 GEMM”，但不能代替 occupancy/roofline 计数器。
- 绝对不能说：我没有 profiler 所以这些数字不可信，或者假装 ncu 跑过 occupancy。
- 追问 1：那你怎么知道瓶颈不是 attention？ → 同一份 microbench 里 attention/rmsnorm/rope 只有 10–50 µs，而 lm_head 是 10.0002 → 0.9794 ms，量级差足够定位。
- 追问 2：microbench 会不会脱离端到端？ → 会，所以端到端 `tiny_llm_bench` 也跑：TPOT 24.348 → 6.087 ms；两套证据互相印证，且都带 commit 与硬件口径。

### D2. llama.cpp 对比是 W8A16 vs 原生 Q4_K_M，非同量化
- 面试官最可能怎么问：你比 llama.cpp 快/慢多少？这公平吗？
- 10 秒承认口径：我的 tiny-llm 是 W8A16 推理，llama.cpp 跑的是原生 Q4_K_M，两边不是同一种量化，所以只说 TPOT 比值 1.65×，并声明量化口径不同。
- 证据式回答：`tiny-llm/docs/performance/benchmark-methodology.md:31-54` 写明公平性声明；实测归档 `2026-08-18-decode-optimization.md:103-104` 记录 llama.cpp `llama-bench -ngl 99` 的 tg64 3.7 ms vs tiny-llm 6.087 ms。关键 commit `tiny-llm@753d913`（方法论文档）与 `f897084`（C2 快照）。`NUMBERS_CARD.md` §1 明确“非同量化”。
- 绝对不能说：tiny-llm 比 llama.cpp 快，或这是同一 kernel 的公平对比。
- 追问 1：那为什么还要比？ → 为了给数量级和优化路径一个外部参照；核心故事是 M==1 访存优化，不是打败 llama.cpp。
- 追问 2：如果都跑 Q4_K_M 会怎样？ → 我没有把 tiny-llm 推理路径改成 Q4_K_M 再跑；不报没测的数字。现有 GGUF 反量化测试只覆盖加载/参考一致性。

### D3. “2+2 is/equals”量化分歧
- 面试官最可能怎么问：请求 2 结果不一样，是不是你的引擎有 bug？
- 10 秒承认口径：不是调度 bug；是 W8A16 与 Q4_K_M 在 argmax 边界翻转，测试诚实断言为“前缀一致 + EOS 终止 + 分歧注释”。
- 证据式回答：`paged-infer/tests/tiny_llm_text_e2e.rs:159-166,272-280` 记录 llama.cpp `[17,10,17,16819,…,151645]`（equals）vs tiny-llm `[17,10,17,374,…,151645]`（is）。关键 commit `paged-infer@9c974d3`。请求 1 的 24 个 id 含 EOS 与 llama.cpp 全等（E21）。`NUMBERS_CARD.md` §7 有完整序列。
- 绝对不能说：我的引擎完全对齐 llama.cpp 所有请求，或这是“更懂中文”导致的差异。
- 追问 1：为什么第 4 个 token 才分叉？ → 前 3 个 token 两边 argmax 相同；第 4 个候选概率差在重量化边界附近，W8A16 vs Q4_K_M 取到不同 id。
- 追问 2：怎样才能全序列对齐？ → 两端接同一套量化/同一 kernel 再比；不是修调度能解决的。

### D4. cuflash causal skip 只有 ±2%（负结果）
- 面试官最可能怎么问：你这个 causal 优化提升多少？是不是越界优化很成功？
- 10 秒承认口径：不是成功优化；实测 256–4096 变化约 +1.2% 到 −1.9%，低于 10% 阈值，是噪声级负结果。
- 证据式回答：`cuflash-attn/docs/performance/causal-boundary-skip.md:32-41,65-72` 有 before/after 表；关键 commit `cuflash-attn@e1735b3`。保留改动是因为语义自文档化与减少无效访存，不是因为加速。`EVIDENCE_MATRIX.md` E8 和 `NUMBERS_CARD.md` §4 都按负结果记录。
- 绝对不能说：我们做了 causal 优化所以更快；±2% 是显著收益。
- 追问 1：为什么不回滚？ → 回滚省不下可测延迟，还会丢掉对无效访存的回归锁；文档已写明“增益低于噪声”。
- 追问 2：那主路径原来不是已经有 break 吗？ → 是，旧路径已跳过大部分未来块；新 skip 只覆盖旧 break 没处理的边界块，所以增量很小。

### D5. paged KV 第一版有 scatter/gather 额外往返
- 面试官最可能怎么问：分页 KV 是不是零拷贝？会不会比连续 KV 慢？
- 10 秒承认口径：策略 1 是正确性优先的实现，有 scatter 写 / gather 读的显存往返，没有做 kernel 级零拷贝。
- 证据式回答：`tiny-llm/tests/test_ffi.cpp:199` `FFITest.PagedKVStrategyMatchesContiguous` 证明策略 1 与连续 KV 逐 token 一致（E20）；`paged-infer/src/kv_cache.rs` 持 block table，经 FFI 上传（E19）。关键 commit `tiny-llm@7b456cd`。`QA_BANK.md` Q57 明确“代价：多一次显存往返；未接 Graphs”。
- 绝对不能说：paged KV 比连续 KV 快，或我们已经消掉 gather。
- 追问 1：能不能消掉？ → 能，但那是 Phase 4 D1（paged decode 直接读 pool），当前保持冻结；面试不声称已做。
- 追问 2：那为什么还要策略 1？ → 为了验证 ABI、block table、调度与真实模型差分；这是 serving 控制面的正确性证据，不是延迟优化。

### D6. 3 并发是学习验证，不是生产并发
- 面试官最可能怎么问：3 并发能说明你的 serving 有生产并发能力吗？
- 10 秒承认口径：不能。3 并发是 fixture 规模，用来验证调度 + ABI + 真实模型端到端正确性，不是 QPS 或容量规划。
- 证据式回答：`paged-infer/tests/tiny_llm_text_e2e.rs` 的 `qwen2_three_concurrent_paged_requests_match_llama_cpp`（E21），关键 commit `paged-infer@9c3700b`；请求 1 全序列 24+EOS，跑完 `active_sequences==0`。`tiny_llm_executor.rs` 还把最大并发 clamp 到 4（6GB 卡保护），说明这是学习验证规模。`QA_BANK.md` Q58 明确“没证明生产并发或吞吐”。
- 绝对不能说：3 并发 = 生产并发能力；或者这代表 vLLM 级别调度。
- 追问 1：为什么是 3？ → fixture 设计：覆盖多请求交错、块分配/释放、EOS 终止和资源回零；不是从吞吐模型推出来的。
- 追问 2：那你怎么证明调度不变量？ → 调度器有属性测试 `used+free==total`（E23），与 3 并发 e2e 是两层证据：单测锁资源守恒，e2e 锁真实模型 token 对齐。

### D7. cuda-foundations 209 collected 里有 78 skip
- 面试官最可能怎么问：你仓库 209 个测试都跑过了吗？
- 10 秒承认口径：没有全跑。CTest 是 0 failed / 209 collected，其中 78 skipped，实际执行 131 项。
- 证据式回答：`interview/FREEZE_AUDIT.md` §3.1 记录 `ctest --preset default` 输出：0 failed / 209 collected / 78 skipped；skip 集合含 `AdvancedTest.*`、`FusionTest.*`、`GemmTest.*` 等，多数是 02/04 GPU 二进制。`NUMBERS_CARD.md` §6 和 `QA_BANK.md` Q10 都写“不能说 209 全执行”。
- 绝对不能说：209 项全部通过，或 skip 不算数所以等于全跑。
- 追问 1：为什么 skip 这么多？ → 环境/构建门控：部分高级 GPU 二进制在本次 WSL2 环境未运行；这是环境现象，不是源码失败。
- 追问 2：那 cuda-foundations 还值得讲吗？ → 值得讲的是教学阶梯与负结果（E1），不是测试数量；面试旗舰仍是 tiny-llm + paged-infer。

### D8. 单卡 RTX 3060 Laptop、单模型 Qwen2.5-0.5B 的泛化边界
- 面试官最可能怎么问：这些数字换到 A100 / 更大模型 / 更长上下文还成立吗？
- 10 秒承认口径：不成立。所有实测基于 RTX 3060 Laptop 6GB、WSL2、单模型 Qwen2.5-0.5B；换卡、换模型、换 batch 规模都作废。
- 证据式回答：`NUMBERS_CARD.md` 开头写明硬件；§8“数字的边界”列单一硬件、单一模型；`FREEZE_AUDIT.md` §1 环境表；tiny-llm 第二真实模型测试 `SecondModelTest.*` 因没有 `TLLM_GGUF_TEST_MODEL_2` 而 skip（E14）。`EVIDENCE_MATRIX.md` E29 要求所有 benchmark 附硬件/commit/命令。
- 绝对不能说：我的引擎可以泛化到任意 GGUF 模型；或这个 TPOT 在 A100 上也差不多。
- 追问 1：那怎么证明不只在 0.5B 上 work？ → 有 GQA 14→2 真实路径测试（E14），但第二模型 skip；我只能诚实说“单一真实模型已验证，结构层测试覆盖映射”。
- 追问 2：如果面试官只给你 A100 你会怎么验证？ → 我会先重跑同一批命令并更新数字卡，因为原数字的硬件口径全部失效；不会沿用 3060 数字。
