# Runtime 岗位角色手册

> Phase I6。目标岗位偏 **runtime / 推理引擎 / 部署** 时的讲述顺序与重点。
> 同一份材料重排：主推 tiny-llm（decode 优化、CUDA Graphs、FFI、paged KV）+ cuflash 集成点；serving 只讲边界。

## 一句话定位

“我是做 runtime 的：一条命令从 GGUF 跑到真实模型，decode 路径的每个 kernel 我都有 microbench，控制面通过 C ABI v2 接进来。”

## 20 分钟讲述时间轴

| 分钟 | 讲什么 | 用到的 QA_BANK / 数字 |
|------|--------|------------------------|
| 0–2 | 定位 + 端到端故事：“一条命令从 GGUF 到文本” | Q33/Q10 |
| 2–5 | 模型层：GGUF 反量化、W8A16、per-group scale、GQA 14→2 | Q33/Q34/Q35/Q36/Q37 |
| 5–9 | decode 优化核心：转置 M==1、CUDA Graphs、`append_kv_at` bug、TTFT/TPOT 口径 | Q41/Q42/Q39/Q40/Q43；24.348→6.560→6.087 ms |
| 9–12 | ABI 与正确性：FFI 9-int 布局、is/equals 量化分歧 | Q45/Q44/Q46 |
| 12–15 | cuflash 集成点：FP16/BF16 数值、LogicalHBM 带宽口径、反向稳定性——为什么 runtime 不接 bwd | Q29/Q30/Q32 |
| 15–18 | serving 边界：我只负责后端正确性，并发/准入/水位线是控制面的事 | 引用 serving 手册（Q47–Q60），陈述边界结论 |
| 18–20 | 反问 + 边界声明：“paged 路径没接 Graphs；第二真实模型 skip” | Q42 的 OUT / 数字卡 §8 |

## 推荐的 QA_BANK 题号子集（17 题）

`Q29, Q30, Q32, Q33 – Q46`（4 个 FA 数值/边界题 + tiny-llm 全套 Q33–Q46）。理由：runtime 岗把 tiny-llm 全部问题讲透，另外补 FA 的数值与带宽口径作为“我要不要接”的判断依据。

## 简历条目重排顺序

1. `tiny-llm › 1`：一条命令从 GGUF 跑通真实 Qwen2.5-0.5B（E10）——先证明端到端可用。
2. `tiny-llm › 2`：tokenizer 与 HF 逐 id 对齐 30 例 417 token（E11）——证明正确性。
3. `tiny-llm › 3`：CUDA Graphs 默认捕获 decode、on/off greedy 一致（E16）——证明性能路径。
4. `总览 › 2`：转置 M==1 GEMM，TPOT 24.348→6.087 ms、相对 llama.cpp 1.65×（E15）——旗舰数字。
5. `paged-serving › 1`：C ABI v2 九整型布局、Rust `sizeof==36` 守卫（E19）——证明跨语言边界由我定义。
6. `总览 › 3`：3 并发与 llama.cpp greedy 对齐、量化分歧如实记录（E21）——作为后端正确性的收尾。

## 反问问题清单（≥3）

1. 你们 runtime 的 decode 性能证据链是什么？我这边 ncu 不可用，靠 `tiny_llm_kernel_bench`，你们呢？
2. CUDA Graphs 在你们 worker 上捕获范围怎么定，有没有 device 参数化的坑？
3. 控制面和 kernel 仓的 ABI 谁拥有？我设计过 `repr(C)` 9-int 布局 + 守卫测试，想对齐你们的做法。
4. 如果 paged KV 要接 CUDA Graphs，你们更看重复用还是正确性门槛？这块是我明确的下一增量。
