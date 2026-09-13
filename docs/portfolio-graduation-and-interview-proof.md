# AI Infra 作品集毕业门槛与面试证据标准

> 本文回答一个问题：这些仓库在什么条件下可以安全地写入简历、用于现场演示，并经得起
> AI Infra 面试官沿代码、正确性、性能和系统边界继续追问。

本文是 live 标准；`interview/` 下的材料是历史归档，数字和能力状态仍应以技术仓当前
commit、结果包和 [`evidence-index.md`](evidence-index.md) 为准。

## 1. 最重要的结论

1. **不要再增加新项目。** 当前作品集已经覆盖 CUDA、Triton、Attention、LLM runtime、
   KV Cache、Rust serving 和上游 KV tiering。新增第八个仓库的收益远低于补齐当前证据。
2. **只选一个深改造。** 默认选择 `tiny-llm` direct paged decode attention；若目标更偏
   kernel，则选择 `cuflash` 可重入 workspace/stream safety。两项同时开工会稀释证据。
3. **正确性比漂亮数字优先。** GPU case 真正执行、独立 reference、sanitizer、失败路径和
   raw artifact，比一次较大的 speedup 更能通过面试追问。
4. **旗舰叙事只保留两条。**
   - `tiny-llm + paged-serving`：从 HTTP 请求到 CUDA decode 的系统链路。
   - `cuflash + trifuse`：从算法、数值到 CUDA/Triton 工程取舍。
5. **辅助仓不争夺主叙事。** `cuda-foundations` 证明基础和优化方法；`kvtier` 证明上游
   研究与实验能力，不包装成自研生产系统。

## 2. 证据等级

任何 README、简历 bullet、面试陈述和性能图都应标注或能够映射到下列等级。

| 等级 | 名称 | 最小要求 | 可以声称 | 不能声称 |
|------|------|----------|----------|----------|
| E0 | 想法 | roadmap、issue、设计草稿 | 计划研究/实现 | 已实现 |
| E1 | 代码存在 | 可定位 symbol，能 build | 已编写某实现 | 正确、可用、性能更好 |
| E2 | CPU/静态验证 | build、lint、CPU reference、schema tests | 接口和离线逻辑通过回归 | GPU correctness |
| E3 | GPU correctness | 固定 commit/GPU，非 skip 差分测试 | 在记录矩阵内数值正确 | 更快、生产可用 |
| E4 | 可复现性能 | E3 + raw samples + metadata + repetitions | 在记录环境/工作负载下的观察 | 跨 GPU/模型普遍结论 |
| E5 | 系统证据 | E4 + 真实请求、故障、资源生命周期、外部 baseline | 指定系统场景的容量/延迟边界 | 通用生产 SLO |

升级规则：

```text
E0 → 设计批准
E1 → build/lint
E2 → 独立 reference + 负向测试
E3 → 真实 GPU 非 skip + sanitizer
E4 → raw benchmark + provenance + 收敛
E5 → 端到端 workload + failure injection + 公平 baseline
```

禁止跨级。例如：E1 的 kernel 不能直接写成 E4 的性能成果；E2 的 mock server 不能直接
写成 E5 的真实 serving 能力。

## 3. 通用“毕业”定义

一个项目只有同时满足以下条件，才适合作为简历主项目：

### 3.1 可运行

- clean checkout 有明确 build/install 命令；
- 依赖和版本范围固定；
- CPU-only 与 GPU-required 命令分开；
- 无 GPU、模型或 profiler 时状态明确为 `blocked`、`skipped` 或 `not_measured`；
- CI 的绿色状态不依赖隐藏的本地文件。

### 3.2 可证明

- 核心算法有独立 reference；
- normal、ragged、tail、invalid 和 failure case；
- GPU lane 断言至少一个目标 case 真正运行；
- memory-safety 工具或等价检查；
- 失败日志和负结果保留。

### 3.3 可解释

- 能从 public API 讲到关键 kernel/状态机；
- 能画出数据布局、ownership 和生命周期；
- 能解释一个真实 bug、一个负优化结果和一个设计取舍；
- 能说明当前不支持的输入和为什么。

### 3.4 可复现

- exact commit 和 dirty state；
- GPU、driver、CUDA、compiler/framework；
- model/tokenizer/revision/hash；
- dtype、quantization、shape/workload、seed；
- warmup、repetitions、raw data、命令和 limitations。

### 3.5 可防守

- README 声明不超过证据；
- 面试官能从数字进入 raw artifact；
- 面试官能从功能声明进入代码和测试；
- 对生产成熟度、跨硬件泛化和未完成路线不夸大；
- 能明确回答“为什么不用成熟库”和“这个练习项目的学习价值在哪里”。

## 4. 六仓毕业矩阵

### 4.1 `cuda-foundations`

**角色**：CUDA 学习路径和优化方法证据，不是生产 GEMM 库。

当前安全说法：

- 实现从 naive 到 shared-memory、double-buffer、WMMA 的教学阶梯；
- 在记录的 RTX 3060 Laptop 快照上运行 correctness 和 benchmark；
- 能用 cuBLAS 差距说明自研教学 kernel 与生产库的边界。

禁止说法：

- “性能超过 cuBLAS”或“生产级 SGEMM”；
- 仅凭 variant 名称声称消除了 bank conflict 或实现了 load/compute overlap；
- 将 FP32 wrapper 的 allocation/conversion 时间称为纯 Tensor Core kernel 时间。

毕业前最高价值补强：

1. GPU workflow 拒绝 zero-test/unexpected skip；
2. ragged `M/N/K`、NaN/Inf 和 alpha/beta correctness；
3. Compute Sanitizer artifact；
4. Tiled/Bank Conflict Free/Double Buffer 的 Nsight 对照；
5. pure WMMA 与 conversion wrapper 分开计时。

面试官应能追问：

- 为什么 tiled 后仍可能 memory-bound？
- shared memory bank conflict 如何从 profiler 观察，而不是从源码名字猜？
- occupancy 提高为什么不一定让 kernel 更快？
- WMMA 的输入、累加和输出 dtype 分别是什么？

### 4.2 `trifuse`

**角色**：Triton kernel、PyTorch custom op 和快速实验工程证据。

当前安全说法：

- 已有 Triton SGEMM/fused op 和 `torch.library.custom_op`；
- 已注册 fake implementation 并覆盖 `torch.compile` 路径；
- CPU-only skip 与真实 GPU 测试分开记录。

禁止说法：

- 没有可执行测试时声称完整支持 `torch.export`；
- 把 Python API 存在写成 `torch.ops.trifuse` 已注册；
- 用不同工作量的 PyTorch baseline 计算 speedup；
- 把 online-softmax 实验 kernel称为生产 FlashAttention。

毕业前最高价值补强：

1. eager/fake 的 shape、dtype、device、stride 错误一致性；
2. `torch.export` 测试或降级声明；
3. raw per-iteration samples 与 benchmark provenance；
4. Triton 与 PyTorch/cuBLAS/SDPA 同口径 baseline；
5. 明确 FlashAttention 是否进入 custom-op namespace。

面试官应能追问：

- `custom_op`、fake/meta、autograd 注册分别解决什么问题？
- Triton block size 和 warps 如何选择？
- 为什么同一个 op 在 eager 正确却可能 export 失败？
- 什么时候选择 Triton，什么时候必须写 CUDA C++？

### 4.3 `cuflash`

**角色**：CUDA Attention 深挖和数值/并发设计证据。

当前安全说法：

- 实现 FlashAttention 风格 forward/backward、online softmax 和 FlashDecoding；
- FP16/BF16 forward 有 WMMA 路径；
- 有 CPU/PyTorch reference、边界测试和真实 GPU 测试快照。

禁止说法：

- 等价于完整 FlashAttention-2/3；
- 没有 dispatch/profiler 证据时宣称所有 FP16/BF16 case 都使用 Tensor Core；
- 当前函数级 static scratch 仍存在时宣称多 stream/thread-safe；
- 单个 kernel benchmark 等同端到端 LLM 性能。

毕业前最高价值补强：

1. decode workspace ownership 和 stream lifecycle 设计；
2. 移除或安全隔离 shared static scratch；
3. BF16、chunk tail、极端 score、多 stream 和 OOM/launch error；
4. 当前 commit 的 GPU correctness + sanitizer artifact；
5. forward/backward/decode 的公平 raw benchmark。

面试官应能追问：

- online softmax 的 `m/l/O` 如何合并？
- Split-KV combine 为什么数值稳定？
- caller stream 与 workspace 复用的 happens-before 如何建立？
- 哪些错误能在 launch 后同步发现，哪些只能在 stream sync 时发现？

### 4.4 `tiny-llm`

**角色**：旗舰数据面，证明真实权重 runtime、量化、KV 和 CUDA decode。

当前安全说法：

- 能加载记录范围内的 GGUF/量化模型并完成单 GPU 推理；
- 有连续 KV 和 paged KV storage/control path；
- CUDA Graph A/B 有绑定 commit、模型和 raw data 的历史证据；
- C ABI 支持 paged-serving 同进程调用。

禁止说法：

- 当前 paged scatter/gather 路径是 direct PagedAttention；
- W8A16 是 Tensor Core INT8，除非对应 kernel 和 dispatch 有证据；
- 一个真实模型通过意味着所有 GGUF 架构/量化都兼容；
- CUDA Graph 的 TPOT 观察自动代表 TTFT 或 serving 吞吐提升。

毕业前最高价值补强：

1. GGUF/量化支持矩阵和第二模型；
2. 不依赖外部模型的 paged/contiguous synthetic oracle；
3. 不依赖私有 GGUF 的 CUDA Graph correctness；
4. direct paged decode kernel；
5. Transformer/FFI 集成和 legacy fallback；
6. 长上下文三路 A/B。

面试官应能追问：

- `max_num_blocks == 0` 的策略语义是什么？
- GQA 中 query head 如何映射 kv head？
- paged storage 和 direct paged compute 有何区别？
- CUDA Graph capture 对地址稳定性和动态 shape 有什么约束？
- 量化误差和运行时瓶颈分别在哪里？

### 4.5 `paged-serving`

**角色**：旗舰控制面，证明调度、KV block、Rust FFI、HTTP/SSE 和负载评测。

当前安全说法：

- 有 pending/admission/decode 调度、BlockPool/page table、429 和 SSE；
- 有 closed-loop/Poisson loadgen 和绑定原始请求的真实 CUDA 结果包；
- 有 cancellation 入口、metrics 和真实 tiny-llm backend 测试入口；
- 当前正式矩阵包含未收敛和 429 负结果。

禁止说法：

- scheduler continuous batching 等于 fused GPU batch compute；
- SSE chunk latency 等于 token-level ITL；
- 没有外部引擎同口径 baseline 时声称领先 vLLM/llama-server；
- 有 21-run 矩阵就代表稳定生产 SLO；
- mock backend 结果代表真实 GPU serving。

毕业前最高价值补强：

1. consumer ownership 驱动的主动 cancellation；
2. SSE 和 `n>1` fan-in 有界背压；
3. 指标生命周期数值测试；
4. loadgen 真实 HTTP/SSE failure tests；
5. tiny-llm feature 的非 skip GPU lane；
6. `/metrics` sampling 和 cancel/HOL/fairness workload；
7. 三引擎公平矩阵。

面试官应能追问：

- 请求在哪个时点占用 sequence 和 KV blocks？
- client disconnect 如何做到 exactly-once release？
- 无界 channel 的风险是什么，为什么不能简单让 engine await？
- TTFT、TPOT、inter-chunk latency 和 throughput 如何定义？
- 429 是准入策略、容量不足还是测试错误？

### 4.6 `kvtier`

**角色**：上游阅读、复现和可审计实验脚手架，不是自研 HiCache engine。

当前安全说法：

- 研究并定位 SGLang HiCache/KV tiering 数据流；
- 提供 W/E/R/P workload、server snapshot、counter/cache validation 和结果审计；
- 当前真实 GPU benchmark 尚未完成，因此保留为实验脚手架。

禁止说法：

- 自研了完整 KV tiering；
- run 末全局 counter 增加就证明每个 R 都从 host 正确回载；
- 固定 sleep 证明 backup 已完成；
- 单卡 L2 脚本等价于多卡、L3、speculative decoding 或 #31600 完整复现。

毕业前最高价值补强：

1. pinned upstream commit 和 symbols 的复核日期；
2. schema v2 provenance；
3. per-phase host reload correctness；
4. CPU-only CI；
5. 一个真实 GPU 单卡 L2 baseline；
6. IO backend/KV dtype 矩阵；
7. 多卡/L3 只保留 design-only，直到资源批准。

面试官应能追问：

- 如何证明 cache hit 是本请求的，而不是全局 counter 噪声？
- L2 host DRAM、L3 storage 的延迟和带宽瓶颈分别是什么？
- eviction、prefetch、load-back 如何与 compute overlap？
- 为什么复现上游 issue 仍然有工程价值？

## 5. 旗舰系统证据链

`tiny-llm + paged-serving` 的系统叙事必须沿一条真实请求展开：

```text
HTTP request
  → 参数校验与 admission
  → scheduler request/sequence state
  → BlockPool 与 block table
  → Rust `repr(C)` mirror
  → tiny-llm C ABI allocate/step/free
  → Transformer layer
  → KV scatter/gather 或 future direct paged decode
  → GPU logits/argmax
  → token/text postprocess
  → SSE/unary response
  → success/cancel/error cleanup
```

每一箭头至少需要一种证据：

| 边界 | 最低证据 |
|------|----------|
| HTTP → admission | 4xx/429/accepted integration tests |
| request → sequence | state transition 和 exactly-once terminal tests |
| sequence → KV blocks | baseline 前后资源计数 |
| Rust → C | size/alignment/field offset 和参数契约 |
| C → CUDA | 真实 backend 非 skip test |
| CUDA → output | independent reference / token-logit differential |
| output → SSE | parser、`[DONE]`、disconnect 和 protocol failure tests |
| terminal → cleanup | success/cancel/timeout/error 后资源回基线 |

如果其中一段只有 mock，应明确说“该段只验证控制面”；不能将整条链路称为真实 GPU e2e。

## 6. 三种目标岗位的最低组合

### Kernel / GPU Performance

必须具备：

- `cuflash` 一个可白板推导的 kernel；
- `cuda-foundations` 一个从 baseline 到 profiler 的优化故事；
- `trifuse` 一个 CUDA/Triton 取舍故事；
- 真实 GPU correctness + sanitizer；
- 一个有 raw `ncu/nsys` 的正或负性能结论。

加分但非必需：

- direct paged attention；
- WMMA dispatch 证据；
- upstream kernel contribution。

### LLM Runtime

必须具备：

- `tiny-llm` 的 model loading、quantization、Transformer、KV、decode；
- paged/contiguous 的差分 oracle；
- CUDA Graph 的约束和 fallback；
- `tiny-llm ↔ paged-serving` ABI；
- 一个真实模型和第二种模型/几何证据。

加分但非必需：

- direct paged decode；
- batch compute；
- profiler 解释 decode memory traffic。

### Serving / Inference Systems

必须具备：

- `paged-serving` request lifecycle；
- admission、429、backpressure、cancellation 和 cleanup；
- closed-loop 与 open-loop 的区别；
- TTFT/TPOT/throughput/error 的定义；
- 真实 tiny-llm backend；
- 一份带负结果和收敛判断的 serving matrix。

加分但非必需：

- 外部引擎 baseline；
- metrics time series；
- HOL/fairness/failure-injection；
- 多 GPU 理论设计。

## 7. 面试展示脚本

### 7.1 90 秒

1. 先说两条旗舰主线，不逐仓报菜名。
2. 展示组织 README 的架构和证据索引。
3. 选择一个真实结果，只说硬件、workload、观察和限制。
4. 主动说明一个未完成边界，例如当前 paged path 尚需 direct compute。

### 7.2 10 分钟

```text
1 分钟  问题与仓库分工
2 分钟  一条请求或一个 kernel 的数据流
2 分钟  最难的 correctness/ownership 问题
2 分钟  实验设计和 raw evidence
2 分钟  一个负结果或失败恢复
1 分钟  当前限制和下一步
```

### 7.3 现场 demo

演示优先级：

1. 打开固定 commit 的证据报告和 raw artifact；
2. 运行短小、确定性的 CPU/build test；
3. 有可靠 GPU 时运行 30–90 秒 correctness canary；
4. 展示已有 profiler report；
5. 最后才运行短 benchmark。

不要在面试现场：

- 从头编译大型依赖；
- 下载模型；
- 运行长矩阵；
- 临时安装 driver/CUDA；
- 只展示一张没有 metadata 的速度截图。

## 8. 简历声明模板

正确结构：

```text
实现 <具体机制>，通过 <独立 reference / failure matrix / GPU gate> 验证；
在 <GPU、模型、shape、commit> 下，以 <baseline> 为对照完成 <重复方式>，
观察到 <具体指标>，限制为 <边界/未收敛/不外推内容>。
```

如果还没有性能证据：

```text
实现并验证 <机制>，覆盖 <关键边界与错误路径>；
建立可复现 benchmark/profiling harness，性能结论待真实 GPU 矩阵完成。
```

禁止结构：

```text
全面优化 LLM 推理，性能提升 2x，达到生产级。
```

除非“全面”“2x”“生产级”分别有完整系统证据和适用范围。

## 9. 最终发布检查

在置顶仓库或投递前逐项确认：

1. 组织 README 中只显示当前仓名和状态；
2. 每个置顶仓 README 有一句定位、边界、build/test、证据和限制；
3. 所有性能数字能进入 raw artifact；
4. 结果绑定 clean commit 或明确 dirty diff；
5. CPU/build、GPU correctness、performance 分层；
6. skip 不计入通过；
7. 失败/OOM/429/not-converged 保留；
8. `tiny-llm`/`paged-serving` ABI 两侧一致；
9. archive 的旧数字不被复制为当前状态；
10. 简历 bullet 与仓库当前证据逐字对齐；
11. 能在 90 秒、10 分钟和 30 分钟三种长度讲同一事实；
12. 至少排练一次“没有 GPU 的现场演示”。

## 10. 机会成本与停止规则

应停止的工作：

- 新建相似 kernel 或第八个练习仓；
- 在 correctness 未完成前反复追逐 microbenchmark；
- 同时实现 direct paged、workspace 重构和多 GPU；
- 为了 README 数字改变测试或过滤失败；
- 用低成本 Agent 决定 ABI、并发和数值 contract；
- 长期维护重复的路线图副本。

应优先投入：

```text
真实 GPU 非 skip gate
  > 独立 reference 与 failure matrix
  > 一个深改造
  > profiler 归因
  > 公平外部 baseline
  > README 与简历收口
```

作品集的目标不是证明“什么都做过”，而是证明能够定义边界、建立证据、解释系统，并对
尚未完成的部分保持技术诚实。
