# 七仓系统搭建关系、集成路径与面试证据地图

更新时间：2026-09-13。

本文回答三个问题：

1. 七个仓库究竟是什么关系，哪些会一起运行，哪些只是学习或对照关系？
2. 从空环境开始，应该以什么顺序构建、验证和集成？
3. 每个阶段完成后，面试时可以证明什么，不能夸大成什么？

逐仓任务 ID、允许文件、验收命令和 Agent 停止条件由
[`P0_P1_AGENT_BACKLOG.md`](https://github.com/open-infra-ai/ai-infra-interview-prep/blob/main/P0_P1_AGENT_BACKLOG.md)
维护；L3/L4 设计门禁由
[`L3_L4_DESIGN_REVIEW_PACKAGES.md`](https://github.com/open-infra-ai/ai-infra-interview-prep/blob/main/L3_L4_DESIGN_REVIEW_PACKAGES.md)
维护。本文只定义稳定的跨仓架构和证据关系，不复制任务实现细节。

---

## 1. 先纠正一个常见误解

七个仓库不是一个需要统一编译的 monorepo，也不是七个平级产品。

真正的运行时硬依赖只有：

```text
paged-serving
    │ Rust FFI declaration
    ▼
tiny-llm C ABI
    │ model/runtime data plane
    ▼
CUDA kernels + KV storage + decode
```

其余仓库与旗舰系统的关系是：

```text
cuda-foundations ── 提供 CUDA/GEMM/性能分析基础，不进入生产链接

cuflash ────────── 提供 CUDA Attention/FlashDecoding 深挖和对照，
                    当前不应声称已直接替换 tiny-llm attention

trifuse ────────── 提供 Triton + PyTorch custom op 横向表达，
                    当前不进入 tiny-llm C++ runtime

kvtier ─────────── 提供 SGLang HiCache/KV tiering 上游研究，
                    当前不进入 paged-serving 的生产数据路径

open-infra-ai ──── 提供导航、契约、证据索引和状态治理，
                    不参与请求执行
```

因此面试时必须使用“旗舰系统 + 深挖实验 + 研究证据”的结构，而不是声称七仓已经组成一个
生产平台。

---

## 2. 七仓职责矩阵

| 仓库 | 系统层 | 核心所有权 | 主要输入 | 主要输出 | 不负责 |
|------|-------|-----------|---------|---------|--------|
| `cuda-foundations` | GPU 基础层 | CUDA 编程模型、SGEMM 优化阶梯、measurement discipline | 矩阵、shape、kernel 配置 | correctness、latency、roofline/profiler 解释 | LLM Serving、生产算子库 |
| `trifuse` | 框架算子层 | Triton 实现、`torch.library`、fake/meta/export、框架契约 | PyTorch tensors、shape/dtype/device | eager/fake/export 结果、Triton benchmark | C++ runtime、HTTP Serving |
| `cuflash` | CUDA Attention 层 | FlashAttention/FlashDecoding、workspace、stream、数值与 dispatch | Q/K/V、mask、scale、stream | attention output、LSE、kernel evidence | 请求调度、模型加载 |
| `tiny-llm` | Runtime 数据面 | GGUF、量化、Transformer、KV、decode、C ABI | 模型、token ids、KV/block table | logits/token、runtime metrics/error | HTTP、限流、SSE |
| `paged-serving` | Serving 控制面 | admission、scheduler、BlockPool、请求生命周期、HTTP/SSE | 请求、并发、到达率、生成参数 | token stream、TTFT/TPOT、状态和错误 | kernel 数值实现、权重解析 |
| `kvtier` | KV 研究层 | SGLang HiCache/tiering 的可审计实验和上游跟踪 | pinned upstream、workload、IO backend | correctness、IO/KV dtype matrix、限制 | 自研生产 tiering engine |
| `open-infra-ai` | 作品集治理层 | 项目边界、跨仓契约、evidence manifest、公开索引 | 各仓 verified artifacts | 可审计证据链、面试入口 | 技术实现和 benchmark 生成 |

---

## 3. 两条旗舰主线

### 3.1 旗舰系统：`tiny-llm + paged-serving`

这是面试时最重要的一条系统故事：

```text
OpenAI-compatible request
  → HTTP validation
  → admission / rate limit
  → request state machine
  → continuous-batching scheduler
  → BlockPool / block table
  → tiny-llm C ABI
  → Transformer decode
  → KV access / attention / GEMM / sampling
  → token result
  → bounded SSE channel
  → client
```

控制面与数据面的边界：

| 主题 | `paged-serving` | `tiny-llm` |
|------|-----------------|------------|
| request ownership | owner | 不感知 HTTP |
| admission/429 | owner | 返回 runtime error，不决定限流 |
| scheduling | owner | 执行被选中的 decode step |
| block allocation policy | owner | 消费 block table 和 KV address |
| model weights | 不解析 | owner |
| kernel dispatch | 不决定 | owner |
| cancellation | 触发并回收 request/blocks | 停止或拒绝后续 step |
| streaming | owner | 只返回 token/result |
| GPU correctness | 通过真实 backend 集成验证 | 主要 owner |
| service SLO | 主要 owner | 提供 runtime 分解指标 |

### 3.2 Kernel 深挖：`cuflash + trifuse + cuda-foundations`

这条主线不是第二个 Serving 系统，而是回答“为什么快、为什么对、如何集成”：

```text
cuda-foundations
  └─ 建立 memory hierarchy / tiling / WMMA / measurement 基础

cuflash
  └─ 用 CUDA C++ 深挖 attention、online softmax、workspace、stream、dispatch

trifuse
  └─ 用 Triton + PyTorch custom op 表达相同类别算子，
     证明 framework contract、fake/meta、export 和开发效率取舍
```

面试时可以比较：

- CUDA 与 Triton 的开发边界；
- 手工 shared memory/warp/tensor core 控制与编译器生成之间的取舍；
- standalone microbenchmark 与真实 decode 热点的差别；
- eager、fake、export、compile 与 C ABI runtime 的不同集成面；
- correctness tolerance、mask/layout 与 benchmark timing boundary。

不能声称：

- `cuflash` 已经是 `tiny-llm` 的生产 attention，除非源码和 profile 证明实际 dispatch；
- Triton 结果可以直接代表 CUDA 结果；
- 某个 microbenchmark 加速比等于端到端 tok/s 改善。

---

## 4. 逻辑依赖、构建依赖和证据依赖

### 4.1 逻辑依赖

逻辑依赖表示“先学或先理解”，不表示 linker dependency：

```text
cuda-foundations
  → cuflash
  → tiny-llm attention/KV optimization

trifuse
  → PyTorch integration comparison

tiny-llm
  → paged-serving real backend

kvtier
  → future KV reuse/tiering/disaggregation discussion
```

### 4.2 构建依赖

| 组合 | 是否必须一起构建 | 原因 |
|------|------------------|------|
| `tiny-llm` + `paged-serving` | 集成测试和真实 demo 时必须 | Rust 声明与 C header/动态库必须匹配 |
| `cuflash` + `tiny-llm` | 否 | 当前是算法和性能对照；只有正式接入后才成为依赖 |
| `trifuse` + `tiny-llm` | 否 | Python/PyTorch 与 C++ runtime 是不同执行环境 |
| `cuda-foundations` + 任意仓 | 否 | 教学和 profiler 基线 |
| `kvtier` + 旗舰系统 | 否 | pinned SGLang 研究 lane |
| `open-infra-ai` + 任意仓 | 否 | 只消费结果和 manifest |

### 4.3 证据依赖

性能结论必须依赖更低层证据：

```text
build
  → reference correctness
  → GPU non-skip correctness
  → sanitizer / failure-path checks
  → raw benchmark
  → profiler attribution
  → end-to-end serving matrix
  → resume/interview claim
```

缺少任意前置时，只能停留在对应证据等级，不能跳级。

---

## 5. 从空环境开始的推荐搭建顺序

### Phase A：独立仓可信构建

目标：证明每个仓的构建、测试和 skip 语义真实可信。

顺序：

1. `cuda-foundations`：CPU/build 与 GPU tests 分开；无 GPU 必须明确 skip/fail；
2. `trifuse`：安装匹配的 PyTorch/Triton 环境，验证 eager/fake/export 路径；
3. `cuflash`：构建 CUDA targets，先跑小形状 reference，再跑 sanitizer；
4. `tiny-llm`：构建 runtime/tests，确认模型无关 synthetic tests；
5. `paged-serving`：构建 Rust service，先用 mock/reference backend 验证协议；
6. `kvtier`：固定上游 SGLang commit，先跑 CPU-only/schema/audit；
7. `open-infra-ai`：只检查链接、schema 和 evidence index。

Phase A 的交付物：

- exact commit 和 dirty state；
- toolchain/GPU/driver metadata；
- build/test command 和 exit code；
- passed/failed/skipped 数量；
- 不需要模型/GPU的最小 smoke test；
- 已知限制。

### Phase B：GPU correctness 与生命周期

目标：证明代码真实执行在 GPU 上，并覆盖越界、数值和资源回收。

优先链：

```text
CUDA-P0-001..003
  → CUF-P0-001..004
  → TLLM-P0-002/003
  → TRI-P0-001/002/008
```

必须观察：

- 真实 device identity；
- test 没有静默 skip；
- non-divisible、short/long、zero/invalid 参数；
- sanitizer 输出；
- stream/concurrency；
- allocation/free 和 error path；
- independent reference，而不是同算法重写。

### Phase C：旗舰数据面

目标：从“分页存储 + gather 到连续 scratch”升级为真实 direct paged decode。

顺序：

```text
TLLM-P0-002 independent paged/contiguous oracle
  → G0–G8 direct-paged design review
  → TLLM-P0-004 kernel
  → TLLM-P0-005 Transformer + C ABI
  → TLLM-P1-001 long-context A/B
```

完成后必须能够回答：

- block table 如何映射 layer/request/kv_head/position/dim；
- last block 非整除如何 mask；
- GQA/MQA 的 query head 到 KV head 如何映射；
- 为什么 direct path 可能减少 HBM traffic；
- 间接寻址、分支和低 occupancy 可能造成什么代价；
- fallback 条件是什么；
- profile 是否证明真实走了 direct path。

### Phase D：旗舰控制面

目标：证明服务在并发、取消、慢客户端和失败时仍保持资源正确。

顺序：

```text
PSRV-P0-001 cancellation
  → PSRV-P0-002 bounded backpressure
  → PSRV-P0-003 metric semantics
  → PSRV-P0-004 HTTP/SSE failure regression
  → PSRV-P1-002 real tiny-llm backend
  → PSRV-P1-003 telemetry/fairness/HOL
  → PSRV-P1-004 serving matrix
```

完成后必须能够展示：

- client disconnect 后请求停止推进；
- BlockPool、request table、channel 和 runtime state 回到基线；
- 慢客户端不会无限占用内存；
- queue wait、prefill、decode、streaming 的 timing boundary；
- 429/5xx/cancel/timeout 的独立计数；
- concurrency/arrival rate 上升时的 TTFT p95/p99、TPOT、throughput、error rate；
- 与 baseline 的配置和量化差异。

### Phase E：性能与公开证据

每个正式结果包采用：

```text
results/<date>-<hardware>-<subject>/
  manifest.json
  README.md
  raw/
  derived/
  checks/
```

只有 `verified` 结果可以进入项目 README 的当前结论；只有 `published` 结果可以进入组织证据
索引和简历。代码、toolchain、workload 或 timing boundary 变化后必须检查是否 `stale`。

---

## 6. 端到端 demo 拓扑

### 6.1 最小 correctness demo

```text
fixed prompt
  → paged-serving HTTP
  → tiny-llm deterministic greedy decode
  → fixed token ids
  → SSE response
```

用途：

- 证明 API、scheduler、ABI、runtime、tokenizer 和 decode 能连接；
- 不用于声称容量或性能。

### 6.2 生命周期 demo

```text
start long generation
  → client disconnect
  → cancellation propagates
  → scheduler removes request
  → BlockPool/runtime allocations return to baseline
```

用途：

- 证明 ownership、cancellation、resource cleanup；
- 可以回答生产系统故障路径问题。

### 6.3 性能 demo

```text
fixed workload distribution
  → warmup
  → repeated independent processes
  → concurrency / arrival-rate sweep
  → raw request records
  → aggregate TTFT/TPOT/tok/s/p95/p99/error/memory
  → profiler attribution
```

用途：

- 证明系统容量和瓶颈；
- 不把单个 kernel latency 直接换算为系统收益。

### 6.4 Kernel 对照 demo

```text
same Q/K/V + mask + dtype + shape
  ├─ independent reference
  ├─ cuflash CUDA path
  └─ trifuse Triton path
       → correctness first
       → raw latency distribution
       → Nsight/Torch profiler
       → explain wins and losses
```

用途：

- 比较实现模型与开发取舍；
- 不要求某一实现所有 shape 都赢。

---

## 7. 阶段成果与安全声明

| 阶段 | 已完成条件 | 可安全声明 | 禁止声明 |
|------|-----------|-----------|---------|
| S0 定位 | 仓库边界、当前 symbol、限制可定位 | “我审计并定义了实现边界” | “已实现 roadmap” |
| S1 可构建 | clean commit build/lint/test 可复现 | “项目可在记录环境构建并通过这些检查” | “GPU 正确/高性能” |
| S2 正确性 | independent reference、GPU non-skip、负向/边界测试 | “在列出的 shape/dtype/tolerance 上与 reference 对齐” | “支持所有输入/生产稳定” |
| S3 生命周期 | sanitizer、concurrency、cancel/error/resource baseline | “覆盖了指定故障和资源回收路径” | “高可用/无内存问题” |
| S4 性能 | raw samples、metadata、收敛、profile、comparison key | “在固定环境和 workload 下观察到该结果” | “全面快于某框架” |
| S5 系统 | 真实请求、真实 backend、容量曲线、失败注入、公平 baseline | “完成端到端 serving 证据链” | “生产规模、云规模、多机实测” |
| S6 外部验证 | 上游 issue/PR/review 或第三方复现 | “结果得到外部协作/复核” | “PR 未合并却声称上游采用” |

---

## 8. 面试时如何把七仓讲成一个系统

### 8.1 30 秒

> 我没有把项目做成七个互不相关的 demo，而是收敛成两条线。`tiny-llm` 是单 GPU
> C++/CUDA runtime，`paged-serving` 是 Rust Serving 控制面，两者通过双源 C ABI
> 形成真实请求到 decode 的旗舰链路。`cuflash`、`trifuse` 和 `cuda-foundations`
> 用于证明 CUDA/Triton kernel、框架集成和 profiler 深度；`kvtier` 是对 SGLang KV
> tiering 的可审计研究；组织仓负责证据生命周期，防止把测试、性能和生产结论混在一起。

### 8.2 两分钟

按以下顺序：

1. **问题**：单卡 LLM Serving 的延迟和显存受 KV、decode 和调度共同影响；
2. **架构**：Rust 控制面与 C++/CUDA 数据面分离；
3. **深改造**：从 gather-to-contiguous 改为 direct paged decode；
4. **正确性**：independent oracle、non-divisible/GQA/mask、sanitizer；
5. **可靠性**：cancel、disconnect、backpressure、资源回基线；
6. **性能**：TTFT/TPOT/吞吐/尾延迟/显存 + Nsight；
7. **限制**：单卡、模型/量化差异、多 GPU 只做理论或受限实验；
8. **迁移能力**：用 CUDA/Triton 对照和 evidence manifest 说明如何验证新优化。

### 8.3 十分钟白板

建议画四层：

```text
API: HTTP / SSE / errors
Control: admission / scheduler / lifecycle / BlockPool
Runtime: model / ABI / Transformer / KV / sampling
Hardware: kernels / memory / streams / GPU
```

然后选择一个纵向问题贯穿四层，例如：

- 客户端取消如何释放 GPU/KV 资源；
- block size 如何同时影响调度碎片和 attention 寻址；
- direct paged attention 为什么可能改善或恶化 TPOT；
- 慢客户端如何造成 head-of-line blocking；
- 某个 kernel 优化为什么没有转化为端到端收益。

---

## 9. 面试官常见追问与证据入口

| 追问 | 首选仓库 | 必须展示的证据 |
|------|---------|---------------|
| 你如何证明 CUDA kernel 是对的？ | `cuflash` / `tiny-llm` | independent reference、shape/dtype/mask matrix、sanitizer |
| PagedAttention 和分页 KV 有何区别？ | `tiny-llm` | 当前/历史路径、direct address formula、profile dispatch |
| continuous batching 真正 batch 了 GPU compute 吗？ | `paged-serving` | scheduler batch 与 runtime execution 的边界 |
| 如何定义 TTFT/TPOT/ITL？ | `paged-serving` | timestamp contract、token coverage、raw requests |
| 为什么选择 Rust + C++？ | 两个旗舰仓 | control/data plane、FFI ownership/error/lifetime |
| Triton 和 CUDA 如何取舍？ | `trifuse` / `cuflash` | 相同 contract、correctness、baseline、profile |
| 遇到性能回退怎么定位？ | 全部 | comparison key、A/B、timeline、kernel metrics、rollback |
| KV tiering 是否已生产化？ | `kvtier` | pinned upstream、实验范围和 prohibited claim |
| 多 GPU 做过吗？ | 理论材料 | 明确“未在当前项目实测”，解释 NCCL/TP 设计和验证计划 |
| 项目如何避免 cherry-pick 数据？ | `open-infra-ai` | raw artifacts、manifest、failed samples、stale/revoked |

---

## 10. PR 和集成顺序

跨仓改动不要放在一个巨大 PR：

1. contract/reference PR；
2. implementation PR；
3. integration PR；
4. benchmark/profiling PR；
5. evidence index PR。

对于 `tiny-llm` 与 `paged-serving`：

```text
tiny-llm additive ABI
  → paged-serving dual-compatible declaration
  → tiny-llm implementation/default switch
  → paged-serving remove compatibility path
  → end-to-end evidence
```

禁止：

- 两个仓同时删除旧 ABI，导致中间 commit 不可测试；
- benchmark PR 同时修改 production algorithm；
- 先发布性能数字，再补 correctness；
- 因 CI 环境无 GPU 而把 GPU test 改成静默成功。

---

## 11. 最终作品集结构

GitHub/简历展示顺序：

1. `tiny-llm + paged-serving`：一个完整系统项目；
2. `cuflash`：一个 CUDA 性能深挖项目；
3. `trifuse`：Triton/PyTorch 集成补充；
4. `cuda-foundations`：基础与 profiler 教学；
5. `kvtier`：前沿 KV 研究；
6. `open-infra-ai`：证据和导航入口。

最终不是证明“写过七个仓库”，而是证明：

> 能定义接口、数据布局、生命周期、正确性和性能证据；能把一个请求从 HTTP 追到 GPU，
> 能解释结果为什么可信、优化为什么有效或无效，以及系统在取消、慢客户端和资源不足时
> 如何保持正确。
