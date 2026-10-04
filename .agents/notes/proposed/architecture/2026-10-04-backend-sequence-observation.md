# Agent Note: 终态回收的真实后端登记观察

Status: proposed

## Problem

逻辑 KV 利用率归零、同实例继续成功，不能单独证明 tiny-llm 的分页序列登记被移除。
[真实释放负对照](https://github.com/open-infra-ai/paged-serving/blob/4b095567f70ae5b835826e5897a9a42c7ac53b8e/.agents/notes/implemented/testing/2026-10-04-real-backend-terminal-reuse.md)
故意不通知后端释放时，连续 KV 的四槽位耗尽，但分页 KV 仍能接下一批请求。
缺少独立于 Rust 调度统计的观察，分页生命周期测试会漏掉这一类错误。

## Proposal

任务 `PSRV-P1-002/OBS` 只补一个同步、只读的原生登记数量查询，以及使用它的真实
生命周期测试。设计评审状态为 `pending`；本提案不是批准记录，不授权修改生产实现。
查询不进入正常 step 热路径，不添加 HTTP `/metrics` 字段、不改变调度或释放行为。
持续 GPU CI、真实网络断连和显存字节回收属于独立后续任务。

### G0：调研基线与事实

调研基线在 `fix/interview-credibility-2026-10-04`，开始调研时均为 clean：

| 仓库 | 40 位基线 commit | 锚点 |
|---|---|---|
| tiny-llm | `b9cbdf922dcfaf0dfc8c704bd285b4b04e757f51` | `include/tiny_llm/ffi.h`、`src/ffi.cpp`、`tests/test_ffi.cpp` |
| paged-serving | `4b095567f70ae5b835826e5897a9a42c7ac53b8e` | `src/tiny_llm_ffi.rs`、`src/tiny_llm_executor.rs`、`tests/tiny_llm_backend.rs` |
| meta | `5592c0228a55b4468813efefff47a870c8230eaa` | `docs/cross-repo-contracts.md` |

`TinyLlmHandleImpl::sequences` 是两种策略的 C++ 登记表；分页 allocate 增加登记，free
移除登记。Rust `TinyLlmExecutor::allocated` 是另一层状态，不能代替原生表的观察。
引擎把 executor 移入 `Box<dyn GPUExecutorTrait>`；测试需保留受控的查询入口，不能
从 Rust 表推算数量或把原始 handle 暴露给调用方。

本地已有 RTX 3060 Laptop 6 GB、固定 Qwen 0.5B GGUF 和 `/usr/bin/compute-sanitizer`；
模型/静态库来源与 SHA 见上面的执行证据。本提案不采集新 GPU 结果；工具存在不代表
Sanitizer 已通过。实现前重验模型/库/两仓 commit，并如实记录 dirty state。

### G1/G2：唯一查询接口

拟新增签名，评审批准后才写入 live 契约与双源代码：

```c
int tinyllm_registered_sequence_count(const TinyLlmHandle *handle, uint64_t *out_count);
```

- C 头文件引用 `<stdint.h>`；Rust 声明对应 `*const TinyLlmHandle`、`*mut u64`，
  返回 `c_int`。输出是调用方提供的一个 64 位标量，不是数组或设备指针。
- 成功返回 0，只写 `*out_count = sequences.size()`；handle 或 out_count 为空返回
  当前 `TLLM_ERR`（-1），失败不改输出。非空 handle 必须仍存活。
- 只观察已登记序列数，不是已用物理块数、连续槽位数、token 数或 cudaMalloc 字节数。
- `TinyLlmConfig` 的 ABI v2、36 字节布局和已有函数完全不变。新增符号保留旧消费者
  可用，但新消费者不能链接旧静态库；缺符号必须 link failure，不能退化成返回零。
- Rust 拟增加 `TinyLlmExecutor::registered_sequence_count(&self) -> Result<u64, EngineError>`，
  直接调用 C 查询，非零返回 `BackendError`；不读取或对齐 Rust shadow 表来制造成功。

### G3/G4：所有权与时序

handle 仍由唯一 `TinyLlmExecutor` 所有；查询不分配、不释放、不 launch kernel，也不做
`cudaDeviceSynchronize`。同 handle 的 load 后操作、step、free_sequence、查询、销毁
必须串行，不宣称线程安全。查询只发生在同步 step/终态处理返回之后。

测试专用 wrapper 用 `Arc<Mutex<TinyLlmExecutor>>`，转发 execute、capabilities 和
sequences_finished；测试保留一个 Arc 用于读取真实原生表。每次访问取得同一锁，查询
结束释放锁，不在持锁期间调用 engine；最后一个 Arc 销毁时释放 handle。没有新 unsafe
共享 handle 或生产锁。故障 wrapper 使用同一观察通道，但故意不转发释放通知。

### G5：错误与受限能力

查询失败直接使测试失败；终态之后非零数量也失败，不能只打印 warning。现有
sequences_finished 的 warn-only 错误出口不在本任务中修改，查询仅让残留可被验收发现。
缺模型/库/GPU 沿用已落地的显式失败门禁。缺失登记与已释放显存是两种结论，数量归零
不能宣称固定 KV pool、scratch 或模型显存已释放。

### G6：先定义能失败的验收

策略 1/2 均固定 max_seqs=4、decode_reserve=512、串行执行：

| 观察点 | 正常实现的原生数量 | 拦截释放对照 |
|---|---|---|
| 新建后端，无请求 | 0 | 0 |
| 四请求完成 prefill / 一次 decode | 4 / 4 | 4 / 4 |
| 普通完成或 prefill/decode 后取消，终态处理返回 | 0 | 取消后 4 |
| 越界失败的终态处理返回 | 0 | 按实际成功分配的登记数报告，不猜测 |
| 同实例满容量探针完成 | 0 | 分页仍成功但有登记残留；连续槽位耗尽 |

新增 C++ 测试覆盖两策略的 allocate/free 数量、重复分配、分页重复 free、无效参数及
失败输出不变；保存现有连续 free 的错误语义，不把两策略统一改成幂等。
Rust 真实 engine 测试覆盖已有完成/取消/失败与同实例复用。最关键的负对照必须在
分页策略下实际观测到取消后 4 而不是零，否则观察测试本身不可信。

先构建 `tiny_llm`、`tiny_llm_tests`，执行原生观察测试与现有 FFI 差分；再执行 Rust
两策略后端矩阵与完整 feature suite，使用 `--include-ignored --test-threads=1`。
原生测试执行设置 `TLLM_GGUF_TEST_MODEL`，核验实际用例数与 skipped=0。
对观察测试运行 Compute Sanitizer memcheck，配置非零 error-exitcode；无法启动或工具
限制记录为未通过，不用普通 GPU test 代替。最后执行默认 CPU suite、fmt、clippy 和
两技术仓及 meta 的 notes gate。

### G7：不产生性能声明

这批是生命周期 correctness，不计 speedup、TTFT/TPOT 或显存回收收益。查询只用于
串行测试；若以后接入定时遥测，采样扰动和时钟对齐需另外设计并做 A/B。新证据记录
两仓 exact commit/dirty state、改动文件 SHA、模型/静态库 SHA、环境与原始退出码；
不覆盖已有 evidence 或 Serving 历史包。

### G8：文件归属、提交与回滚

1. 先更新 meta `docs/cross-repo-contracts.md` 的 live 语义，再同批修改 tiny-llm
   `include/tiny_llm/ffi.h`、`src/ffi.cpp`、`tests/test_ffi.cpp`、CHANGELOG 与本仓笔记。
2. paged-serving 同批修改 `src/tiny_llm_ffi.rs`、`src/tiny_llm_executor.rs`、
   `tests/tiny_llm_backend.rs`、CHANGELOG 与本仓笔记。不改 executor trait、build.rs、
   Cargo.lock、server 指标或 workflow；计划需要扩界时回到设计评审。
3. 各仓精确提交，记录双 commit 与库 SHA；只推授权整改分支，不自动 merge PR #23。
   默认分支集成应先提供兼容的新 tiny 库，再切换 Rust 消费者，不能留下不可链接状态。
4. 回滚 Rust 查询消费者和新测试后，旧接口仍可继续工作；原生新增查询可保留。
   如必须移除符号，先确认没有消费者。运行时调度与释放策略没有替换或 fallback。

## Alternatives considered

继续使用满容量复用探针不增加 API，对连续槽位耗尽已有有效反例；但分页登记不受
max_batch_size 限制，现有负对照已证明它不足。保留复用测试，增加原生数量作为另一条
观察，而不是提高请求量后宣称覆盖。

只公开 Rust allocated 数量可以避免双仓改动，且易于单测；但 Rust 在调用 native free
之前删除自己的登记，native free 失败时也可能显示零，不能证明真正后端状态。

用 debug 日志解析分配/释放轨迹不扩 ABI，还能看到序列 ID；但日志开关、顺序和信息
完整性不是可稳定查询的状态，不能独立断言残留。只读标量查询是当前最小可验证接口；
需要逐 ID 定位或显存字节观测时重新评审，不在这批引入快照数组。

## Acceptance criteria

用户明确批准此受限设计，且 G0-G8 评审结论及接受的限制留在仓库，才开始生产改动。
两策略正向矩阵与释放负对照均实际执行；查询来自原生表，测试不能 skip、不能以
Rust 表或利用率代替。原生无效参数、Sanitizer、feature link/完整 suite 与默认回归
都有可追溯输出。到此只关闭 `PSRV-P1-002/OBS`；持续 GPU lane、HTTP 网络回收、
独立审阅和默认分支集成仍各自未完成。

## Risks

新增符号要求绑定新的静态库，不能拿旧 library SHA 充当新实现证据。测试 wrapper
共享所有权会推迟 handle 销毁，所有观察须发生在真实终态后、最终 Drop 前。数量可以
发现净残留，不能识别等量错误替换、每个 ID 的身份或实际 pool 字节占用。将来若测试
显示需要这些观察，再扩设计；不把当前轻量查询包装成通用后端监控系统。

## Related notes

[ABI 双源治理](../../implemented/architecture/2026-08-21-abi-contract-dual-source.md)
部分重叠，仍是代码/语义契约的权威治理，不被本提案替代；
[证据与个人能力分离](../../implemented/process/2026-10-04-live-evidence-and-interview-status.md)
仍约束声明，本篇仅描述待批准的观察接口。其余 meta 笔记与具体原生查询无关。
