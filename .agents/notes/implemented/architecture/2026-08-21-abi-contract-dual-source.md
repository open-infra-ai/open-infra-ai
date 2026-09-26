# Agent Note: ABI 契约代码双源，语义契约单源 live 文档

Status: implemented

## Problem

tiny-llm（C++ 数据面）与 paged-serving（Rust 控制面）经同进程 C ABI 静态链接
协作。两种语言各持一份结构体与函数签名声明，缺少「谁改谁」的约定就会静默
漂移——布局错位在 ABI 边界上表现为内存损坏而非编译错误。

## Decision

- ABI 契约**代码双源**：`tiny-llm/include/tiny_llm/ffi.h` ⇄
  `paged-serving/src/tiny_llm_ffi.rs`。`TinyLlmConfig` 为 9 个 int 的 repr(C)
  布局，paged-serving 侧用 `size_of == 9*4` 布局守卫测试锁定一致性。
- 语义契约（12 条：维度命名 / 布局 / GQA / RoPE / KV 事务 / 采样顺序 / 流式文本等）
  单源在 `docs/cross-repo-contracts.md`（live 版），派生自只读审计快照。
- 布局、RoPE convention、ABI 变化是 **breaking change**：先改契约文档，
  双仓同批改动，两仓 CHANGELOG 各记一条。
- 差分验证锚点：`tiny-llm/tests/test_ffi.cpp`（策略 1 vs 策略 2 逐 token）、
  `paged-serving/tests/tiny_llm_backend.rs`、`tiny_llm_text_e2e.rs`。

## Alternatives considered

- **bindgen / cbindgen 单源生成** — 彻底消除手工双源最强；但为一个 ~9 字段的
  ABI 引入代码生成工具链，反而加大审计面；`size_of` 守卫测试已能廉价捕获漂移。
- **跨进程 IPC 隔离后端** — 故障隔离更强；但本作品集练的就是同进程窄接口
  调度，序列化开销会掩盖要展示的 serving 概念。

## Consequences

- **收益**：ABI 边界窄、可差分测试；语义演进有 live 文档落点，审计快照保持只读。
- **代价**：双源靠守卫测试+纪律维持，不是编译期保证；新增 ABI 字段必须同批
  动两仓，遗漏不会在单一仓内立刻报警。

## Verification

`paged-serving` 布局守卫测试（`size_of::<TinyLlmConfig>() == 36`）；契约条目见
`docs/cross-repo-contracts.md` §10.1；2026-08-23 changelog 记录了 logprobs
缓冲区契约澄清（收紧校验，不改签名布局）。
