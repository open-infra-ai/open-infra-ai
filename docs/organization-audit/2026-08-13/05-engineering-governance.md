# 工程治理与发布基线

## 1. 目标

这是个人学习组织，不需要照搬大型企业流程。治理的目标是让每个仓库的绿灯、版本和性能声明可以被第三方理解和复现，而不是增加模板数量。

## 2. 最小仓库基线

每个活跃仓库至少应具备：

- README：定位、当前真实能力、明确非目标、验证状态。
- ROADMAP：只记录尚未完成的阶段和完成证据。
- CHANGELOG 或 release notes。
- LICENSE。
- 一条无需 GPU 的 CI lane。
- 一条 CUDA compile lane（CUDA 项目）。
- 对 GPU 数值状态的明确说明。
- 可从干净环境执行的开发检查命令。

`SECURITY.md`、CODEOWNERS、issue templates 可通过未来的组织 `.github` 仓库提供默认值，不必复制到每个学习仓。

## 3. CI 状态审计

| 仓库 | 当前状态 | 主要缺口 |
|---|---|---|
| `paged-infer` | Rust fmt/clippy/build/test/doc 较完整 | CPU 模型缺少数值 oracle；无真实 backend |
| `tiny-llm` | format + CUDA compile/test | hosted runner GPU case 大量 skip，核心绿灯含义过弱 |
| `cuflash-attn` | 工程最完整，含 compile matrix、CodeQL、package、GPU workflow | GPU runner 可用性和 benchmark artifact 未形成证据闭环 |
| `triton-fused-ops` | 无 GitHub Actions | CPU reference、ruff、mypy、package 都无人守护 |
| `cuda-kernel-academy` | pre-commit、docs、preset list | 没有 CUDA configure/build/test |

建议 check 命名显式表达证据，例如：

```text
host-unit
cuda-compile-sm80
gpu-numerical-a100
gpu-sanitizer
package-smoke
benchmark-nightly
```

避免笼统的 `build-test` 让读者误以为 GPU 数值已经执行。

## 4. 依赖治理

### Rust

- 继续提交 `Cargo.lock`。
- CI 使用 `--locked`。
- 需要可发布 library 时再区分 package lock 策略。

### Python/Triton

- 当前依赖只有宽下界，且没有可见 lockfile。
- 开发环境至少维护一套已验证组合：Python、Torch、Triton、CUDA。
- package metadata 可以保持范围依赖，但 CI/benchmark 必须使用锁定环境。
- 不把 Torch/Triton 小版本变化后的结果与旧 benchmark 混用。

### CMake/CUDA

- FetchContent 必须固定 tag 或 commit；关键性能依赖优先固定 commit。
- 允许使用预装依赖或离线 cache，避免 configure 必须访问 GitHub。
- 显式源文件列表优于核心 library 的 `GLOB_RECURSE`。
- 每个 release artifact 必须经过干净目录 package smoke。

## 5. 版本策略

当前源码版本高于最新本地 tag 的情况包括：

- `tiny-llm`：源码 `2.0.2`，最新 tag `v2.0.1`。
- `cuflash-attn`：源码 `0.5.0`，最新 tag `v0.3.0`。
- `triton-fused-ops`：源码 `2.0.0`，最新 tag `v1.0.0`。

这不一定是错误，但需要采用统一规则：

1. 主分支的下一个版本标为 `X.Y.Z-dev`，或把版本只在 release PR 中提升。
2. tag、release title、package version 和生成的 version header 来自单一版本源。
3. release workflow 验证 tag 与源码版本相符。
4. CHANGELOG 使用 `Unreleased` 区段。

## 6. 发布策略

### `tiny-llm`

当前 release 复制 demo、headers 和顶层 CMakeLists，但没有复制可链接 library、依赖或完整源文件。这种 artifact 不应标为 SDK。

二选一：

- 只发布可直接运行的 CLI，并把 artifact 定义为 executable bundle。
- 完成 `install(TARGETS)`、export config、依赖查找和 package smoke，发布真正的开发包。

在 CLI 尚不能生成文本前，不建议继续正式 release。

### `cuflash-attn`

保留现有安装/package smoke 方向。release notes 应附上：

- GPU numerical 最近一次通过环境。
- benchmark 是否刷新。
- 支持的 dtype/head_dim/causal 边界。
- 明确不支持 GQA/varlen/decode specialization。

### 维护仓库

Academy、Triton 和 Paged 可以按里程碑发布，不必高频版本化。教学修正应在 changelog 中明确标为 correctness/errata。

## 7. 文档真实性规则

- “已实现”表示代码存在且有相应层级测试。
- “已验证”必须同时写明环境、reference 和验证日期。
- “支持 GPU/CUDA 版本”应区分编译支持与真实执行支持。
- “高性能”必须链接可复现 benchmark。
- “生产级/工业级”需要远高于当前项目的兼容、可靠性和维护承诺，当前应避免。
- 历史性能快照必须带版本，并与当前数字明显区分。

## 8. PR 验收模板

每个实现 PR 至少回答：

```text
风险/任务编号：
修改的契约：
明确非目标：
新增或修改的外部 oracle：
CPU 检查：
CUDA 编译检查：
GPU 数值检查：
跳过项及原因：
文档和版本影响：
```

不要把“没有 GPU 所以测试 skip”填写为“测试通过”。

