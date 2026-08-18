# 审计范围与方法

## 1. 审计目标

本次审计服务于三个目标：

1. 判断五个仓库的学习定位和架构边界是否合理。
2. 找出会阻断真实 AI Inference 正确性、可复现性或作品可信度的问题。
3. 把问题转化为可以交给低成本实现模型执行的设计和验收规范。

本次审计不是生产安全认证，也不是性能认证。当前环境无法访问 GPU，因此所有 GPU 数值和性能结论都必须保留为待验证状态。

## 2. 包含范围

- 五个仓库的受版本控制源码与测试。
- README、ROADMAP、CHANGELOG 和主要设计文档。
- CMake、Cargo、Python packaging、Makefile 等构建入口。
- GitHub Actions、发布脚本、package/install 配置。
- 核心推理数据流：模型加载、QKV、RoPE、attention、KV Cache、采样、调度和 HTTP 接口。
- 跨仓职责、复用方向和端到端验证缺口。

## 3. 不包含范围

- GitHub 组织后台设置、branch protection、secret、真实 Actions 历史和自托管 runner 状态。
- 未出现在本地工作区中的私有仓库、issue、PR 和外部模型文件。
- GPU 数值执行、compute-sanitizer、Nsight profiling 和真实性能复测。
- 生产部署安全、容量规划和在线 SLO。
- 对第三方项目当前版本的互联网比较。

## 4. 审计方法

### 4.1 静态检查

- 从公开入口向下跟踪主要调用链，而不是只统计文件或搜索 TODO。
- 对每个算子同时检查生产者布局、消费者布局、缓存布局和测试 reference。
- 将“有意简化”与“实现违反自身契约”分开。
- 将 README 声明与代码、测试、CI 和 release artifact 互相核对。

### 4.2 动态检查

所有构建产物均写入 `/tmp`，未写入五个仓库：

| 仓库 | 检查 | 结果 |
|---|---|---|
| `paged-infer` | `cargo test --locked`，含 unit/integration/server/doc | 137 项通过 |
| `tiny-llm` | Release、sm_80、关闭测试的完整编译 | `libtiny_llm.a` 与 demo 通过 |
| `cuflash-attn` | Release、sm_80、关闭测试/示例/benchmark 的静态库编译 | 通过；存在大量 NVCC/GCC line-directive warning |
| `triton-fused-ops` | `python -m compileall` | 通过 |
| `cuda-kernel-academy` | `cmake --list-presets` | 通过 |

GPU 状态检查返回 `GPU access blocked by the operating system`。因此本报告没有把任何 GPU case 标记为已执行。

### 4.3 风险等级

| 等级 | 定义 |
|---|---|
| P0 | 会产生错误结果、越界、非法 kernel launch，或直接阻断真实推理主链 |
| P1 | 测试/benchmark/接口/发布证据不可信，可能导致错误结论或无法交付 |
| P2 | 架构债、维护成本或文档漂移，不立即破坏核心正确性 |
| P3 | 品质与一致性改进，可在主要路线完成后处理 |

“P0”不等于线上事故严重度；这些仓库多数明确是学习项目。它表示在继续构建下一层能力之前必须先解决。

## 5. 证据标准

每条关键结论至少满足以下一种条件：

- 生产者和消费者的索引公式直接矛盾。
- 声明的配置与 kernel launch 超过 CUDA 明确限制。
- 状态或资源键缺少实现所需的维度。
- 公共 API、示例、reference 和 helper 之间无法同时成立。
- README/ROADMAP 与实际 CLI、CI 或构建目标直接矛盾。

对于无法在当前环境运行的 GPU 问题，本报告使用“静态确认”“高置信风险”或“待 GPU 验证”区分证据强度。

## 6. 审计限制

- 没有加载真实 Qwen GGUF，无法验证张量名集合和所有 architecture-specific 细节。
- 没有 GPU，无法判断 kernel 数值误差、资源限制、动态 shared memory 和真实吞吐。
- 没有查询远端 GitHub，因此本地 CI 文件存在不代表 workflow 当前可用。
- benchmark 文档中的历史数字没有原始 JSON、环境清单或 profiler artifact，不能复核来源。

## 7. 重审触发条件

出现以下任一事件应做增量重审：

- `tiny-llm` 完成 layout/GQA/RoPE 修改。
- 引入新的模型架构或 GGUF tensor 类型。
- `cuflash-attn` 刷新 benchmark 或增加 decode/GQA。
- `paged-infer` 接入真实 backend。
- Academy 修改 FlashAttention、GEMM pipeline 或 CI。
- CUDA、PyTorch、Triton 的最低版本发生变化。

