# 风险登记表

## 1. 使用方法

状态初始均为 `Open`。处理风险时，应在对应仓库的 PR 或任务记录中引用风险编号；只有满足“关闭条件”后才能改为 `Closed`。降低文档措辞只能缓解宣传风险，不能关闭正确性风险。

## 2. P0 风险

| ID | 仓库 | 问题 | 影响 | 证据强度 | 关闭条件 |
|---|---|---|---|---|---|
| TLLM-001 | tiny-llm | QKV token-major 输出被 head-major attention 解读 | prefill/decode 数值错误 | 静态确认 | 显式 layout 契约；prefill/decode 与外部 oracle 覆盖 `S>1,H>1` |
| TLLM-002 | tiny-llm | GQA 未映射 query head 到 KV head | 越界或错误 KV | 静态确认 | `kv_h = q_h / group_size` 等明确映射；14→2 fixture 通过 sanitizer 和差分 |
| TLLM-003 | tiny-llm | RoPE 未进入运行时计算 | 所有位置相关 logits 错误 | 静态确认 | Q/K RoPE 单测、单层差分、真实模型 logits 对齐 |
| TLLM-004 | tiny-llm | 目标架构 bias/tied output 权重契约不完整 | 模型加载成功但输出错误，或拒绝合法 GGUF | 高置信 | 以目标 GGUF tensor 清单为准实现并测试可选/必需规则 |
| PINF-001 | paged-infer | CPU KV Cache key 缺少 layer | 跨层 K/V 相互覆盖 | 静态确认 | cache 地址包含 layer；多层 incremental vs full recompute 差分 |
| TRIT-001 | triton-fused-ops | RoPE helper/API/example 排列不一致 | 输出与主流模型 RoPE 不一致 | 静态确认 | 选择单一 convention；golden vector 与外部实现一致；示例/API/reference 同步 |
| CKA-001 | cuda-kernel-academy | FlashAttention 以 2048 threads/block 启动 | kernel launch 非法 | 静态确认 | 合法线程布局；检查 launch error；与 CPU/PyTorch 数值差分 |
| CKA-002 | cuda-kernel-academy | 多层 forward 中间 buffer 输入输出别名 | 三层以上网络结果错误 | 静态确认 | ping-pong buffer；4+ 层 CPU oracle 和 sanitizer 测试 |

## 3. P1 风险

| ID | 仓库 | 问题 | 影响 | 关闭条件 |
|---|---|---|---|---|
| TLLM-101 | tiny-llm | demo 不支持 ROADMAP 中的 `--prompt` | 无法交付端到端模型证据 | CLI 完成固定 prompt 生成并与 llama.cpp 对齐 |
| TLLM-102 | tiny-llm | 多处 `Result` 被忽略，且与异常混用 | 失败后继续计算或错误边界不清 | 明确错误策略；所有关键 append/advance/kernel 错误传播测试 |
| TLLM-103 | tiny-llm | hosted CI 的 GPU 数值测试大量 skip | 绿灯不能证明核心计算 | 独立 GPU required check；无 GPU 时明确 neutral/未验证，而非核心 check 通过 |
| TLLM-104 | tiny-llm | release 包含 binary/header/CMakeLists，却缺少可链接库/源码/install config | artifact 无法按承诺消费 | package smoke 在干净目录完成 find_package、编译、运行 |
| PINF-101 | paged-infer | `temperature/top_p` 被验证但不生效 | API 行为与声明不符 | 实现采样或在 API 中拒绝非 greedy 配置 |
| PINF-102 | paged-infer | 单 token decode 直接作为 SSE 文本片段 | byte/BPE 流式文本可能损坏 | 增量 decoder 状态测试覆盖多字节与合并 token |
| PINF-103 | paged-infer | Chat 仅拼接角色字符串 | “OpenAI compatible”语义过宽 | 改称 API-shaped，或实现可配置 chat template 和兼容性测试 |
| CUFA-101 | cuflash-attn | benchmark 文档与实现计时方法不一致 | 性能结果不可复核 | 统一代码和文档；输出原始 JSON、CUDA Event 样本 |
| CUFA-102 | cuflash-attn | 文档使用不存在的 CMake 参数和路径 | 用户无法复现 | 从干净 clone 执行文档命令的自动 smoke test |
| CUFA-103 | cuflash-attn | naive baseline 声称物化 N²，实际为逐行 shared scores | 基线定义和带宽指标失真 | 更名/重写 baseline 与流量模型，增加范围限制 |
| CUFA-104 | cuflash-attn | GPU workflow 依赖可能不存在的自托管 runner | 数值验证可能长期未执行 | 定期成功记录和 artifact；README 显示最近验证环境/date |
| CUFA-105 | cuflash-attn | 历史性能表缺少与当前提交对应的原始 artifact | 旧数字被误当成当前实现证据 | 历史结果明确归档；当前表由原始 JSON 和环境 manifest 生成 |
| TRIT-101 | triton-fused-ops | autotuner 只对 dummy wrapper 有效 | 对外宣称的调优基础设施不可用于真实算子 | 至少一个真实 kernel 配置可变且集成测试通过 |
| TRIT-102 | triton-fused-ops | benchmark reference 默认 CPU backend | CUDA tensor 被搬到 CPU，类型/计时/比较失败 | benchmark 显式 CUDA reference，加入最小 GPU smoke |
| TRIT-103 | triton-fused-ops | 无 CI | 格式、类型、CPU reference、打包无人守护 | 添加 CPU CI；GPU 验证单独 required/periodic |
| TRIT-104 | triton-fused-ops | property test 可随机到不适合 PR 的巨大矩阵 | GPU OOM、编译变体过多或 CI 长时间不稳定 | 小 shape property 与真实 shape nightly 分离 |
| TRIT-105 | triton-fused-ops | 显式 `num_heads` 时不验证与 hidden/head_dim 一致 | 部分输出未写或逻辑越界 | 验证 `num_heads * head_dim == hidden_dim` 并覆盖失败路径 |
| TRIT-106 | triton-fused-ops | empty fast path 位于 positive-dimension 校验之后 | 声明支持但分支不可达 | 明确支持/拒绝 empty 的单一契约并测试 |
| CKA-101 | cuda-kernel-academy | CI 只列 presets，不配置/编译 CUDA | 大量代码不可由 CI 证明可编译 | 最小 sm_80 compile matrix 和 host/package smoke |
| CKA-102 | cuda-kernel-academy | TensorCraft 没有 attention 测试 | 致命 launch 配置未被发现 | attention launch+数值+边界测试进入测试目标 |
| CKA-103 | cuda-kernel-academy | bank conflict 与 double buffering 教学叙述失真 | 学习者形成错误性能模型 | 用 warp 访问映射和 profiler 证据重写说明 |
| CKA-104 | cuda-kernel-academy | software pipeline 索引与线程映射高风险且缺少覆盖 | 可能越界/重复计算 | 小矩阵 oracle、compute-sanitizer、映射设计说明 |
| CKA-105 | cuda-kernel-academy | TopK 对 `n>1024` 只处理前 1024 项 | 静默返回错误 top-k | 先明确拒绝大 n；多 block TopK 作为独立扩展 |

## 4. P2 风险

| ID | 范围 | 问题 | 建议 |
|---|---|---|---|
| ORG-201 | 全组织 | 没有统一 layout/RoPE/KV 契约 | 采用本报告契约并以 fixture 固化 |
| ORG-202 | 全组织 | GPU 测试状态不统一 | 统一区分 compile、CPU、GPU numerical、sanitizer、benchmark |
| ORG-203 | 全组织 | benchmark 缺少共同结果 schema | 使用机器可读结果与环境 manifest |
| ORG-204 | 全组织 | 版本与最新 tag 漂移 | 明确 unreleased 版本策略，release 从单一版本源生成 |
| ORG-205 | 全组织 | CONTRIBUTING/SECURITY/CODEOWNERS 分布不一致 | 以后用组织 `.github` 仓库提供默认模板；当前非阻塞 |
| PINF-201 | paged-infer | `GPUExecutorTrait` 命名绑定 GPU，默认却是 CPU | 改为中性的 backend/executor 语义 |
| PINF-202 | paged-infer | crate/CLI 仍称 mock，README 称 CPU reference | 统一当前 backend 说明 |
| TLLM-201 | tiny-llm | `file(GLOB_RECURSE)` 和 FetchContent 网络依赖降低可重复性 | 显式源列表；允许系统依赖或依赖锁定缓存 |
| CUFA-201 | cuflash-attn | 当前 API 只支持等头 dense attention | 文档明确；若服务 runtime 再单独设计 GQA/decode |
| CKA-201 | cuda-kernel-academy | 根安装只覆盖 common header/docs，package 语义不完整 | 教学仓可取消 package 暗示，或完成组件化安装 |

## 5. 依赖顺序

```text
TLLM-001 layout
  -> TLLM-002 GQA
  -> TLLM-003 RoPE
  -> TLLM-004 model weights
  -> L2 layer oracle
  -> L3 logits oracle
  -> TLLM-101 CLI/token oracle
  -> paged-infer backend integration
```

`cuflash-attn` 和 Academy 的修复可与上述主链并行，但任何性能宣传都应等待对应正确性检查完成。
