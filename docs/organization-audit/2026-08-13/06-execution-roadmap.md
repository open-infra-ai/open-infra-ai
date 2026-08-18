# 分阶段实施路线图

## 1. 排序原则

路线按依赖关系而不是仓库平均分配。每个阶段必须有可观察完成证据，不能只以代码提交作为完成标志。

## 2. 阶段 0：冻结与基线

目标：防止在错误契约上继续堆功能。

任务：

- 暂停新增 kernel、模型架构和性能宣传。
- 固定目标模型、prompt、token IDs 和模型文件 SHA-256。
- 固化本报告的布局、GQA、RoPE、KV 可见性契约。
- 建立任务编号和结果记录位置。

完成证据：

- 契约获得确认。
- fixture generator 不 import 五仓的被测实现。
- 能生成第一组 RoPE、GQA attention 和单层 fixture。

## 3. 阶段 1：修复 `tiny-llm` 数值主链

按顺序执行：

1. TLLM-001：统一 QKV/attention/KV Cache layout。
2. TLLM-002：实现 GQA 映射和 14→2 测试。
3. TLLM-003：实现 Q/K RoPE。
4. TLLM-004：核对 Qwen 权重、bias 和 tied output。
5. 统一错误传播，禁止忽略 KV append/advance 失败。

完成证据：

- L1 单算子差分通过。
- L2 单层所有中间值对齐。
- compute-sanitizer 对 GQA fixture 无越界。

## 4. 阶段 2：真实模型闭环

任务：

- 加入真正的 `--prompt` generation CLI。
- 实现目标模型 chat template 或先明确使用 raw prompt。
- 与 llama.cpp 比较 prefill/decode logits。
- temperature=0 下逐 token 对齐。
- 修复首个分歧，直到固定短序列完全一致或误差有明确量化解释。

完成证据：

- 一条命令从固定 GGUF 到文本输出。
- 保存模型 hash、prompt、token IDs、每步 top logits。
- 至少 3 个 prompt、每个 16 个以上生成 token 的对齐报告。

## 5. 阶段 3：修复维护仓的正确性

### `paged-infer`

- layer-aware KV Cache。
- incremental decode vs full recompute。
- 采样参数诚实语义。
- stateful stream decoder。

### `triton-fused-ops`

- 统一 half-split RoPE cache。
- 外部 golden fixture。
- 真实 kernel autotuner integration 或删除未兑现接口。
- CPU CI 与独立 GPU smoke。

### Academy

- 修复非法 attention launch。
- 修复多层 buffer alias。
- 重写 bank conflict/double buffering 教程。
- 最小 CUDA compile/test CI。

完成证据：对应 P0/P1 风险满足风险表关闭条件。

## 6. 阶段 4：重建 `cuflash-attn` 性能证据

任务：

- 先在真实 GPU 复跑全部数值、property 和 sanitizer。
- 重写或重新命名 naive baseline。
- 使用 CUDA Event/manual time 对齐 benchmark 实现与文档。
- 固定矩阵：dtype、causal、head_dim、sequence、batch、heads。
- 保存 JSON、环境 manifest 和 ncu/nsys 关键结果。

完成证据：

- 文档命令从干净 clone 可执行。
- 每个表格数字能追溯到 artifact 和 commit。
- 当前实现与 PyTorch SDPA 的比较语义完全相同。

## 7. 阶段 5：Serving 集成选择

只有 `tiny-llm` 的 L3/L4 成立后才开始。

候选方案：

- 同进程 C ABI/FFI：延迟低，但 Rust/C++ 生命周期和异常边界复杂。
- 独立 runtime 进程：接口清楚、隔离好，但有序列化和部署成本。
- 仅做设计展示，不实际集成：对于求职投入产出可能最优。

决策前先做窄 spike，只回答：

- batch descriptor 能否表达 prefill/decode。
- KV ownership 在哪一侧。
- 取消和 backend failure 如何传播。
- 是否真的能展示 continuous batching 的价值。

## 8. 阶段 6：上游贡献

完成至少一条真实模型证据链后，把学习成果转向 vLLM、SGLang、llama.cpp、Triton 或 FlashAttention 上游的小型贡献。对转行而言，一条经过 review 的上游 PR 通常比继续扩展自建框架更有信号。

## 9. 停止条件

遇到以下情况应停止扩展并重新评估：

- 目标模型频繁变化，导致 oracle 无法稳定。
- 没有可持续使用的 GPU 验证环境。
- 同一概念出现第三份实现但前两份没有外部验证。
- benchmark 只有汇总数字，没有原始样本和环境。
- 修复任务同时触及多个仓库且无法独立验收。

