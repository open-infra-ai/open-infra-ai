# 便宜模型执行任务规范

## 1. 使用方式

低成本实现模型适合执行边界清楚、测试先行、一次只改一类契约的任务。不要直接给它“修好推理引擎”或“优化 FlashAttention”这样的开放目标。

每个任务必须包含：

```text
任务编号
仓库和允许修改的文件
问题证据
目标契约
明确非目标
实现步骤
必须新增的测试
验收命令
禁止事项
交付格式
```

## 2. 通用任务模板

```markdown
# TASK-XXX：任务标题

## 范围
- 仓库：
- 允许修改：
- 禁止修改：

## 当前问题
给出具体生产者/消费者 shape、索引或状态变化，不只写“有 bug”。

## 目标契约
写出输入、输出、layout、dtype、错误和生命周期语义。

## 非目标
- 不新增无关功能。
- 不重构公共 API，除非设计明确要求。
- 不做性能优化。

## 实现顺序
1. 先增加会失败的测试。
2. 做最小实现修改。
3. 运行定向测试。
4. 运行仓库全量静态/编译测试。

## 验收标准
- [ ] ...

## 必跑命令
列出命令和预期；没有 GPU 时明确哪些不能声称通过。

## 交付
- 修改摘要。
- 文件清单。
- 测试输出摘要。
- 未验证项。
```

## 3. 任务拆分原则

### 好任务

- “为 attention decode 定义 token-major stride 并用非对称 fixture 验证。”
- “为 KV Cache key 增加 layer 维度并加入两层 incremental oracle。”
- “统一 RoPE cache 为 concat half-split，并更新 helper/reference/example。”

### 坏任务

- “重构 tiny-llm。”
- “让所有测试通过。”
- “提升性能到 PyTorch 的两倍。”
- “顺便清理代码风格。”

## 4. 实现约束

- 不为已知类型增加 `getattr`、多层 `isinstance` 或宽泛 `try/except`。
- 不通过吞掉错误、放宽 tolerance 或 skip 测试制造绿灯。
- 不复制 reference 的索引 helper 到被测实现。
- 不同时保留新旧两套 layout 作为长期兼容路径。
- 不在修正确性时引入 autotuning、模板元编程或新依赖。
- 不改变 benchmark 数字，除非实际重新运行并保存环境。

## 5. Review 提示词

实现完成后，应交给另一个模型做只读 review：

```text
请只读审查 TASK-XXX 的实现。重点检查：
1. 是否真正满足目标 layout/状态契约；
2. 测试是否来自独立 oracle，是否存在共模错误；
3. 非对称 shape、GQA、边界位置是否覆盖；
4. 是否忽略错误或通过 skip/放宽阈值掩盖失败；
5. 是否出现超出任务范围的重构。
按 blocker/high/medium 输出文件和行号，不修改代码。
```

## 6. 首批建议任务包

| 顺序 | 任务 | 依赖 | 规模建议 |
|---:|---|---|---|
| 1 | TLLM-001 layout 契约与 adapter | 无 | 1 个 PR |
| 2 | TLLM-002 GQA attention | 1 | 1 个 PR |
| 3 | TLLM-003 RoPE | 1 | 1 个 PR |
| 4 | TLLM-004 Qwen tensor contract | 1–3 | 1–2 个 PR |
| 5 | TLLM-L2 单层 oracle | 1–4 | 1 个 PR |
| 6 | PINF-001 layer-aware KV | 无 | 1 个 PR |
| 7 | TRIT-001 RoPE 契约 | 无 | 1 个 PR |
| 8 | CKA-001/002 教学致命错误 | 无，分别执行 | 各 1 个 PR |
| 9 | CUFA-101–103 benchmark 修正 | 有真实 GPU后 | 2 个 PR |

具体实现要求见 `designs/` 目录。每个任务完成后先 review，再进入下一个依赖任务。

