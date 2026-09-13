# 跨仓证据 Artifact 契约与生命周期

> 目标：让任何人或 AI Agent 在不知道实验背景的情况下，仅凭 manifest、raw artifact 和
> exact commit，就能判断结果做了什么、是否有效、可以支持哪些声明，以及何时已经过期。

本文不统一各仓的业务结果 schema；它定义跨仓发布证据的**最小外层 manifest**。技术仓可以
继续保留自己的 JSON/JSONL/CSV，但正式结果包应额外提供一个符合
[`evidence-manifest.schema.json`](evidence-manifest.schema.json) 的 manifest。

## 1. 为什么还需要 manifest

仅有报告、截图或 summary JSON 时，后续 Agent 容易：

- 找不到产生结果的 exact commit；
- 把 dirty run 当作 release 结果；
- 忽略 GPU、模型、dtype、shape 或 workload 差异；
- 用 summary 代替已经丢失的 raw samples；
- 把 CPU/mock、GPU correctness、benchmark 和 serving 混为一谈；
- 在代码变化后继续引用已经不适用的旧数字；
- 只挑选最好的一次结果；
- 从失败或未收敛结果中生成正向简历 claim。

manifest 是“证据目录”，不是结果本身。没有 raw artifact 时，manifest 不能补造证据。

## 2. 结果包最小结构

推荐：

```text
results/<YYYY-MM-DD>-<hardware>-<subject>/
  manifest.json
  README.md
  raw/
    samples.jsonl
    requests.jsonl
    profiler.nsys-rep
    profiler.ncu-rep
    stdout.log
    stderr.log
  derived/
    summary.json
    table.csv
    plot.png
  checks/
    correctness.log
    sanitizer.log
```

规则：

- `raw/` 只保存实验直接输出，不手工编辑；
- `derived/` 必须能从 `raw/` 重建；
- `README.md` 负责面向人的方法、观察和限制；
- `manifest.json` 负责机器可读 provenance、状态、artifact hash 和 claim 边界；
- 大型 profiler 文件可放 release/object storage，但 manifest 必须保留稳定 URL 和 SHA-256；
- 模型权重、secret、真实联系人和私有主机信息不得提交。

## 3. Manifest 最小字段

| 字段 | 含义 |
|------|------|
| `schema_version` | 当前为 `1` |
| `evidence_id` | 仓库内稳定、不可复用的结果 ID |
| `status` | `draft/candidate/verified/published/stale/revoked/blocked/not_converged` |
| `evidence_level` | 与毕业标准中的 E0–E5 一致 |
| `kind` | correctness、sanitizer、benchmark、profiling、serving 或 integration |
| `repository` | URL、exact commit、branch、dirty state 和可选 diff artifact |
| `environment` | 时间、GPU、driver、CUDA 和 toolchain |
| `subject` | 被测实现、baseline 和比较键 |
| `workload` | dtype、quantization、shape/model/request distribution/seed |
| `protocol` | command、warmup、repetitions、计时边界和同步方式 |
| `correctness` | benchmark 前置正确性状态和 artifact |
| `artifacts` | path/URL、role、media type、SHA-256 |
| `metrics` | 数值、单位、聚合、样本数和收敛状态 |
| `limitations` | 不可外推内容 |
| `claims` | 该证据允许或禁止的自然语言声明 |
| `review` | reviewer、决定和时间 |

没有某项资源时不要伪造值：

- 没有 GPU 时，`environment.gpus` 使用空数组；
- 没有 benchmark 时，`metrics` 使用空数组；
- 没有 baseline 时省略 `subject.baseline`，并在 `limitations` 说明；
- 命令未运行时，不写入 `protocol.commands`；
- artifact 不存在时，不登记路径。

## 4. 状态机

```text
draft
  → candidate
  → verified
  → published
        ↓
      stale

任意状态 → revoked
资源不足 → blocked
统计失败 → not_converged
```

### `draft`

本地试跑或 schema 尚未完整。不能进入跨仓证据索引或简历。

### `candidate`

manifest、raw data 和报告齐全，但尚未独立 review。可以用于内部分析，不能称为正式结果。

### `verified`

满足：

- commit、环境和 workload 可定位；
- correctness 前置通过；
- raw artifact 存在且 hash 匹配；
- 统计和计时边界可审计；
- reviewer 没有 blocker。

### `published`

`verified` 结果已从技术仓 README 或组织
[`evidence-index.md`](evidence-index.md) 链接。发布不会扩大原始 scope。

### `stale`

结果仍是历史事实，但不能代表当前代码。不要删除；写明失效触发条件和 replacement。

### `revoked`

发现错误 baseline、数据损坏、错误统计、泄露、不可复现或 correctness 缺陷。所有引用应撤回。

### `blocked`

缺 GPU、模型、权限、依赖或批准。记录阻塞，不生成 metric。

### `not_converged`

实验真实完成，但预定义收敛标准失败。保留所有 raw samples，不能筛选后改为 `verified`。

## 5. 状态晋级门禁

### `draft → candidate`

- `manifest.json` 通过 schema；
- raw files 存在；
- 每个 artifact 有 SHA-256；
- report 能说明方法和限制；
- 无 secret/私有模型/个人信息。

### `candidate → verified`

- reviewer 重新计算至少一个 derived metric；
- exact commit 可获取；
- dirty run 有 diff artifact，否则拒绝；
- benchmark 的 `correctness.status == passed`；
- baseline 与 implementation 的 comparison key 相同；
- failure/OOM/429 没有从 raw data 中删除；
- CV 或项目约定的稳定性检查明确。

### `verified → published`

- claim 文本不超过 E 等级；
- README/图表链接回 manifest；
- 组织 evidence index 只写“能证明/不能证明”；
- 旧结果若被替代，标记 replacement，不覆盖历史目录。

## 6. 失效触发条件

以下变化发生时，维护者或 Agent 必须检查结果是否 `stale`：

1. 被测 production symbol 的行为变化；
2. public API/ABI、layout、mask、quantization 或 scheduler 语义变化；
3. correctness oracle 或容差被修正；
4. benchmark timing boundary 变化；
5. baseline 实现或配置变化；
6. model/tokenizer/chat template/revision 变化；
7. workload distribution、concurrency 或 request length 变化；
8. 发现 raw artifact 与 summary 不一致；
9. 依赖或 compiler 升级导致 dispatch 路径变化；
10. profiler 证明实际走的不是被声明路径。

仅 README 文案、拼写或不影响行为的 refactor 不自动让证据过期。是否失效由影响路径决定，
不是简单按日期决定。

## 7. 公平比较键

`subject.comparison_key` 用来阻止 Agent 将不同实验拼成 speedup。建议按项目构造，例如：

```text
cuflash:
  op + direction + dtype + B/H/S/D + causal + GQA + timing_boundary

tiny-llm:
  model_hash + quantization + context + decode_tokens + strategy + graph_mode

paged-serving:
  engine + model_hash + context/output distribution + arrival model +
  concurrency/rate + streaming + logprobs + host topology

kvtier:
  upstream_commit + model + KV dtype + tier/backend + W/E/R/P +
  cache capacity + concurrency
```

两个结果只有 comparison key 相同，且 implementation 与 baseline 做等价工作时，才允许计算
speedup。硬件不同的结果可以并列展示，但不能直接相除。

## 8. Artifact 规则

每个 `artifacts[]` 项：

- `role` 必须说明是 raw、summary、report、log、correctness、sanitizer、profiler 或 plot；
- `path` 只能是结果包内相对路径，或稳定的 HTTPS URL；
- `sha256` 对本地文件必填；
- `media_type` 使用明确 MIME；
- plot 不是 raw evidence；
- terminal copy/paste 不是长期 artifact；
- profiler 截图不能替代 `.ncu-rep`/`.nsys-rep` 或完整导出；
- summary 不得删除 failed/OOM/429 sample。

计算 hash：

```bash
sha256sum raw/samples.jsonl
```

若平台不是 GNU coreutils，可使用等价 SHA-256 工具，但报告中必须记录命令。

## 9. Claim 规则

`claims[]` 同时保存允许和禁止的句子：

```json
{
  "status": "allowed",
  "text": "在记录的 RTX 3060 Laptop 与该 shape 下观察到中位 kernel latency ...",
  "scope": "仅该 commit、GPU、dtype、shape 和 timing boundary"
}
```

```json
{
  "status": "prohibited",
  "text": "该实现普遍快于生产库",
  "reason": "只有单 GPU/单 shape，且没有跨硬件证据"
}
```

生成简历、README 或面试材料的 Agent 必须从 `allowed` claim 开始改写，并保留 scope；不能从
metric 最大值自行生成更强结论。

## 10. Agent 读取协议

未来 Agent 遇到性能或结果任务时按顺序执行：

1. 找到技术仓最新结果目录；
2. 读取 manifest，不先读宣传性摘要；
3. 验证 commit、dirty、status 和 evidence level；
4. 验证 raw artifact 存在与 hash；
5. 检查 correctness 前置；
6. 检查 comparison key；
7. 从 raw 重算关键 metric；
8. 检查 failed/OOM/429/not-converged；
9. 阅读 limitations 和 prohibited claims；
10. 最后再更新报告、README 或跨仓索引。

禁止：

- 自动把 `candidate` 改为 `verified`；
- 自动把 `not_converged` 改成 `passed`；
- 因为新结果更差而删除旧或新 raw data；
- 从文件名猜测 GPU/model/commit；
- 用本次机器的环境补填旧实验；
- 修改 raw data 让 validator 通过；
- 让实现 Agent 成为唯一 reviewer。

## 11. Reviewer 检查表

- [ ] manifest 通过 schema；
- [ ] commit 可获取，dirty 状态真实；
- [ ] raw artifact hash 匹配；
- [ ] correctness 与 benchmark 是同一实现；
- [ ] baseline 做等价工作；
- [ ] timing boundary 清楚；
- [ ] warmup、repetitions、seed 可复现；
- [ ] 失败样本保留；
- [ ] metric 可从 raw 重算；
- [ ] 收敛规则在看结果前定义；
- [ ] claim 保留 GPU/model/workload 限定；
- [ ] 没有 secret、个人信息或受限模型；
- [ ] 旧 published 证据的 stale/replacement 状态已处理。

## 12. 采用顺序

无需一次迁移全部历史结果：

1. 所有**新正式结果**先采用；
2. 下一次引用某个历史结果时补 manifest；
3. 旗舰 `tiny-llm + paged-serving` 优先；
4. 然后补 `cuflash` profiler/benchmark；
5. 教学快照和历史 archive 保持原样，除非重新发布；
6. 没有 raw data 的旧数字只保留为历史叙述，不升级为 verified。

该顺序避免为了整理历史而推迟新的 correctness、实现和实验。
