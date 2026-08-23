# 仓库职责与内容边界

本文定义 `open-infra-ai` 组织、个人求职仓与上游社区之间的内容归属。目标是让面试官
打开任意仓库时都能快速判断：它解决什么问题、证据在哪里、哪些内容不属于这里。

## 1. 组织 meta 仓：`open-infra-ai/open-infra-ai`

保留以下公开、稳定、跨项目内容：

- 组织 landing 页、五个技术仓导航与状态注册表；
- `LEARNING_PATH.md` 组织级学习顺序；
- `docs/cross-repo-contracts.md` 跨仓语义契约；
- 可复现证据索引与仓库治理记录；
- 明确标注只读的历史计划、历史面试材料与组织审计快照。

不再新增活跃简历、投递状态、公司清单、每日打卡或上游评论草稿。历史档案不因边界调整
而改写；它们记录的是当时事实。

## 2. 五个技术仓：只承载可运行的技术作品

| 仓库 | 唯一主责 | 不承载 |
|------|----------|--------|
| `cuda-foundations` | CUDA 基础、SGEMM 阶梯、通用推理组件教学 | 完整推理引擎、求职计划 |
| `triton-fused-ops` | Triton 算子与 `torch.library` 集成对照 | CUDA 专项实现的重复副本 |
| `cuflash-attn` | FlashAttention/FlashDecoding CUDA 专项深挖 | `tiny-llm` 的 generate 路径 |
| `tiny-llm` | 真实权重加载、量化、decode、KV 与端到端推理加速 | HTTP 调度控制面 |
| `paged-infer` | Paged KV、continuous batching、调度、HTTP/SSE 与 serving 评测 | 重复实现模型算子与权重加载 |

`tiny-llm` 与 `paged-infer` 只通过受测试的 C ABI 集成；`cuflash-attn` 不接入
`tiny-llm` generate 路径。技术仓名称已经被简历和证据引用，保持冻结，不做大规模重命名。

## 3. 个人执行仓：`holtwood/ai-infra-interview-prep`

集中维护所有仍会快速变化的个人执行材料：

- 12 周学习与求职路线、能力矩阵、每周复盘；
- 脱敏简历草稿、目标公司清单、投递模板；
- 面试问题矩阵与讲述练习；
- 上游 issue 筛选脚本、复现器和评论/PR 草稿。

真实联系方式、联系人、内推关系、薪资和投递状态只放 Git 忽略的 `.local` 文件，不提交到
公开仓库。

## 4. 上游社区贡献：以上游仓库为最终证据

复现脚本和调查草稿可以先在个人执行仓孵化；一旦形成评论或 PR，最终可验证证据应指向
上游 issue、PR、commit 和 CI。不要为每个 KV Cache、调度或 kernel 想法新建一个玩具仓：
先在现有技术仓使用 `experiments/`、benchmark 或测试夹具形成闭环；只有受众、依赖、发布
周期和维护边界都明显独立时才新建仓库。

## 5. 面试叙事边界

- 推理加速主线：`tiny-llm`；
- CUDA 专项深度：`cuflash-attn`；
- Serving/调度扩展：`paged-infer`；
- 基础与跨语言对照：`cuda-foundations`、`triton-fused-ops`。

五仓共同组成能力链，但简历不要把五仓平铺成五个同等重要项目。主项目讲深，另外两项作为
针对岗位的证据补充，其余只在技能或 GitHub 导航中出现。
