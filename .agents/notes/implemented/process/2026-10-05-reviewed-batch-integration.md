# Agent Note: 可信度整改按审阅后的固定提交合入默认分支

Status: implemented

## Problem

整改实现已经推送，但默认分支、较早的取消 PR 与个人任务卡各指向不同提交。
只查看某个旧 PR 的绿色检查，无法确认整批代码已经进入默认分支。

## Decision

用户明确授权独立子代理审阅与直接合并。每仓使用独立 PR，合并前固定 head、确认
全部适用 CI 成功并关闭已发现的问题；合并请求绑定该 head，不绕过仓库保护规则。
使用 merge commit 保留实验和笔记引用的原提交，合并后 fetch 并验证已审 head
确实是远端默认分支的祖先。共享归档工具四个源码副本在六仓必须一致。

此授权只覆盖当前整改批次，不批准 OBS 的具体新增 C ABI、GPU HTTP 验收设计、
runner 注册、云 GPU 采购或上游公开写操作。旧 paged PR #23 和准备仓 PR #9 不被
替换或自动关闭；旧 head 不作为整批整改的交付凭据。个人交付物也不自动勾选。

[证据与能力分离](2026-10-04-live-evidence-and-interview-status.md)部分重叠：它继续
定义结论范围，本笔记定义默认分支集成与复核；[归档预检](../bug-fix/2026-10-04-archive-preflight.md)
与[完整交付命令](2026-10-04-delivered-note-commands.md)分别持有工具行为。
既有历史冻结、身份和 ABI 双源决定保持原样，不被本笔记吸收或归档。

## Review scope

四个独立只读代理分别审阅 Serving 生命周期、loadgen/结果证据、DPA/Triton 基准和
跨仓治理/任务交接；执行代理与审阅代理分离，主代理再检查合并目标和最终 CI。
这不是人工行业专家背书，也不把 CI 成功当作独立审阅。

发现的六项问题均经过修复复审：文本 EOF 丢终态竞态；DPA 正确性标志/差值矛盾；
缺模板看板命令；损坏 manifest 覆盖封印；归档迁移导致出链改变；补封印路径漏验。
CUDA 基础仓旧测试对 AGENTS.md 的全面禁令也与现行约定对齐，仍拒绝旧框架文件，
并新增已交付 skill 与 notes gate 的文件检查。

## Integration state

| 仓库 | 整改 PR | 当前集成状态 |
|---|---|---|
| tiny-llm | [#18](https://github.com/open-infra-ai/tiny-llm/pull/18) | 已合并，merge e069d07 |
| paged-serving | [#24](https://github.com/open-infra-ai/paged-serving/pull/24) | 已合并，merge e60a315 |
| trifuse | [#3](https://github.com/open-infra-ai/trifuse/pull/3) | 已合并，merge d64bbc3 |
| cuflash | [#13](https://github.com/open-infra-ai/cuflash/pull/13) | 已合并，merge 2054b71 |
| cuda-foundations | [#16](https://github.com/open-infra-ai/cuda-foundations/pull/16) | 已合并，merge 8143f08 |
| meta | [#3](https://github.com/open-infra-ai/open-infra-ai/pull/3) | 本笔记随该 PR 交付，最终状态以 PR 元数据为准 |
| 面试准备仓 | [#10](https://github.com/open-infra-ai/ai-infra-interview-prep/pull/10) | 同步技术仓真实合并状态，最终状态以 PR 元数据为准 |

技术仓的合并提交已经逐项核验。meta 与准备仓分别交付本记录和任务入口，不预写
自身尚未发生的合并 SHA；准备仓没有远端 CI，使用独立审阅与本地链接、计划一致性
和进度检查作为其验收，不能称为七仓远端 CI 全绿。

## Alternatives considered

squash 能保持默认分支历史简洁，但会丢失已引用证据提交的祖先关系。merge commit
保留这些提交，不重写实验引用。GitHub 可按仓库既有设置自动清理远端 head 分支，
原提交仍由默认分支祖先和 PR 保留；导航使用 commit permalink 或默认分支路径。

等待七仓全部 CI 完成再合入任何一仓，能得到单次收口时间；但当前批次没有新增 C ABI，
技术仓可以逐仓集成。导航与任务卡只引用已完成的真实状态，不宣称多仓事务原子性。

## Verification

Serving 实际执行 282 个默认测试、17 个 doc tests，真实 tokenizer 单独明确 ignored；
MSRV 1.88 locked all-target check、stable clippy/fmt 通过，真实 TCP 四场景另重复
10 轮。DPA 17 项和 Serving 结果审计 40 项通过。六仓各 40 项临时目录真实归档
CLI 回归通过，共 240 项、零 skip；脚本副本 SHA-256 一致。CUDA 基础仓 pre-commit、
8 项站点回归和文档构建通过。最终 head 的远端 CI 与默认分支祖先关系单独复核。

## Consequences

默认分支交付能追溯到审阅和测试所用提交，旧 PR 不再被误当作全部实现。代价是
跨仓导航需要同步，CI 与合并后的检查必须分开记录。没有新 GPU 性能采集；OBS
仍 proposed/pending，GPU 登记/显存回收、持续 lane 与本人面试能力都未因此关闭。
