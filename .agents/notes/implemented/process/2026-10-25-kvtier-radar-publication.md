# Agent Note: kvtier 与 radar 转公开

Status: implemented

## Problem

组织两个私有仓在 free plan 下无分支保护：`kvtier`（研究孵化，公开门槛
四条仅部分达成）、`ai-infra-radar`（生态工作台，daily-scan 自 2026-10-05
暂停）。私有状态同时使状态注册表对组织仓库的覆盖不完整，而 kvtier 已
提交的上游 PR 与发布的评论本就是公开事实，与私有仓库状态互相矛盾。

## Decision

所有者裁定 2026-10-25 两仓全部转公开：

1. `kvtier` 登记入状态注册表（`active`，上游研究线），加入命名冻结；
   证据纪律以诚实口径保持——注册表与其 README 明确「尚无完成审计的
   GPU 结果包、亦无已合入上游 PR」，完成审计前不作为简历主项目。
2. `ai-infra-radar` 保持工作台定位不入注册表（公开但非技术作品）；
   daily-scan 维持暂停，README 标注恢复方法。
3. 两仓公开前完成隐私扫描（无令牌、个人邮箱或 .env）与个人绝对路径
   清理。
4. 分支保护：kvtier required `verify`；radar 基础保护（不设必需检查，
   避免挡住恢复后 daily-scan 机器人对 main 的直推）。

本决策取代 kvtier 孵化隔离规则（工作区 AGENTS.md 原规则 6 与
2026-09-10 笔记，该笔记同批归档）。

## Alternatives considered

- **升级 GitHub Pro 保持私有并获得保护** — 变更最小；但组织定位是
  开源作品集，上游参与记录公开而孵化仓私有的矛盾继续存在。
- **仅 kvtier 公开、radar 保持私有** — radar 的个人 triage 台账公开无
  实质收益；但扫描确认无敏感内容，统一口径消除治理盲区更彻底。

## Consequences

- **收益**：组织 10 仓全部公开且默认分支均受保护；注册表覆盖完整；
  kvtier 的上游贡献记录与孵化过程可见。
- **代价**：公开面新增一个尚无审计结果的 active 仓与一个暂停的工作台，
  需要诚实标注维持证据纪律；radar 的个人参与策略对外可见；上游 PR
  状态漂移需引用前复核。

## Verification

两仓 README 状态段与 kvtier「公开门槛与现状」逐条标注；kvtier 仓内
决策笔记 `2026-10-25-publication-decision.md`（旧门槛笔记已归档）；
根 changelog 2026-10-25 第三批记录；`npm run verify-notes` 通过。

[历史快照：kvtier 孵化仓隔离](../../archived/process/2026-09-10-kvtier-incubator-isolation.md)
