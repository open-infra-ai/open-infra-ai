# Agent Note: 状态注册表三处同步

Status: implemented

## Problem

作品集需要对内外回答同一问题：「这个仓现在做到哪了？」答案散落在三个表面——
meta 仓 README 注册表、各仓 README 状态行、GitHub topics。三面各自更新必然漂移：
2026-09-04 `cuda-foundations` 收敛为 `stable` 后，其 GitHub topic 仍挂着
`active`，直到 2026-09-10 复盘才被发现。

## Decision

状态语义三级：`active`（演进中）/ `stable`（作品完成，只修正确性 bug 与文档）/
`archived`（不再维护）。meta 仓 README 的注册表是唯一权威；任何状态变化在
**同一批改动**里同步三处：meta README 注册表、该仓 README 状态行、GitHub topics。

## Alternatives considered

- **只留 meta 注册表一处** — 零同步成本最强；但各仓 README 与 GitHub topics 是
  访客实际先看到的表面，只留一处等于对外没有状态信号。
- **各仓 README 为权威、meta 聚合** — 贴近仓内事实；但作品集视角需要一张总表
  作权威判定，两处权威等于没有权威。

## Consequences

- **收益**：对外信号一致且可审计；发现漂移 = 有变更没走同步批。
- **代价**：同步是人工纪律（topics 无本地文件可校验）；已知漂移案例见
  工作区根 `changelog/2026-09-10-public-surface-fixes.md`
  （cuda-foundations topic 滞后），推送时需人工核对 topics。

## Verification

检查方法：meta `README.md` 状态注册表 ↔ 各仓 README「状态」行 ↔
`gh repo view <repo> --json repositoryTopics`。2026-09-27 实测发现
`cuda-foundations` 的 GitHub topics 缺少状态词（注册表与 README 为
`stable`），为本规则的进行中漂移项。另注意 meta 注册表目前仍把私有仓
`kvtier` 列为 `active` 并挂了公开链接——与
[孵化仓隔离](2026-09-10-kvtier-incubator-isolation.md) 冲突，待处置。
