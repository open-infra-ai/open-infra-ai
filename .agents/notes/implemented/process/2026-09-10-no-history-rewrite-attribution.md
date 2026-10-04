# Agent Note: 不为清除 AI 工具尾标重写历史

Status: implemented

## Problem

组织内 7 个仓共有 184 个提交带第三方 AI 工具尾标（Copilot / Cursor / Devin /
Qwen-Coder 等）。直觉反应是 rewrite + force-push 清干净；在动手前需要先验证
「重写能否达成目标、代价是什么」。

## Decision

不重写历史。这是实测结论，不是口味偏好：

1. **目标不可达成**：绝大多数尾标由 GitHub 合并 PR 时生成；`refs/pull/*` 永久
   保留（全组织 52 个，11 个含尾标）。实测 PR #15 的分支提交不在 master 上，
   仅靠 `refs/pull/15/head` 存活，API 可解析、网页 HTTP 200——重写 master
   清不掉这些公开副本。
2. **代价过高**：meta 仓证据材料引用 58 个可解析 commit SHA；重写令其全部失效，
   与「可追溯证据」原则直接冲突；38 个 tag 移位、Releases 重新指向。
3. 正确做法是在各工具侧配置提交署名（已做：`attribution.commit` 置空，
   写 `~/.commandcode/settings.json` 与工作区 `.commandcode/settings.json`），
   防止**新**提交带入尾标。

## Alternatives considered

- **rewrite master + force-with-lease** — 最强的「干净历史」表象；但 refs/pull
  不可重写使目标根本不可达，外加 SHA/tag/Release 全部失效，纯亏。
- **什么都不做也不立规** — 零成本；但新提交持续带入尾标，问题只增不减。

## Consequences

- **收益**：58 个被引用 SHA、38 个 tag、全部 release 链接保持可解析。
- **代价**：历史中的尾标永久存在；后人想再清一遍时必须先撞见本篇——这正是
  留下它的原因。

## Verification

根 `changelog/2026-09-10-public-surface-fixes.md` §5-6 记录了实测数据与
两次已清理提交的 tree-hash 比对（`fa14dda`→`2b15fb2`、`aa24144`→`e95bcd3`）。
