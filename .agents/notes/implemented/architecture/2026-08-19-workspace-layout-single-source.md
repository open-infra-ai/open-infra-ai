# Agent Note: 并排检出工作区，meta 仓单源承载计划

Status: implemented

## Problem

本地工作区同时检出多个仓库，早期工作区根与 meta 仓各存一份计划，两份漂移；
根目录混入 `.git` / 工具配置残留，Agent 分不清权威来源。复现命令又引用了
`models/` 这类工作区级路径，随便挪动会破坏既有证据入口。

## Decision

工作区根 `/home/shane/github/open-infra-ai` 是**非 git 并排检出工作区**：
不建仓级 `.git`，治理约定写在根 `AGENTS.md`，治理变更记录在根 `changelog/`。

- 计划、landing、学习路径以 meta 仓 `open-infra-ai/` 为唯一源；根上历史计划
  stub 已于 2026-08-21 删除。
- `models/` 路径被复现命令引用，保持原位不挪。
- 各技术仓在根下并排检出，独立 git 历史、独立远端 `github.com/open-infra-ai/*`。

## Alternatives considered

- **根目录做成 git 仓 / workspace manifest 仓** — 一次 clone 拿走全部最有诱惑；
  但各仓要独立发布到 GitHub 组织，嵌套 `.git` 会把历史搅在一起，公开仓与本地
  工作区拓扑耦合。
- **根上保留指向 meta 的计划 stub** — 曾作为过渡方案存在；stub 仍是指针级重复，
  会随 meta 仓内容漂移，2026-08-21 已删除并核实为纯指针。

## Consequences

- **收益**：计划与契约单源，Agent 不会再改错副本；技术仓独立演进、独立发布。
- **代价**：根 `changelog/` 与 `AGENTS.md` 自身不在任何 git 仓内，无版本历史；
  治理变更只能靠 changelog 文件自身记日期追溯。

## Verification

根目录无 `.git`；`open-infra-ai/` 含 `LEARNING_PATH.md`、`docs/`、`archive/plans/`；
根 `changelog/2026-08-19-workspace-governance.md` 与 `2026-08-21-org-reorg.md`
记录本次收口与 stub 删除。
