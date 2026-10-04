# Agent Note: 全仓接入 write-notes-like-deepseek 决策笔记体系

Status: implemented

## Problem

AI 高密度改动下，「为什么这样定、放弃了什么」只存在于 PR 描述与聊天上下文；
根 `AGENTS.md` 的治理规则是散文式约定，没有机械牙齿。新会话看不到被否过的
方案，容易重走回头路或击穿既有边界（如
[ABI 双源契约](../architecture/2026-08-21-abi-contract-dual-source.md)）。

## Decision

7 个仓（本仓 + tiny-llm / cuflash / paged-serving / trifuse / cuda-foundations /
kvtier）统一接入 write-notes-like-deepseek 体系：

- `.agents/notes/{proposed,implemented,rejected,archived} × 6 class`，
  路径即状态，不建 INDEX。
- `.agents/skills/write-notes-like-deepseek/` 承载 SKILL.md + references +
  templates；`scripts/*.ts` 为校验脚本（`npm run verify-notes` 串跑三线）。
- 各仓 `.github/workflows/verify-notes.yml` 在 push/PR 时跑三线校验；
  各仓 `AGENTS.md` 写入笔记纪律，有 CONTRIBUTING.md 的仓补一条要求。
- 本批同时把已在 changelog / 契约文档中生效的决策回写为 implemented 笔记。

[归档预检](../bug-fix/2026-10-04-archive-preflight.md)部分细化归档 CLI 的输入门禁，
[完整交付命令](2026-10-04-delivered-note-commands.md)收窄可运行入口；两者不替代
笔记归属、分类和跨仓治理决定。

## Alternatives considered

- **继续只靠 CHANGELOG + 治理散文** — 零新机制最强；但 changelog 记「做了什么」
  不记「放弃了什么」，AGENTS.md 规则没有强制门，新会话照样看不见。
- **只在 meta 仓建一套笔记树** — 一处维护最简；但违背「笔记与代码同仓」
  原则，技术仓内决策没有落点。
- **逐仓试点后再铺开** — 更稳；但 7 仓同批铺开能让跨仓契约类笔记立刻互链，
  且骨架成本极低，一次到位。

## Consequences

- **收益**：决策有机械校验（目录/状态/备选方案/归档封印），被否方案沉淀在
  `rejected/` 与 `archived/` 防重犯。
- **代价**：C++/Rust 仓引入 package.json + npx tsx 校验依赖（Node ≥18）；
  纪律维护成本前置到每次非平凡改动。

## Verification

各仓 `npm run verify-notes` 全绿；CI workflow `verify-notes.yml` 已接入；
skill 本体在 `.agents/skills/write-notes-like-deepseek/`。
