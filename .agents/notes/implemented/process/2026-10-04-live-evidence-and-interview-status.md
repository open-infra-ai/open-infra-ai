# Agent Note: 当前证据、任务状态与本人能力分别维护

Status: implemented

## Problem

live 路线把已有 direct/split-KV 实现列为待开发，并将源码、历史测试数转换为本人能力
评分。新会话可能重复开发，简历也可能把 kernel 结果扩写成端到端收益。

## Decision

meta 维护跨仓技术事实和证据导航，个人执行仓维护闭卷诊断、剩余六周任务与模拟成绩。
direct/split-KV 按实现、kernel 实验、端到端缺口分层；主动取消区分 OPEN PR 与默认
分支。引用保留日期、硬件和范围，未测本人能力不由 Agent 自动涨级或勾选完成。

[既有工作区单源笔记](../architecture/2026-08-19-workspace-layout-single-source.md)
只部分重叠：目录与归属决定保持原样，本笔记定义 live 状态的证据解释；
[历史冻结笔记](2026-09-10-immutable-historical-records.md)仍约束归档，原始实验与
历史计划不改写。上述笔记不被改写为不同决定。

## Alternatives considered

把所有路线重新集中到 meta 便于一处浏览，但会混入高频变化的私人求职记录，与已有
内容归属冲突。技术优先级在 meta，个人日程沿用执行仓已有 weekly 文件。

保留旧任务卡并只加顶层免责声明最省编辑，但下一位 Agent 按具体 task ID 仍会误执行。
关键任务卡同步现状，并将已实现与剩余验收分开。

## Verification

live 入口区分已有 direct/split-KV、默认 legacy 和端到端缺口；正式 Serving 结果可导航，
21-run 结构 validator 通过。个人执行仓 `make check verify progress` 通过；
12 周日期保持连续，W7–W12 沿用已有文件，15 道题和三张答辩牌均待本人复测。
旧自评明确标为历史，42 个个人交付物未自动勾选。没有提交、推送或合并 PR。

## Consequences

避免重复开发并保留未收敛与负结果供答辩；代价是多份 live 导航仍需按新实现同步。
日期化快照会过期，新会话必须复核代码、PR 与证据。
此整改不等于 GPU 持续门禁、有界背压或个人面试能力已经完成。
