# Agent Note: 组织链接唯一写法与仓名冻结

Status: implemented

## Problem

组织两次改名（AICL-Lab → aicl-lab → open-infra-ai）后，`github.com/aicl-lab/*`
只靠 GitHub 301 重定向工作，而旧 Pages 域名 `aicl-lab.github.io` 已真实 404；
五仓用户可见链接、徽章、Pages 部署门控分散指向新旧两套地址。仓名也经历过
多次变更，简历与证据材料已引用现有仓名。

## Decision

1. 用户可见链接一律写 `github.com/open-infra-ai/{repo}` 与
   `open-infra-ai.github.io/{repo}/`（小写组织名，与真实 remote 一致）。
2. 技术仓名冻结：`tiny-llm` / `cuflash` / `paged-serving` / `trifuse` /
   `cuda-foundations`——已被简历与证据材料引用。meta 仓与组织同名，
   组织改名时同批改名。
3. 豁免保护历史叙述（审计快照、CHANGELOG 历史条目、archive/plans、interview/
   中的旧名），功能性内容（命令、路径、URL）随改名同步维护——边界见
   [历史记录不可改写](2026-09-10-immutable-historical-records.md)。

## Alternatives considered

- **保留旧链接靠 301 维持** — 零改动最省事；但 Pages 域名已实际失效、301 不覆盖
  Pages，2026-08-19/08-21 两轮规范化修的都是真断链。
- **仓名随项目演进自由改** — 命名永远贴合现状最整齐；但外部引用（简历、
  证据矩阵、58 个 commit SHA）会腐烂，kvtier 三次改名的成本已经演示过一遍。

## Consequences

- **收益**：公开身份稳定，链接可机械检查（搜索旧组织名即可发现违规）。
- **代价**：未来的任何改名都是跨仓、跨外部材料的批量操作；`kvtier` 在孵化期
  不适用本冻结（见 [孵化仓隔离](2026-09-10-kvtier-incubator-isolation.md)）。

## Verification

`rg -i 'aicl-lab|AICL-Lab'` 命中应只剩豁免文件；远端列表
`git remote -v` 全部为 `github.com/open-infra-ai/*`。
