# Agent Note: 接入 writing-for-agents 与 documentation-writer 两个写作 skill

Status: implemented

## Problem

工作区的文档有两个不同读者群——agent（AGENTS.md、skill、指针文档）和人
（README、docs/ 站点）。此前没有写作规范：给 agent 看的文档容易写成散文堆叠
（指针措辞弱、信息层级混乱、否定式指令），给人看的文档容易混象限
（README 里塞教程、指南里夹参考表）。

## Decision

7 个仓统一安装两个互补 skill，各仓 `AGENTS.md` 加「写文档」指针节：

- `.agents/skills/writing-for-agents/`（mattpocock/skills）：写给 agent 的文档
  ——context pointer、信息层级（in-file step / in-file reference / disclosed
  reference）、leading word、pruning 与 no-op 判定。
- `.agents/skills/documentation-writer/`（github/awesome-copilot）：写给人的技术
  文档——Diátaxis 四象限（tutorial / how-to / reference / explanation），
  先澄清类型、受众、目标、范围再动笔。

分工即指针的两条触发支线：改 AGENTS.md/skill → 前者；改 README/docs → 后者。

## Alternatives considered

- **只装 documentation-writer** — 覆盖读者面更「正式」；但本工作区改动最大的
  恰是 agent 侧文档（AGENTS.md、笔记体系），缺了 writing-for-agents 等于
  只治一半。
- **把两个 skill 的规则摘编进 AGENTS.md 正文** — 单文件自洽最强；但两份
  skill 加起来上百行，全塞进去正是 writing-for-agents 反对的 context load
  浪费——指针单行 + 按需加载才是它的建议形态。
- **自建写作规范** — 可定制；但两个上游 skill 已经过公开迭代，自写等于
  重造低质量副本。

## Consequences

- **收益**：agent 文档与人读文档各有方法论锚点；指针单行成本换两份完整
  参考。writing-for-agents 同时也规范了笔记体系自身的指针措辞。
- **代价**：documentation-writer 的「澄清→提纲→批准」流程较重，小改
  README 照走全流程会显得繁琐——日常小修直接改，新文档/大改再走全流程。

## Verification

各仓 `.agents/skills/` 下两目录存在且 SKILL.md frontmatter 完整；
各仓 AGENTS.md 有「写文档」节两条指针。
