# Agent Note: 作品集对账：cpp-guide 入册、执行仓迁出与私有工作台登记

Status: implemented

## Problem

治理文档停在「5 公开技术仓 + kvtier」的世界，组织现实是 10 个仓：
`cpp-high-performance-guide` 与 `ai-infra-radar` 在全部治理文档中零覆盖，而毕业
标准明确写了「不要新增第八个仓库」；`ai-infra-interview-prep` 以「个人执行仓」
名义公开挂在组织下，违反仓库边界文档对组织「只承载可复现的公开技术作品」的
定义，且根 README 与 meta README 对其归属口径互相矛盾；两套 changelog 分别自
2026-09-27 / 2026-08-23 停更，radar 建仓、cpp-guide 入驻、interview-prep 迁移
均无治理记录。

## Decision

1. 保留组织（不并入个人账号、不另立组织），理由：简历与证据材料冻结了
   `github.com/open-infra-ai/{repo}` 链接；组织是策展过的作品集橱窗，与
   holtwood 个人生活项目分离；公开成员身份下贡献图不受影响。
2. `cpp-high-performance-guide` 以 **stable** 入册状态注册表与边界文档（CPU 侧
   教学线，不进入旗舰系统集成图），仓名加入冻结清单。
3. `ai-infra-interview-prep` 迁回个人账号 `holtwood` 并转为私有，组织不再承载
   求职执行材料；边界文档第 3 节改指个人私有仓。
4. `ai-infra-radar` 登记为组织私有工作台（不进公开注册表），每日扫描 cron 暂停
   （2026-10-05 后无人工 triage，自动化产出无人消费），恢复 triage 时再启用。
5. 恢复双 changelog 记录本批变更，并补记此前三次未记录的治理事件。

## Alternatives considered

- **下线组织、并入个人账号** — 少一层跳转；但证据链接全量重写、转移重定向脆弱，
  且与个人 20+ 生活/工具仓混杂会稀释作品集叙事，否决。
- **执行仓留在组织公开面** — 链接零迁移成本；但公司候选池、面试话术与短板自评
  公开对求职没有展示收益，且持续违反组织自身章程，否决。
- **cpp-guide 转回个人账号** — 组织叙事更纯（只 AI Infra）；但它是组织内第二
  成熟的公开作品（约 13.4k 行 C++、测试/CI/文档站俱全），离开作品集是净损失，
  且「open infra」本身覆盖 CPU 侧系统工程，否决。

## Consequences

- **收益**：注册表、边界文档、org profile、根 README 与组织现实一致；公开面不再
  暴露求职过程材料；cpp-guide 的徽章、克隆地址、文档站与 Pages 部署守卫统一归属
  open-infra-ai，文档站恢复自动部署。
- **代价**：`holtwood/ai-infra-interview-prep` 私有后外部链接 404（GitHub 转移
  重定向仍在），系统地图的 backlog 引用改为文字指称；radar 每日数据在 triage
  恢复前不再累积；「七仓」的组成口径需要在系统地图中以文字澄清。

## Verification

根 `changelog/2026-10-25-org-reconciliation.md` 与本仓
`changelog/2026-10-25-registry-and-boundaries.md` 同批记录；
`npm run verify-notes` 通过后随本批提交推送。
