# 2026-10-25 注册表扩容与边界执行收口

- `cpp-high-performance-guide` 以 stable 入册状态注册表、90 秒入口、作品集架构图与
  完成证据摘要；边界文档技术仓表扩为七仓，明确其为 CPU 侧教学线、不进入旗舰系统
  集成图。
- `ai-infra-interview-prep` 迁回个人账号 `holtwood` 并转为私有；边界文档第 3 节
  改指个人私有仓，README「求职与面试执行」段与系统地图 backlog 引用同步去链接化，
  并补记 radar / cpp-guide 两个此前零覆盖仓的登记。
- 新增决策笔记 `implemented/process/2026-10-25-portfolio-reconciliation.md`，
  记录「组织 vs 个人账号」的保留判断与三项归属决策。
- README 完成证据摘要由过期快照刷新为可验证的当前 CI 口径：tiny-llm
  **252/252**（2026-10-19）、paged-serving cargo **615 项**（2026-10-04）、
  cuflash RapidCheck **81/81** 与常规 **71/71**（2026-10-19）、trifuse
  CI 档 **65 passed / 66 skipped**（2026-10-19）、cpp-guide **119/119**
  （2026-10-25）；cuda-foundations 的 CI 仅 CPU 构建冒烟，GPU 快照口径保留。

## 同日第三批：kvtier 与 radar 转公开

- 所有者裁定 `kvtier`、`ai-infra-radar` 转公开，组织仓全部公开（10 仓，
  零私有）。
- `kvtier` 以 `active` 入册（上游研究线）并加入命名冻结；其公开门槛四条
  仅部分达成（无审计 GPU 结果包、无已合入上游 PR），注册表与 README 以
  诚实口径标注，完成审计前不作为简历主项目；仓内旧「公开门槛」笔记归档、
  新「提前公开决策」笔记入库，个人绝对路径随公开批次清理。
- `ai-infra-radar` 保持工作台定位不入册（公开但非技术作品），daily-scan
  维持暂停并在其 README 标注恢复方法。
- 两仓公开前完成隐私扫描（无令牌、个人邮箱或 .env）；分支保护：kvtier
  required `verify`，radar 基础保护（不设必需检查，避免挡住恢复后 bot
  对 main 的直推）。
- 新增决策笔记 `implemented/process/2026-10-25-kvtier-radar-publication.md`，
  归档被取代的 `2026-09-10-kvtier-incubator-isolation.md`（其指向
  canonical-org-identity 的相对链接先行改为跨目录可解析形式以满足归档
  预检）。
