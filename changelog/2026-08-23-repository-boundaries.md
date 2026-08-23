# 2026-08-23 收拢仓库边界与当前证据

- 新增 `docs/repository-boundaries.md`，明确 meta 仓、五个技术仓、个人求职执行仓和
  上游社区贡献的职责边界。
- 将活跃 `job-hunting/` 内容迁往个人仓 `holtwood/ai-infra-interview-prep`；meta 仓
  不再维护简历、公司清单、投递状态与评论草稿。
- 保留 `interview/`、历史计划和 `docs/organization-audit/` 原文，未改写历史事实。
- 更新 README 的 2026-08-23 本地验证快照和面试展示优先级：`tiny-llm` 为推理加速
  主项目，`cuflash-attn` 为 kernel 深挖，`paged-infer` 为 serving 系统扩展。
- 将 `tiny-llm` 的 6.1 ms/token 标为历史 schema v1 跨请求估算；随后在 clean commit
  `565da79` 完成 schema v2 五组配对 CUDA Graph A/B，正式归档 TPOT -37.2%、
  decode 吞吐 +59.3%、10 个进程原始 JSONL与 TTFT 噪声边界。
- 技术仓名称保持冻结，不做会破坏证据链接的大规模重命名。
