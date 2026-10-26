# 2026-10-26 GPU、真实后端与 profiler 证据补强

- **tiny-llm**：新增真实 GPU 非 skip 门禁与脱敏日志；RTX 3060 Laptop 上两种
  GGUF 模型、两种 GQA 几何达到 252/252 通过、0 skip。归档 decode ncu 报告和
  汇总 CSV，明确其局部采样、无 nsys timeline 与无可打开原始报告的限制；
  证据目录的 `.log` 显式豁免全局日志忽略规则。
- **paged-serving**：复跑真实 tiny-llm 后端，固定双仓 commit、库和模型 hash，
  8/8 通过、0 ignored；该本地结果不替代持续 GPU CI、网络故障或生产 SLO 验证。
- **cpp-high-performance-guide**：GCC release 预设加入 CI，本地 release 119/119
  通过；远端 workflow 结果待后续推送确认。`.gitignore` 继续排除 agent 本地
  文件，但放行决策笔记，修复笔记只在本地存在、无法随提交入库的问题。
- **kvtier**：加入 CPU-only 离线测试 CI；完成 WSL2/RTX 3060 Laptop、
  Qwen2.5-0.5B 的 6 轮单卡 L2 回载基线，逐请求 host 命中、counter 和结果审计
  通过，配对 R/P TTFT 比中位数 0.215。保留失败包、上游 commit、本地补丁及
  WSL2 指针根因，状态保持 active，不宣称 #31600 复现、自研 engine 或通用性能。
- **meta 同步**：README、证据索引、毕业门槛和审计路线图纳入上述结果；持续
  GPU/Sanitizer、完整 profiler、公平外部 baseline、多卡和生产 SLO 保持开放。

## 笔记判定

本批 meta 改动是当前事实与证据导航同步，无新的工作区治理决策。技术决策与验证
记录分别保存在 `tiny-llm`、`paged-serving`、`cpp-high-performance-guide` 和
`kvtier` 各仓的 `.agents/notes/`，不在 meta 仓复制第二份。
