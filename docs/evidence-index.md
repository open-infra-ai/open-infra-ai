# 作品集证据索引

本页只负责跨仓导航。原始数据、复现命令和限制由产生证据的技术仓维护，
不在 meta 仓复制第二份结果。

## 证据层级

1. **真实实验结果**：绑定 commit、模型/输入、硬件与软件环境，保留原始输出和汇总。
2. **真实硬件正确性**：在记录的 GPU 环境上通过差分、属性或端到端测试，
   但不自动构成性能结论。
3. **CPU/CI 回归**：证明接口、调度、参考模型或文档未回退，不替代 CUDA 运行时验证。
4. **脚手架或报告模板**：只表示实验已可准备；在产生绑定环境的结果前，不得称为已完成实验。

## 当前可导航证据

| 仓库 | 可审计入口 | 能证明什么 | 不能证明什么 |
|------|--------------|--------------|----------------|
| [tiny-llm](https://github.com/open-infra-ai/tiny-llm) | [CUDA Graphs A/B 报告](https://github.com/open-infra-ai/tiny-llm/blob/master/docs/performance/results/2026-08-23-cuda-graphs-ab.md) · [summary JSON](https://github.com/open-infra-ai/tiny-llm/blob/master/docs/performance/results/data/2026-08-23-cuda-graphs-ab.summary.json) · [raw JSONL](https://github.com/open-infra-ai/tiny-llm/blob/master/docs/performance/results/data/2026-08-23-cuda-graphs-ab.raw.jsonl) | 报告所述硬件、模型、shape 和 decode 设置下的配对 A/B | 其他 GPU、模型、长上下文或生产 Serving 性能 |
| [paged-serving](https://github.com/open-infra-ai/paged-serving) | [P1 历史基线](https://github.com/open-infra-ai/paged-serving/tree/master/benchmarks/serving/results/2026-09-04-RTX3060Laptop-paged-serving) · [P2 流式矩阵](https://github.com/open-infra-ai/paged-serving/tree/master/benchmarks/serving/results/2026-09-04-RTX3060Laptop-paged-serving-p2-streaming) · [批量末端后处理 canary](https://github.com/open-infra-ai/paged-serving/tree/master/benchmarks/serving/results/2026-09-05-RTX3060Laptop-paged-serving-p2-batch-postprocess-canary) · [当前正式矩阵](https://github.com/open-infra-ai/paged-serving/tree/master/benchmarks/serving/results/2026-09-07-RTX3060Laptop-paged-serving-p2-batch-postprocess-streaming) · [评测入口](https://github.com/open-infra-ai/paged-serving/tree/master/benchmarks/serving) | 绑定 RTX 3060 Laptop、模型 SHA-256、双仓 commit、原始请求与报告的真实 CUDA closed-loop/Poisson 观察；当前 21-run 矩阵验证当前提交上的首文本 TTFT、Poisson seed、成功率与 429 边界 | 跨 GPU/模型/引擎的通用容量或生产 SLO；当前 c2/c8 与三档 Poisson 仍有未收敛 TTFT p95，且没有配对基线，不能形成批量末端后处理的速度比值；canary 为 `n=1`、无预热，不能作性能结论 |
| [cuflash](https://github.com/open-infra-ai/cuflash) | [benchmark 口径](https://github.com/open-infra-ai/cuflash/blob/master/docs/performance/benchmarks.md) · [causal 边界块优化快照](https://github.com/open-infra-ai/cuflash/blob/master/docs/performance/causal-boundary-skip.md) | 指定 RTX 3060 快照与该优化的形状边界 | 不同 GPU 上的通用加速比或生产库等价性 |
| [trifuse](https://github.com/open-infra-ai/trifuse) | [README 验证和 benchmark 边界](https://github.com/open-infra-ai/trifuse#%E9%AA%8C%E8%AF%81) | Triton kernel 与参考实现的数值对照及记录的本机快照 | 跨 GPU 性能普适性 |
| [cuda-foundations](https://github.com/open-infra-ai/cuda-foundations) | [RTX 3060 实测页](https://github.com/open-infra-ai/cuda-foundations/blob/master/docs/en/benchmarks/rtx3060-laptop-2026-08-17.md) | 教学 kernel 的本机优化阶梯和与 cuBLAS 的差距 | 生产算子库或推理系统性能 |

## 旗舰系统当前缺口

`tiny-llm + paged-serving` 已有跨语言正确性链路及多份真实 CUDA serving 结果包；
但下列缺口关闭前，仍不宣称通用容量、跨引擎排名或生产成熟度：

- 当前 2026-09-07 矩阵中，closed c1/c4 通过 TTFT p95 与吞吐的 10% 重复波动检查，
  c2/c8 与三档 Poisson 的 TTFT p95 仍未通过；Poisson 0.64 / 1.28 req/s 还分别出现
  9 / 74 个 HTTP 429；
- 发压机与服务同机，未采集 GPU 时钟/功耗，且没有独立网络发压机；
- 尚无 llama-server/vLLM 的同模型、同上下文、同工作负载对照；
- 当前 tiny-llm FFI 的 Transformer layer forward 仍逐序列执行；正常 greedy 的末端已改为
  GPU batch final RMSNorm / LM head / argmax 与一次 batch token 回传，但 logprobs 仍走主机
  logits 路径，不能把调度 batch 当作 fused compute batch；当前矩阵没有配对基线，不能将
  任何差异归因于这项末端后处理；
- KV 利用率采样、prefix cache、抢占、chunked prefill 与 token 级 ITL 仍未实现。

## 引用规则

- 履历和面试只引用技术仓中已归档的结果，并保留硬件和 workload 限定。
- 正确性测试数不替代性能证据；单 kernel benchmark 不替代端到端 Serving 结果。
- 无法回溯到原始结果的历史数字不进入履历、首页或新的性能声明。
