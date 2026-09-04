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
| [paged-serving](https://github.com/open-infra-ai/paged-serving) | [Serving 评测入口](https://github.com/open-infra-ai/paged-serving/tree/master/benchmarks/serving) · [结果报告模板](https://github.com/open-infra-ai/paged-serving/blob/master/benchmarks/serving/RESULT_REPORT_TEMPLATE.md) | 评测方法、数据集、校验器和结果包契约已准备 | **尚无可引用的真实 CUDA closed-loop/Poisson 报告** |
| [cuflash](https://github.com/open-infra-ai/cuflash) | [benchmark 口径](https://github.com/open-infra-ai/cuflash/blob/master/docs/performance/benchmarks.md) · [causal 边界块优化快照](https://github.com/open-infra-ai/cuflash/blob/master/docs/performance/causal-boundary-skip.md) | 指定 RTX 3060 快照与该优化的形状边界 | 不同 GPU 上的通用加速比或生产库等价性 |
| [trifuse](https://github.com/open-infra-ai/trifuse) | [README 验证和 benchmark 边界](https://github.com/open-infra-ai/trifuse#%E9%AA%8C%E8%AF%81) | Triton kernel 与参考实现的数值对照及记录的本机快照 | 跨 GPU 性能普适性 |
| [cuda-foundations](https://github.com/open-infra-ai/cuda-foundations) | [RTX 3060 实测页](https://github.com/open-infra-ai/cuda-foundations/blob/master/docs/en/benchmarks/rtx3060-laptop-2026-08-17.md) | 教学 kernel 的本机优化阶梯和与 cuBLAS 的差距 | 生产算子库或推理系统性能 |

## 旗舰系统当前缺口

`tiny-llm + paged-serving` 已有跨语言正确性链路，但在下列产物归档前，
不宣称真实 GPU Serving 容量、QPS 或生产成熟度：

- 同一结果包中的 `tiny-llm` / `paged-serving` commit；
- 模型 SHA-256、量化格式、GPU/CPU/驱动/CUDA 环境；
- closed-loop 与 Poisson 工作负载配置；
- `per_request.jsonl`、`summary.json`、原始日志和完整复现命令；
- TTFT/TPOT/吞吐的 p50/p95、错误/OOM 计数、token coverage 和已知限制。

## 引用规则

- 履历和面试只引用技术仓中已归档的结果，并保留硬件和 workload 限定。
- 正确性测试数不替代性能证据；单 kernel benchmark 不替代端到端 Serving 结果。
- 无法回溯到原始结果的历史数字不进入履历、首页或新的性能声明。
