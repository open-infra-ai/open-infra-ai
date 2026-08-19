# 面试呈现清单

Phase 3 本地交付核对。`phase-3-interview` tag 已推送（meta = `9e0b4f7`），五仓 `phase-3-docs`，六仓 ahead 0。本清单用于面试前核对，不代替实际操作。

## GitHub profile 建议

Pinned 顺序（与讲述优先级一致）：

1. [tiny-llm](https://github.com/open-infra-ai/tiny-llm) — 旗舰 runtime
2. [cuflash-attn](https://github.com/open-infra-ai/cuflash-attn) — kernel 深度
3. [paged-infer](https://github.com/open-infra-ai/paged-infer) — serving 控制面
4. [cuda-foundations](https://github.com/open-infra-ai/cuda-foundations) — L1 教学
5. [triton-fused-ops](https://github.com/open-infra-ai/triton-fused-ops) — 同题异构 / torch.library
6. [aicl-lab](https://github.com/open-infra-ai/aicl-lab) — landing + 本面试包

Landing 一句话：四层学习链，不是迷你 vLLM。

## 各仓 README 复查要点

| 仓 | 必须能立刻看到 | 禁止出现 |
|----|----------------|----------|
| tiny-llm | TPOT 6.09/6.1；W8A16；策略 1；graphs 默认 | 「待 GPU」；比 llama.cpp 快且不提量化 |
| paged-infer | 策略 1 默认；3 并发对齐；is/equals 诚实 | 3030 vs 5118；生产并发 |
| cuflash-attn | grid.y 修复；causal skip **负结果**；FlashDecoding | ±2% 当加速成功；LogicalHBM=物理带宽 |
| triton-fused-ops | 三 op + torch.library；TRIT-001 | 假 FP8 E4M3；compile skip 当 pass |
| cuda-foundations | 冻结；阶梯含更慢的 padding 步；04 预览 | 209/209 全执行；旧 slug 当现名 |
| aicl-lab | 五仓地图 + Interview Evidence 链接 + tag 链 | 把五仓源码改动混入 meta、或把 `phase-3-docs`/`phase-3-interview` 链说反 |

## 面试前 24h 检查

```bash
cd /home/shane/github/aicl
for d in cuda-foundations triton-fused-ops cuflash-attn tiny-llm paged-infer aicl-lab; do
  (cd $d && echo "== $d ==" && git status -sb && git log --oneline -1 && git tag --points-at HEAD)
done
```

抽测（有 GPU、有时间再跑；无 GPU 用 `interview/FREEZE_AUDIT.md`）：

```bash
# 旗舰：tiny-llm 测试（需 TLLM_GGUF_TEST_MODEL）
# paged-infer 默认 CI（不含 e2e）
cd /home/shane/github/aicl/paged-infer && cargo test
```

对照 [`FREEZE_AUDIT.md`](FREEZE_AUDIT.md) 的 skip 数，不要现场「再优化一下数字」。

## 线上面试 demo 顺序（有 GPU）

1. **tiny-llm bench**（主数字）  
   `./build/tiny_llm_bench ../models/qwen2.5-0.5b-instruct-q4_k_m.gguf --prompt "你好" --max-tokens 64 --warmup 3 --iters 5`  
   报 TPOT，立刻补一句：llama.cpp 同卡 3.7 ms，比值 1.65，量化不同。
2. **paged-infer 3 并发**（正确性，需 `--features tiny-llm`）  
   命令见 E18/E21。开口先说：默认 CI 218 不含这次 GPU e2e。
3. **triton op schema**（接入形态）  
   `python -c "import torch, triton_ops; print(torch.ops.triton_ops.sgemm)"`

无 GPU：打开 [`NUMBERS_CARD.md`](NUMBERS_CARD.md) 第 9 节五个数 + [`resume-bullets.zh.md`](resume-bullets.zh.md) 备选表。

## 讲述顺序（10 分钟总叙事）

1. 电梯版 [`talks/00-master-narrative.md`](talks/00-master-narrative.md)
2. 故事① 6.6×→1.65×
3. 故事② ABI v2 策略 1
4. 故事③ `append_kv_at`
5. 停。等追问再进单仓稿。

## 红灯（再读一遍）

见 [`MOCK_INTERVIEW.md`](MOCK_INTERVIEW.md) 第 4 节。最容易说漏嘴的三句：LogicalHBM、±2% 成功、3 并发生产能力。
