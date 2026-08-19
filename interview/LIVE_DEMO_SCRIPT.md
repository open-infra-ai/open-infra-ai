# 线上面试 demo 脚本

> Phase I4。三种场景 + preflight + 失败恢复树。命令以仓库 README 与 EVIDENCE_MATRIX 复现段为准。
> 数字只从 [`NUMBERS_CARD.md`](NUMBERS_CARD.md) 取；所有命令先在本机彩排一遍，别现场改参数。

---

## 0. Preflight（demo 前 10 分钟）

一条命令做六仓 `git status -sb` + 最近 commit + HEAD tag 检查：

```bash
cd /home/shane/github/aicl
for d in cuda-foundations triton-fused-ops cuflash-attn tiny-llm paged-infer aicl-lab; do
  (cd $d && echo "== $d ==" && git status -sb && git log --oneline -1 && git tag --points-at HEAD)
done
```

- 期望：六仓 `## master...origin/master`、ahead 0；五仓 HEAD 显示 `phase-3-docs`，meta 显示 `phase-3-interview`。
- 若某仓 ahead 非 0：先 `git push` 或记下偏差，demo 时不刷状态。

---

## 1. 场景 A（有 GPU）

固定顺序：tiny-llm bench → paged-infer 3 并发 → triton op schema。

### Step 1 · tiny-llm bench（主数字）

**命令**：

```bash
cd /home/shane/github/aicl/tiny-llm
./build/tiny_llm_bench ../models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --prompt "你好" --max-tokens 64 --warmup 3 --iters 5
```

**预期关键输出**：TPOT mean `6.087 ms/token`（表内写作 6.09 / README 6.1）；TTFT `10.567 ms`；decode `164.283 tok/s`（数字卡 §1，commit `f897084`）。

**15 秒口播词**：“这是我们 0.5B 模型在本机 W8A16 路径的 decode 中位延迟 6.09 ms；同卡 llama.cpp 是 3.7 ms，比值 1.65——注意两边量化不同，llama.cpp 是原生 Q4_K_M。”

**失败切换**：bench 起不来 → 不再修，直接切到场景 B：打开数字卡第 9 节报同组数字。

### Step 2 · paged-infer 3 并发（正确性）

**命令**：

```bash
cd /home/shane/github/aicl/paged-infer
TINY_LLM_DIR=/home/shane/github/aicl/tiny-llm \
TINY_LLM_MODEL=/home/shane/github/aicl/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
PINF_TOKENIZER_JSON=/home/shane/github/aicl/models/tokenizer.json \
  cargo test --features tiny-llm --test tiny_llm_text_e2e -- --nocapture
```

**预期关键输出**：
- 请求 1：全序列 24 个 id 与 llama.cpp greedy 相等，末位 EOS `151645`（`Hello! I'm just a computer program...`）；
- 请求 2：前缀 `[17,10,17]` 一致，第 4 个 token `374`（是/is）vs 16819（equals），两边都到 EOS `151645`；
- 结束断言 `active_sequences==0`（数字卡 §7，commit `9c3700b` + `9c974d3`）。

**15 秒口播词**：“这个测试用真实 Qwen 跑 3 并发分页请求；请求 1 全序列和 llama.cpp 严格一致，请求 2 我们诚实记录量化的 is/equals 分支——注意默认 CI 的 218 个测试没开这个 feature，这是单独开的 e2e。”

**失败切换**：e2e 起不来 → 切回“正确性留口”：报 E20 的策略差分测试名与 218 测试数，打开 `FREEZE_AUDIT.md` §3.5。

### Step 3 · triton op schema（接入形态）

**命令**：

```bash
cd /home/shane/github/aicl/triton-fused-ops
.venv/bin/python -c "import torch, triton_ops; [print(getattr(torch.ops.triton_ops, n)) for n in ['sgemm','fused_rmsnorm_rope','fused_gated_mlp']]"
```

**预期关键输出**：三行 schema，形如 `sgemm(Tensor A, Tensor B, ...) -> Tensor`、`fused_rmsnorm_rope(...)`、`fused_gated_mlp(...)`（见 E5，注册名 `triton_ops::*`）。

**15 秒口播词**：“这是我用 torch.library 把三个 Triton 算子注册成 `torch.ops.triton_ops.*`，和 vLLM 的 custom op 接入模式同类；编译 smoke 测试是 skip，我不拿它当通过。”

**失败切换**：venv/import 报错 → 关掉屏幕共享，切一句话：“三个 schema 已注册，具体输出在 E5 的复现命令里”，然后进入反问。

---

## 2. 场景 B（无 GPU / 共享屏幕）

打开两个文件，不跑 GPU：`interview/NUMBERS_CARD.md` 第 9 节（五个首选数）+ `interview/FREEZE_AUDIT.md` 测试表。

**三句口播词**：
1. “我报五个核心数字：TPOT 6.09 ms，lm_head 从 10.0 降到 0.98 ms，tiny/llama 1.65×（非同量化），请求 1 的 24+EOS 全等、请求 2 的 is/equals 我如实记录，还有 causal skip 的 ±2% 负结果。”
2. “这些数字都在数字卡里带本机硬件、commit 和复现命令；换到别的卡我不沿用。”
3. “测试规模看这张表：cuflash 71 项、tiny-llm 175、paged-infer 218，skip 的门控原因都在 freeze audit 里写清楚了。”

---

## 3. 场景 C（面试官自己跑命令）

三条可直接粘贴到对方终端的命令（**在你自己终端也先验证一遍**）。每条标预计耗时与 skip 门控。

**C1 · tiny 数字（预计 1 分钟内，需要已构建 binary + GGUF）**

```bash
cd /home/shane/github/aicl/tiny-llm
./build/tiny_llm_bench ../models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --prompt "你好" --max-tokens 64 --warmup 3 --iters 5
```

skip 门控：无 `tiny_llm_bench` 或 GGUF 文件，则改用 `interview/NUMBERS_CARD.md` 第 9 节。

**C2 · triton schema（预计 10 秒内，需要 venv）**

```bash
cd /home/shane/github/aicl/triton-fused-ops
.venv/bin/python -c "import torch, triton_ops; print(torch.ops.triton_ops.sgemm)"
```

skip 门控：venv 未装依赖，则口头报 E5 的三个注册名。

**C3 · paged 3 并发 e2e（预计 2–3 分钟，需要 backends 构建 + 模型 + tokenizer.json）**

```bash
cd /home/shane/github/aicl/paged-infer
TINY_LLM_DIR=/home/shane/github/aicl/tiny-llm \
TINY_LLM_MODEL=/home/shane/github/aicl/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
PINF_TOKENIZER_JSON=/home/shane/github/aicl/models/tokenizer.json \
  cargo test --features tiny-llm --test tiny_llm_text_e2e -- --nocapture
```

skip 门控：模型/tokenizer 缺失 → 改跑无 feature 的 `cargo test`（218 项）并说明 e2e 本次未开。

---

## 4. 失败恢复树

| 分支 | 一句应对 |
|------|----------|
| bench 报错 | “这个环境缺预构建/依赖，我不现场修；给你看同组数字的数字卡第 9 节。” |
| OOM | “6GB 卡撑不住更多并发，我这边本来就是单卡环境；降到单请求重新跑，或只讲 E20 的差分证据。” |
| tokenizer 缺失 | “HF tokenizer 的 json 不在这个环境，我改跑默认 `cargo test`，e2e 单独说明。” |
| 无 GPU | “无 GPU 就走场景 B：数字卡五个数 + freeze audit 测试表，全部命令和硬件口径都在文档里。” |
