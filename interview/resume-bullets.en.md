# Resume bullets (English)

Numbers only from [`NUMBERS_CARD.md`](NUMBERS_CARD.md). Each bullet ≤25 words, ending `→ E<n>` for [`EVIDENCE_MATRIX.md`](EVIDENCE_MATRIX.md).

---

## Overview (3)

1. Built a four-layer, five-repo CUDA-to-serving portfolio with independently testable boundaries. → E30
2. Transposed M==1 GEMM cut TPOT 24.348→6.087 ms (1.65× vs llama.cpp, different quant). → E15
3. Rust control plane: 3 concurrent paged requests match llama.cpp greedy; quant split recorded. → E21

## cuda-foundations (3)

1. SGEMM ladder 0.58/0.92/0.66/0.68/1.09 TFLOPS; slower padding step kept in the table. → E1
2. Demoted 04-inference-engine to a teaching preview; runtime must not include it. → E26
3. Renamed cuda-kernel-academy→cuda-foundations; old slug has 0 hits in product repos. → E27

## triton-fused-ops (3)

1. RMSNorm+RoPE, SwiGLU, and FA forward each have standalone references and diffs. → E3
2. Fixed TRIT-001: RoPE uses Llama/Qwen half-split concat, not repeat_interleave. → E4
3. Registered `torch.ops.triton_ops.*`; compile smoke is an honest skip, not a pass. → E5

## cuflash (3)

1. Multi-dtype FA fwd/bwd diffs; flattened grid.y for B×H>65535 with a smoke test. → E7
2. Causal boundary-block skip measured ±2% and documented as a negative result. → E8
3. Implemented FlashDecoding Split-KV with a CPU reference and chunk-count invariant. → E9

## tiny-llm (3)

1. One-command GGUF path generates from real Qwen2.5-0.5B Instruct. → E10
2. Tokenizer matches HuggingFace token-by-token on 30 cases / 417 tokens. → E11
3. CUDA Graphs on by default; greedy tokens match graphs-off. → E16

## paged-serving (3)

1. C ABI v2 is nine ints; Rust guards `sizeof(TinyLlmConfig)==36`. → E19
2. Property tests lock `used+free==total` across cancel and failure. → E23
3. OpenAI API, SSE, and `paged_*` metrics: 37 server integration tests. → E25

(Token-parity e2e is overview bullet 3, to avoid repeating the same number.)

---

## If the interviewer has no GPU

Do not invent a live bench. Point at frozen artifacts.

| Claim | GPU-free proof | Pointer |
|-------|----------------|---------|
| Portfolio exists | Six GitHub repos; interview pack in this folder; `phase-2-e` on product repos | E30 |
| GEMM ladder | Benchmark page includes the slower bank-conflict-free step | E1 |
| Triton contract | TRIT-001 commit + torch.library schemas in README | E4, E5 |
| FA honesty | `causal-boundary-skip.md` states gain below noise | E8 |
| Runtime numbers | `FREEZE_AUDIT.md` + decode-optimization table copied into NUMBERS_CARD | NUMBERS_CARD §1 |
| Scheduler | Default 218 `cargo test` without `--features tiny-llm` | E23, freeze audit |
| Token parity | Fixture ids and is/equals comments in e2e source; T1 did not rerun GPU e2e | E18, E21 |
| Forbidden | No 3030/5118 MiB; no ncu; skips are not passes | E22, E28 |

Spoken line: "I cannot reproduce GPU timings on this call; NUMBERS_CARD row N has the commit and command."
