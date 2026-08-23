# FFI logprobs 缓冲区契约澄清

- 在 live 跨仓契约中明确 `tinyllm_step` 的 `next_tokens` 与 `logprobs` 最小容量。
- `logprobs` 采用逐序列、逐候选的 `(token_id, logprob)` 两个 `float` 交错布局，
  因此总容量为 `num_sequences * logprobs_k * 2`，不是候选对的数量。
- 明确负 `logprobs_k`、超过词表大小以及请求 logprobs 但传空缓冲区均为参数错误。
- 本次为现有 ABI 实现的契约澄清与参数校验收紧，不改变函数签名或结构体布局。
