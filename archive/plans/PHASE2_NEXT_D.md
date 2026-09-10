# PHASE 2 下一批任务（Batch 4：D0 → D5，分页 KV 端到端）

> **生成时间**：2026-08-18（C0–C3 完成后）
> **执行方**：DeepSeek Flash / 其他低成本模型
> **上游文档**：`PHASE2_PLAN.md` 第 7 节（D 阶段）；本批是其可执行细化。
> **状态**：C0–C3 ✅ 完成。tiny-llm HEAD=6d0471e，工作区 clean，**ahead 4 未推送**。
> TPOT 已 24.35 → 6.56ms（graphs 默认开启后 6.09ms），llama.cpp 比值 6.6× → 1.65×。
>
> 本批目标：让 paged-infer 的真实 tiny-llm 后端**真正使用分页 KV（block_tables）**，跑通 3 并发端到端，与 llama.cpp greedy 输出逐 token 一致，资源守恒不变量成立。这是 mini-vLLM 叙事的最后一块旗舰拼图。
>
> 本批所有 GPU 验证命令均在 RTX 3060 Laptop 6GB 上执行；模型文件：
> `/home/shane/github/aicl/models/qwen2.5-0.5b-instruct-q4_k_m.gguf`（下文用 `$MODEL` 表示）；
> tokenizer：`/home/shane/github/aicl/models/tokenizer.json`（下文用 `$TOK` 表示）。

---

## 0. 执行协议

1. 一次一个任务，验收全绿后 commit，再进入下一个。
2. 只改任务列出的文件；发现计划外问题写 NOTE，不扩大范围。
3. **禁区**：
   - 不修改 tiny-llm `KVCacheManager` 的连续 KV 布局（策略 2 仍要可用）；
   - 不修改 paged-infer `scheduler.rs` 的 P0 行为；
   - 不做 chunked prefill、不做抢占、不给 paged 路径接 CUDA Graphs（正确性优先）。
4. 分页 KV 第一版**允许 gather/scatter 多一次显存往返**；先正确后优化，不要顺手写复杂 kernel。
5. 每个任务一个 commit，message 按任务末尾格式。

---

## 任务 D0：推送 C0–C3 提交并打 tag（P0）

```bash
cd /home/shane/github/aicl/tiny-llm
git log origin/master..HEAD --oneline   # 应为 ca70de2 / db5451b / f897084 / 6d0471e
git push origin master
git tag phase-2-c && git push origin phase-2-c
```

**验收**：`git status -sb` 显示 ahead 0；`git ls-remote --tags origin | grep phase-2-c` 有输出。

---

## 任务 D1：ABI v2 同步（C 与 Rust 两侧）

### D1a：tiny-llm 侧（一个 commit）

**改动文件**：`include/tiny_llm/ffi.h`、`src/ffi.cpp`（仅签名与调用点、`(void)` 参数）、`tests/test_ffi.cpp`（所有 `tinyllm_step` 调用点）。

**步骤**：
1. `TinyLlmConfig` 在 `max_batch_size` 后新增：
   ```c
   int max_num_blocks;  // 分页 KV 池的物理块总数；0 = 策略 2（连续 KV）
   ```
   头注释写明：该字段使 `TinyLlmConfig` 变为 9 个 int 的 repr(C) 布局（ABI v2）。
2. `tinyllm_step` 在 `block_tables` 参数后新增 `const int *num_blocks`：
   ```c
   int tinyllm_step(TinyLlmHandle *handle, const int *seq_ids, const int *input_tokens,
                    const int *positions, const int *seq_lens, const int *block_tables,
                    const int *num_blocks, const unsigned char *is_prefill,
                    int num_sequences, int *next_tokens, float *logprobs, int logprobs_k);
   ```
   头注释定义 `num_blocks[i] = 第 i 个序列的 block_tables 长度`（扁平化 block_tables 的总长为 `sum(num_blocks)`）。
3. `src/ffi.cpp` 的 `tinyllm_step` 同步签名；本任务**只加参数并 `(void)num_blocks;`**，行为仍是策略 2。
4. `tests/test_ffi.cpp` 所有调用在 `block_tables` 后补 `nullptr`。
5. 构建与测试。

**验收**：
```bash
cmake --build build -j$(nproc)
./build/tiny_llm_tests
# 期望：170 tests 中 162 passed / 8 skipped（skip 数不变）
```

**提交**：`feat(ffi): ABI v2 add max_num_blocks and per-sequence block counts`

### D1b：paged-infer 侧（一个 commit）

**改动文件**：`src/tiny_llm_ffi.rs`、`src/tiny_llm_executor.rs`、`tests/` 中受影响的调用（如有）。

**步骤**：
1. `TinyLlmConfig` 末尾新增 `pub max_num_blocks: i32`（字段顺序与 C 一致：9 个 int）。
2. 模块顶部 C ABI 契约注释更新为 v2。
3. 布局守卫测试更新：
   ```rust
   assert_eq!(size_of::<TinyLlmConfig>(), 9 * 4);
   ```
   并把测试里的构造加上 `max_num_blocks: 0,`。
4. `symbols::tinyllm_step` 在 `block_tables` 后新增 `num_blocks: *const c_int`。
5. `TinyLlmExecutor::execute` 本任务先传 `std::ptr::null()` 给 `num_blocks`（行为不变，D3 再接入真实块表）。
6. `TinyLlmExecutor::new` 中 `TinyLlmConfig` 构造加 `max_num_blocks: 0,`（本任务仍策略 2）。

**验收**：
```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
# 期望：全部通过（218 passed 基线）
# 有真实模型时（可选，本任务不强制）：
TINY_LLM_DIR=/home/shane/github/aicl/tiny-llm TINY_LLM_MODEL=$MODEL \
  cargo test --features tiny-llm --test tiny_llm_backend -- --nocapture
```

**提交**：`feat(ffi): mirror tiny-llm ABI v2 (max_num_blocks + num_blocks)`

---

## 任务 D2a：paged scatter/gather kernel + 单测（tiny-llm）

**改动文件**：新增 `kernels/paged_kv.cu`、`kernels/paged_kv.cuh`；`tests/test_kernels.cu`；`CMakeLists.txt`（若 GLOB 不生效则显式 append，参照 A0 的做法）。

**kernel 规格（第一版简单正确即可，一个元素一个线程）**：

```cuda
// 把连续 [num_tokens, chunk_dim] 的 src 按块表散布到 pool。
// 绝对位置 = position + t；块内偏移 = abs % block_size；
// pool 传入的是"本层"指针（调用方负责 layer 偏移）。
__global__ void paged_scatter_blocks_kernel(const half *__restrict__ src,
                                            half *__restrict__ pool,
                                            const int *__restrict__ block_table,
                                            int num_tokens, int position,
                                            int block_size, int chunk_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * chunk_dim;
    if (idx >= total) return;
    int t = idx / chunk_dim;
    int c = idx - t * chunk_dim;
    int abs = position + t;
    int b = abs / block_size;
    int block_id = block_table[b];
    int within = abs - b * block_size;
    pool[((size_t)block_id * block_size + within) * chunk_dim + c] = src[idx];
}

// 从块表把 [visible_tokens, chunk_dim] 连续读回 dst。
__global__ void paged_gather_blocks_kernel(half *__restrict__ dst,
                                           const half *__restrict__ pool,
                                           const int *__restrict__ block_table,
                                           int visible_tokens, int block_size, int chunk_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = visible_tokens * chunk_dim;
    if (idx >= total) return;
    int t = idx / chunk_dim;
    int c = idx - t * chunk_dim;
    int b = t / block_size;
    int block_id = block_table[b];
    int within = t - b * block_size;
    dst[idx] = pool[((size_t)block_id * block_size + within) * chunk_dim + c];
}
```

launcher：`paged_scatter_blocks(...)` / `paged_gather_blocks(...)`，block=256，grid=`(total+255)/256`；参数含 `cudaStream_t stream=0`。头文件声明放在 `namespace tiny_llm::kernels`。

**测试（`tests/test_kernels.cu` 新增 4 例）**：
1. `PagedScatterGatherRoundTrip`：block_size=16, chunk_dim=128, num_tokens=17（跨块边界）；scatter 后 gather 回读与源逐元素相等（`1e-2`）。
2. `PagedGatherPartialVisibility`：visible_tokens=7 只读前 7 个位置，与 CPU 参考一致。
3. `PagedScatterWritesAtAbsolutePosition`：position=20（block 1 内 offset 4），验证落在 `block_table[1]` 块的第 4 行。
4. `PagedLayersDoNotOverlap`：两个"层"（不同 pool 偏移）写不同数据，互不污染。

**验收**：
```bash
cmake --build build -j$(nproc)
./build/tiny_llm_tests --gtest_filter='*Paged*'
./build/tiny_llm_tests
```

**提交**：`feat(kernels): paged KV scatter/gather primitives with round-trip tests`

---

## 任务 D2b：tiny-llm 分页 KV 池与句柄资源（tiny-llm）

**改动文件**：`src/ffi.cpp`（句柄结构、`tinyllm_load`、`tinyllm_free`、allocate/free 序列的 paged 分支）。

**步骤**：
1. `TinyLlmHandleImpl` 新增：
   ```cpp
   bool paged_kv = false;
   int  max_num_blocks = 0;
   int  max_visible_tokens = 0;
   half *paged_k_pool = nullptr;   // [L * max_num_blocks * block_size * kv_dim]
   half *paged_v_pool = nullptr;
   half *paged_k_scratch = nullptr; // [max_visible_tokens * kv_dim]
   half *paged_v_scratch = nullptr;
   tiny_llm::DeviceBuffer<int> d_block_tables; // 容量 max_num_blocks
   ```
2. `tinyllm_load`：当 `config != nullptr && config->max_num_blocks > 0` 时：
   - 校验 `config->block_size > 0` 且 `max_num_blocks > 0`，否则 err 返回；
   - `h->paged_kv = true; h->max_num_blocks = config->max_num_blocks;`
   - `kv_dim = num_kv_heads * head_dim`；`max_visible_tokens = max_num_blocks * block_size`；
   - 分配 K/V pool（各 `L * max_num_blocks * block_size * kv_dim` 个 half）与 scratch（各 `max_visible_tokens * kv_dim`），`d_block_tables = DeviceBuffer<int>(max_num_blocks)`；
   - **跳过 KVCacheManager 创建**（`h->kv_cache` 保持 nullptr）。
   - `max_num_blocks == 0` 时走现有策略 2 路径，行为完全不变。
3. `tinyllm_allocate_sequence`：
   - paged 模式：只校验参数并写入 `h->sequences[seq_id] = {allocated=true}`，**不调用 kv_cache**。
   - 连续模式：现有逻辑不变。
4. `tinyllm_free_sequence`：
   - paged 模式：`h->sequences.erase(seq_id)`，不调用 kv_cache。
   - 连续模式：现有逻辑不变。
5. `tinyllm_free`：paged 指针非空则 `cudaFree`；`d_block_tables` 是 RAII。

**验收**：
```bash
cmake --build build -j$(nproc)
./build/tiny_llm_tests   # 全绿（策略 2 路径不回归）
TLLM_FFI_DEBUG=1 TLLM_GGUF_TEST_MODEL=$MODEL ./build/tiny_llm_tests --gtest_filter='*FFI*'
# 期望：策略 2 FFI 测试仍通过
```

**提交**：`feat(ffi): paged KV pool allocation and handle lifetime`

---

## 任务 D2c：TransformerLayer 分页前向路径（tiny-llm）

**改动文件**：`include/tiny_llm/transformer.h`、`src/transformer.cpp`。

**步骤**：
1. 在 `transformer.h` 中新增公开视图结构（放在 `LayerWorkspace` 之后）：
   ```cpp
   struct PagedKVCacheView {
       half *k_pool = nullptr;        // [L * max_num_blocks * block_size * kv_dim]
       half *v_pool = nullptr;
       const int *block_table = nullptr; // device int[visible_blocks]
       half *k_scratch = nullptr;     // [max_visible_tokens * kv_dim]
       half *v_scratch = nullptr;
       int visible_blocks = 0;
       int block_size = 0;
       int max_num_blocks = 0;
       int max_visible_tokens = 0;
       int position = 0;              // 本步首 token 的绝对位置
       const int *decode_len = nullptr; // decode 专用 device int；prefill 传 nullptr
   };
   ```
2. 新增公开方法：
   ```cpp
   Result<void> forwardPaged(half *hidden_states, const PagedKVCacheView &kv,
                             int num_tokens, const int *rope_pos, const float *rope_cos,
                             const float *rope_sin, cudaStream_t stream = 0);
   ```
   与 `runLayer` 的差异只在 attention：调用新的私有 `attentionPaged`。
3. `attentionPaged` 实现（在 `src/transformer.cpp`，复用 `feedForward` / `rmsNorm` / w8a16 transposed 快路径）：
   - `kv_dim = num_kv_heads * head_dim`；本层 pool 偏移 `layer_idx_ * max_num_blocks * block_size * kv_dim`；
   - Q/K/V 投影、bias、RoPE 与现有 `attention()` 完全一致（含 `data_t/scales_t` 快路径参数）；
   - `paged_scatter_blocks(k_buf, k_pool_layer, kv.block_table, num_tokens, kv.position, block_size, kv_dim, stream)`；V 同理；
   - 可见长度：prefill `visible = num_tokens`；decode（`num_tokens==1 && kv.decode_len`）`visible = kv.position + 1`；
   - `paged_gather_blocks(k_scratch, k_pool_layer, kv.block_table, visible, ...)`；V 同理；
   - prefill 走 `attention_prefill(q_buf, k_scratch, v_scratch, attn_buf, ...)`；decode 走 `attention_decode(..., kv.decode_len, ...)`；
   - 最后 `w8a16_matmul(attn_buf, wo, ..., output, ...)` 与现有路径一致。
4. 输入校验：`kv.block_table/k_pool/v_pool/scratch` 非空、`visible_blocks>0`、`block_size>0`、`position+num_tokens <= max_visible_tokens`。

**验收**：
```bash
cmake --build build -j$(nproc)
./build/tiny_llm_tests   # 全绿（现有路径不回归）
```

**提交**：`feat(transformer): paged KV forward path (prefill + decode)`

---

## 任务 D2d：FFI 步进接入策略 1（tiny-llm）

**改动文件**：`src/ffi.cpp`。

**步骤**（在 `tinyllm_step` 的序列循环里，按 `h->paged_kv` 分支）：

1. 前置校验：
   - paged 模式要求 `block_tables != nullptr && num_blocks != nullptr`；
   - `num_blocks[s] > 0` 且 `num_blocks[s] <= h->max_num_blocks`；
   - prefill：`num_blocks[s] == ceil(len / block_size)`（允许大于，只要求能容纳 `len`）；
   - decode：`num_blocks[s] >= ceil((st.position + 1) / block_size)`。
2. 块表上传：维护 `table_offset`（前面序列 num_blocks 之和）；每次
   `h->d_block_tables.copyFromHost(block_tables + table_offset, num_blocks[s], h->stream);` 后 `table_offset += num_blocks[s];`。
3. 构造视图：
   ```cpp
   tiny_llm::PagedKVCacheView view;
   view.k_pool = h->paged_k_pool; view.v_pool = h->paged_v_pool;
   view.block_table = h->d_block_tables.data();
   view.k_scratch = h->paged_k_scratch; view.v_scratch = h->paged_v_scratch;
   view.visible_blocks = num_blocks[s];
   view.block_size = /* 从 handle 保存 */;
   view.max_num_blocks = h->max_num_blocks;
   view.max_visible_tokens = h->max_visible_tokens;
   ```
4. **prefill 分支**（paged）：
   - `view.position = 0; view.decode_len = nullptr;`
   - `h->rope_pos.copyFromHost(&zero, 1, h->stream);`
   - `embed(h, toks, len, h->hidden_buf);`
   - 逐层 `layer->forwardPaged(h->hidden_buf, view, len, h->rope_pos.data(), h->rope_cos, h->rope_sin, h->stream)`，失败返回 `TLLM_ERR`；
   - `next_tokens[s] = sample_from_hidden(h, h->hidden_buf + (len-1)*hidden); st.position = len;`
5. **decode 分支**（paged）：
   - `pos = st.position; visible = pos + 1;`
   - `h->decode_len.copyFromHost(&visible, 1, h->stream); h->rope_pos.copyFromHost(&pos, 1, h->stream);`
   - `embed(h, toks, 1, h->hidden_buf + pos*hidden);`
   - `view.position = pos; view.decode_len = h->decode_len.data();`
   - 逐层 `layer->forwardPaged(token_state, view, 1, ...)`；
   - `next_tokens[s] = sample_from_hidden(h, token_state); st.position = pos + 1;`
6. 保留策略 2 分支原样；`(void)num_blocks` 只在策略 2 分支生效。
7. `tinyllm_load` 保存 `block_size` 到 handle（paged 模式用）。

**验收**：
```bash
cmake --build build -j$(nproc)
./build/tiny_llm_tests
# 策略 2 门控测试仍通过：
TLLM_GGUF_TEST_MODEL=$MODEL ./build/tiny_llm_tests --gtest_filter='*FFI*'
```

**提交**：`feat(ffi): strategy-1 paged KV step path`

---

## 任务 D2e：策略 1 的 FFI 门控差分测试（tiny-llm）

**改动文件**：`tests/test_ffi.cpp`。

**步骤**：新增 `FFITest.PagedKVStrategyMatchesContiguous`（门控 `TLLM_GGUF_TEST_MODEL`）：

1. 连续模式基线：
   - `TinyLlmConfig cfg2 = {0,0,0,0,0,0,16,1,0};`（max_num_blocks=0）
   - load → allocate → prefill 6-token prompt（沿用现有测试的 token/positions fixture）→ decode 4 步，收集 `next` 序列 `seq_contig`。
2. 分页模式：
   - `TinyLlmConfig cfg1 = {0,0,0,0,0,0,16,1,8};`（block_size=16, max_num_blocks=8）
   - 手工构造块表：prefill `int blocks[] = {0,1}; int nb=1?` 注意 prompt=6 < block_size=16，所以 `num_blocks=1, block_table={0}`；
   - 调用新签名 `tinyllm_step(..., block_table, &nb, pre, ...)`；
   - decode 各步同样 `nb=1, block_table={0}`（position ≤15 之前）。
   - 收集 `seq_paged`。
3. `ASSERT_EQ(seq_contig, seq_paged)` 逐 token 完全一致（两个模式共用同一模型句柄需分开加载，不能同时用一个句柄）。
4. 另加一个跨块用例：decode 到 position≥16 时 `nb=2, block_table={0,1}`，验证与连续模式一致。

**验收**：
```bash
cmake --build build -j$(nproc)
TLLM_GGUF_TEST_MODEL=$MODEL ./build/tiny_llm_tests --gtest_filter='*PagedKV*:*FFI*'
# 期望：PagedKV 测试通过（无模型时按现有方式 skip）
```

**提交**：`test(ffi): paged strategy differential vs contiguous strategy`

---

## 任务 D3：paged-infer TinyLlmExecutor 接入真实块表

**改动文件**：`src/tiny_llm_executor.rs`（核心）、`src/tiny_llm_ffi.rs`（如需要）。

**步骤**：
1. `TinyLlmExecutor::new`：
   - `TinyLlmConfig.max_num_blocks` 改为 `config.max_num_blocks as i32`（默认 1024，即默认启用策略 1）；
   - 若希望保留显式 fallback，读取环境变量 `PAGED_INFER_TINY_LLM_STRATEGY=2` 时强制 `max_num_blocks: 0`（连续模式）。**本任务实现该开关并在模块文档注释写明。**
2. `execute()` 构造：
   ```rust
   let mut block_tables_flat: Vec<c_int> = Vec::new();
   let mut num_blocks: Vec<c_int> = Vec::with_capacity(n);
   for bt in &batch.block_tables {
       if bt.is_empty() { return Err(EngineError::BackendError("empty block table".into())); }
       num_blocks.push(bt.len() as c_int);
       block_tables_flat.extend(bt.iter().map(|&b| b as c_int));
   }
   ```
3. `symbols::tinyllm_step` 调用传入 `num_blocks.as_ptr()` 与 `block_tables_flat.as_ptr()`；连续模式 fallback 时两个都传 null（并把 `num_blocks` 置空）。
4. 更新模块顶部注释：策略 1 为默认，策略 2 为 `PAGED_INFER_TINY_LLM_STRATEGY=2` fallback。
5. 能力声明/错误传播不变。

**验收**：
```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
# 期望：默认测试全绿（tiny-llm feature 未启用时只验证编译）
TINY_LLM_DIR=/home/shane/github/aicl/tiny-llm TINY_LLM_MODEL=$MODEL \
  PINF_TOKENIZER_JSON=$TOK cargo test --features tiny-llm --test tiny_llm_backend -- --nocapture
# 期望：真实后端接入测试通过（现在走策略 1 块表）
```

**提交**：`feat(executor): strategy-1 paged KV via real block tables`

---

## 任务 D4：3 并发端到端 + llama.cpp 对齐 + 资源守恒

**改动文件**：`tests/tiny_llm_text_e2e.rs`（扩展）、可能 `tests/concurrency_stress.rs`（复用）。

**步骤**：
1. 现有 `qwen2_chat_prompt_matches_llama_cpp` 测试在 D3 后自动走策略 1，先跑通它作为单请求 paged 基线：
   ```bash
   TINY_LLM_DIR=/home/shane/github/aicl/tiny-llm TINY_LLM_MODEL=$MODEL \
     PINF_TOKENIZER_JSON=$TOK cargo test --features tiny-llm \
     --test tiny_llm_text_e2e qwen2_chat_prompt_matches_llama_cpp -- --nocapture
   ```
   期望：token 序列仍等于既有的 llama.cpp 参考数组（24 token + EOS）。
2. 新增测试 `qwen2_three_concurrent_paged_requests_match_llama_cpp`：
   - `EngineConfig::default()`，但 `max_num_blocks=256, block_size=16, max_model_len=256, max_num_seqs=4, max_batch_size=4`；
   - 提交 3 个请求，prompt 分别为：
     1. `<|im_start|>user\nHello, how are you?<|im_end|>\n<|im_start|>assistant\n`（复用既有 reference）
     2. `<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n`（在 D4 执行时先用 llama.cpp 或现 fixture 记录 greedy 序列，把序列写死在测试里；若无新 fixture，先用 prompt 1 的两个变体 + 一个短 prompt 的预期输出，**禁止伪造**，拿不到参考就把该请求改为只断言成功/非空/EOS 终止）
     3. 短 prompt `"Hello"`（只断言 success、非空、finish_reason 为 Stop 或 Length）
   - `engine.run()` 后断言 3 个请求全部成功；
   - 请求 1 的输出 token 严格等于既有 reference；
   - 运行结束后 `engine` 的 KV 利用率回到基线（复用现有资源守恒断言方式：完成请求后 `used + free == total`，`active_sequences == 0`）。
3. 复用 `tests/concurrency_stress.rs`（Mock/CPU 场景）确保调度回归。

**验收**：
```bash
TINY_LLM_DIR=... TINY_LLM_MODEL=$MODEL PINF_TOKENIZER_JSON=$TOK \
  cargo test --features tiny-llm --test tiny_llm_text_e2e -- --nocapture
TINY_LLM_DIR=... TINY_LLM_MODEL=$MODEL PINF_TOKENIZER_JSON=$TOK \
  cargo test --features tiny-llm --test concurrency_stress -- --nocapture
cargo test
```

**提交**：`test(e2e): 3-way paged concurrency with llama.cpp token parity`

---

## 任务 D5：文档收口 + 推送 + 打 tag

**改动文件**：
- tiny-llm：`include/tiny_llm/ffi.h` 头注释（策略 1 已启用）、`README.md` 状态表"分页 KV"✅、`DEVELOPMENT_PLAN.md` 阶段 4.4 标注完成；
- paged-infer：`README.md` 路线图 T11 打勾、`DEVELOPMENT_PLAN.md` T11 状态、`src/tiny_llm_executor.rs` 顶部注释与代码一致；
- cuda-foundations：`LEARNING_PATH.md` 阶段 5 完成证据把"分页 KV 暂未启用"改为已启用并引用 `tests/tiny_llm_text_e2e.rs::qwen2_three_concurrent_paged_requests_match_llama_cpp`；
- 根 `PHASE2_PLAN.md` / 本文件状态同步。

**推送与 tag**：
```bash
# tiny-llm
cd /home/shane/github/aicl/tiny-llm && git push origin master && git tag phase-2-d && git push origin phase-2-d
# paged-infer
cd /home/shane/github/aicl/paged-infer && git push origin master && git tag phase-2-d && git push origin phase-2-d
# cuda-foundations（LEARNING_PATH 改动）
cd /home/shane/github/aicl/cuda-foundations && git push origin master && git tag phase-2-d && git push origin phase-2-d
```

**验收**：
- 三仓 ahead 0；tags 在远端可见；
- `grep -n "忽略 block_tables\|策略 2：连续 KV"` 只允许命中 `TLLM_FFI_STRATEGY=2` fallback 说明与 CHANGELOG/历史文档，不允许命中当前 README 的状态表。

---

## 本批完成定义（D 阶段 DoD）

- [x] tiny-llm 与 paged-infer 的 C ABI 均为 v2（9 int config + num_blocks 参数），两侧布局守卫测试通过。
- [x] scatter/gather kernel 单测通过；`TLLM_GGUF_TEST_MODEL` 下策略 1 与策略 2 逐 token 一致。
- [x] paged-infer 默认策略 1；`PAGED_INFER_TINY_LLM_STRATEGY=2` 可回退策略 2。
- [x] 3 并发真实模型 e2e：请求 1 与 llama.cpp 参考逐 token 一致；资源守恒不变量成立。
- [x] 三仓推送、打 `phase-2-d` tag；文档与代码事实一致。

---

## 本批完成后的下一步

汇报以下内容后停下：
1. 每仓 `git status -sb` 与 commit hash 列表；
2. D2e 差分测试结果与 D4 3 并发输出（请求 1 token 序列必须贴出来）；
3. 峰值显存变化（策略 1 应低于策略 2，记录实测）；
4. 遗留 NOTE。

随后进入 `PHASE2_PLAN.md` 第 8 节 **E 阶段（作品集收尾）**：E1 Triton SGEMM + torch.library、E2 cuflash 优化轮、E3 landing README、E4 release tag/badge。
