# cuda-foundations · 10 分钟讲述稿

数字见 [`../NUMBERS_CARD.md`](../NUMBERS_CARD.md) §6。教学品牌名 CUDA Kernel Academy；GitHub 名 `cuda-foundations`。

## 0. 一句话定位

从 naive SGEMM 到可测量 CUDA 组件的教学仓。

## 1. 2 分钟：做什么、边界、为什么这样切

四模块：01 GEMM 阶梯、02 可复用算子库形态、03 HPC 实验、04 把 kernel/内存/流拼进小系统。04 是 **tiny-llm 的教学预览**，不是第二套运行时（`04-inference-engine/README.md`）。

OUT：真实 GGUF、生产 FA、调度。那些分别在 tiny-llm / cuflash / paged-serving。

为什么不和 Triton 合仓：构建系统（CMake+CUDA vs pip）和面试展示都要求「同题两套 repo 并排打开」。`LEARNING_PATH.md` 是五仓导航入口。

状态：`phase-2-e` 冻结。本次 freeze：`ctest --preset default` **0 failed / 209 collected，78 skipped**（02/04 GPU 二进制为主）。不要说 209 全绿全跑过。

## 2. 3 分钟：最难的实现细节

**细节：阶梯上每一步都要有「为什么」，并且允许某一步变慢。**

1024³ FP32 实测（`docs/en/benchmarks/index.md`）：

| Kernel | TFLOPS | vs cuBLAS |
|--------|--------|-----------|
| Naive | 0.58 | 10.4% |
| Tiled | 0.92 | 16.6% |
| Bank Conflict Free | 0.66 | 11.8% |
| Double Buffer | 0.68 | 12.2% |
| Tensor Core WMMA | 1.09 | 19.6% |
| cuBLAS | 5.58 | 100% |

要讲的不是「我追平了 cuBLAS」，而是：

- Naive → Tiled：共享内存复用，减少 HBM。
- 「Bank conflict free」在本机 **慢于 tiled**（0.66 vs 0.92）。padding 换冲突，也可能换占用和指令。教学仓必须把这个数字留下，否则学习者会以为 padding 总是赢。
- Double buffer 同样没有超过 tiled。没有 `cp.async` 的「软件流水」只是同一线程先 load 再算，叙事不能写成 TMA/warp-specialization。
- WMMA 1.09 仍远低于 cuBLAS 5.58：对齐、epilogue、算法选择都还没做。

这就是本仓交付的能力：**用测量否定自己的直觉**。

## 3. 2 分钟：优化/调试故事

故事就是 GEMM 阶梯本身。

- Before：naive 0.58 TFLOPS。
- 证据：同一尺寸、同一硬件 RTX 3060 Laptop 的阶梯表；profiling 在 WSL2 上 `ncu` 报 `ERR_NVGPUCTRPERM`，runbook 在 `docs/en/guides/profiling.md`。
- After：WMMA 1.09。相对 naive 约 1.9×，相对 cuBLAS 仍约 0.20×。
- 代价：教学实现，不是库。02/04 在本冻结机大量 skip，不能用 CI 绿灯替代 GPU 数值。

## 4. 2 分钟：验证方法

- 01 模块逐步与参考/cuBLAS 比（容差写在测试里）。
- 根 `ctest --preset default`：0 failed；skip 必须开口说。
- 文档站只保留本机测量页；占位 TFLOPS 已清。
- 负结果：bank-conflict-free / double-buffer 未超过 tiled，留在表里。

## 5. 1 分钟：短板与下一步

短板：不是 runtime；ncu 计数器没有；04 不能当作品讲。下一步在冻结清单外：真机 ncu 把 bank conflict 那步讲圆，或把 skip 的 02/04 环境修好。默认不做新模块。

## 6. 追问清单

1. 为什么 tiled 比 naive 快？ → 共享内存 tile，复用 K 维。
2. 为什么你们的 bank-conflict-free 更慢？ → 本机 0.66 vs 0.92；padding 有代价。
3. double buffering 是异步拷贝吗？ → 本教程不是 `cp.async`，不要吹。
4. WMMA 为什么远低于 cuBLAS？ → 算法/对齐/epilogue/启发式都缺。
5. 04 和 tiny-llm 什么关系？ → 教学预览 vs 真实 GGUF 运行时。
6. 209 tests 全过？ → 0 failed / 78 skip，不能混为一谈。
7. 为什么改名 cuda-foundations？ → 仓库名与「Academy 课程品牌」分离；五仓源码旧 slug 0 命中。
8. 没有 ncu 怎么证明瓶颈？ → runbook + 阶梯相对值；承认没有计数器。
9. 会不会把 Triton 合进来？ → 不做；同题异构要两个可打开的仓。
10. 面试只看这一个仓够吗？ → 不够，这是 L1；旗舰是 tiny-llm。
