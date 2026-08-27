# Serving 岗位角色手册

> Phase I6。目标岗位偏 **serving / 推理基础设施 / 调度** 时的讲述顺序与重点。
> 同一份材料重排：主推 paged-serving（调度/资源不变量/3 并发）+ tiny-llm 作为后端；kernel 只讲“为什么控制面不需要写 kernel”。

## 一句话定位

“我是做 serving 控制面的：调度器锁资源不变量，分页 KV 走 C ABI 接真实后端，3 并发是我验证正确性的 fixture，不是生产并发。”

## 20 分钟讲述时间轴

| 分钟 | 讲什么 | 用到的 QA_BANK / 数字 |
|------|--------|------------------------|
| 0–2 | 定位：“控制面不写 kernel，只定义边界与不变量” + 接入层差异 | Q31/Q47 |
| 2–6 | PagedAttention 数据面：碎片、block table、策略 1 数据流 | Q47/Q48/Q57；分页 vs 连续差分 E20 |
| 6–11 | 调度核心：continuous batching、状态机、三层准入、内存水位线+decode reserve、HOL、优先级 | Q49/Q50/Q51/Q52/Q53/Q54 |
| 11–14 | 边界：为什么不做抢占、swap/recompute 做到哪 | Q55/Q56 |
| 14–17 | 验证：3 并发证明了什么、SSE token 级流式边界、Rust capabilities | Q58/Q59/Q60；218 tests、`active_sequences==0` |
| 17–19 | tiny-llm 作为后端：一句带过“转置/Graphs 是 runtime 手册的深挖，我只报 ABI 与正确性” | 引用 runtime 手册（Q33–Q46），陈述边界 |
| 19–20 | 反问 + 边界声明：“无 GPU 也能测调度；默认 cargo test 没开 e2e” | 数字卡 §6 / Q58 |

## 推荐的 QA_BANK 题号子集（15 题）

`Q31, Q47 – Q60`（1 个接入层差异题 + serving 全套 Q47–Q60）。理由：serving 岗把控制面 14 题讲透，另用 Q31 说明为什么接入层差异决定了我不写 kernel。

## 简历条目重排顺序

1. `paged-serving › 1`：C ABI v2 九整型布局、Rust `sizeof==36` 守卫（E19）——先证明跨语言边界严谨。
2. `paged-serving › 2`：属性测试锁定 `used+free==total`（E23）——证明不变量文化。
3. `paged-serving › 3`：OpenAI 兼容 API/SSE/`paged_*` metrics 集成 37 项（E25）——证明对外契约可测。
4. `总览 › 3`：3 并发分页请求与 llama.cpp greedy 对齐、量化分歧如实记录（E21）——证明后端对接正确性。
5. `总览 › 1`：四层五仓可验证学习链（E30）——证明我知道全局边界。

## 反问问题清单（≥3）

1. 你们生产调度器除了“先 decode 后 prefill”，有没有抢占/swap？我这边明确不做抢占，想知道生产做法。
2. 内存水位线里的 decode reserve 怎么标定？我这边是经验保护，你们有 ncu 或压测依据吗？
3. 控制面和 kernel 的 ABI 谁拥有？我用 `repr(C)` 布局守卫防漂移，你们怎么防？
4. 如果做 vLLM 这类任务，good-first-issue 更看重调度不变量还是 custom op 接入？我可以两边都接。
