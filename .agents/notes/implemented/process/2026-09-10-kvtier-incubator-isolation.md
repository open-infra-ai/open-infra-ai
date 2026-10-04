# Agent Note: kvtier 孵化仓隔离

Status: implemented

## Problem

`kvtier` 是 KV 分层存储的本地研究孵化器：尚无完成审计的真实 GPU 结果包，
上游贡献仍在进行。若把它列入公开状态注册表，等于对外声明了尚未兑现的能力，
破坏作品集的证据纪律。

## Decision

`kvtier` 不进入公开状态注册表，不适用
[链接唯一写法与命名冻结](2026-08-21-canonical-org-identity.md) 两条规则；
工作区地图中标注为私有孵化器。达到其 README 公开门槛（真实 GPU 实验 +
可复现固定 + 至少一个外部可验证的上游结果 + 清理个人痕迹）后，另立变更记录
再转公开。

## Alternatives considered

- **现在就公开并标 `active`** — 看起来更透明；但没有已审计结果就注册状态，
  正是本作品集反对的「先声明后补证据」。
- **永久本地、不规划公开** — 最省事；但上游贡献（SGLang HiCache）本就是
  战略主线，公开是目标而非意外。

## Consequences

- **收益**：公开面每个 `active`/`stable` 背后都有证据包；孵化期改名、
  链接写法不受冻结约束（已实际改名三次）。
- **代价**：本地与公开面存在不对称，工作区地图须显式标注防止误引用。

## Verification

根 `changelog/2026-09-10-workspace-normalization.md` 记录规则 6 落地；
kvtier README「公开门槛」一节给出判据。
