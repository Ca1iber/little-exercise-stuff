# 简化 MLA / Simplified MLA

基于 PyTorch 的 MLA 手撕练习，重点理解 KV 联合低秩压缩与因果多头注意力。

A small PyTorch implementation for learning joint low-rank KV compression and causal multi-head attention.

## 接口 / Interface

- 实现 / Implementation: [mla.py](mla.py), `MultiLatentAttention(D, low_rank)`
- 输入与输出 / Input and output: `[B, T, D]`
- 固定 8 个注意力头，`D` 必须能被 8 整除。 / Uses 8 attention heads; `D` must be divisible by 8.
- `low_rank` 为共享 KV 表示的维度；用于压缩时取小于 `D` 的值。 / `low_rank` is the shared KV latent dimension; choose a value smaller than `D` for compression.

## 前向流程 / Forward Pass

1. 一次线性投影生成 Q 和共享低维表示 c。  
   A single linear projection produces Q and a shared low-dimensional representation c.
2. 对 c 做 RMSNorm，再通过两套独立的升维权重生成 K、V。  
   Apply RMSNorm to c, then use separate up-projections to produce K and V.
3. 拆分多头，计算缩放点积注意力，在 Softmax 前屏蔽未来位置。  
   Split into heads and compute scaled dot-product attention, masking future positions before Softmax.
4. 对 V 加权求和，合并多头，再经过输出投影。  
   Aggregate V with attention weights, merge the heads, and apply the output projection.

## 练习范围 / Scope

缓存存取方法是有意保留的流程占位，用于标记归一化后压缩表示的存取阶段。本练习执行整段序列前向。

The cache methods are intentional placeholders marking where normalized latent representations would be stored and retrieved. This exercise runs full-sequence forward passes.

这是聚焦 KV 联合压缩的简化版本：Q 使用直接投影，未加入位置编码。

This simplified version focuses on joint KV compression: Q uses a direct projection, and positional encoding is omitted.
