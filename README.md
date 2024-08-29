# Triton Flash Attention

This code is based on the [Triton attention tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py).

Could be useful if you are looking for
- Triton version of flash attention
- Flash attention w/ trainable bias feature

**Note** : bias of O(MN) should be materialized

Things that could be added in the future...
- [ ] FP8 precision support
- [ ] Hopper-specialized implementation (e.g., Flash Attention 3)

**Welcome the community's contributions!**

```python
flash_attention_fn(q, k, v, bias=None, causal=False, sm_scale=None):
"""
B : batch size
M : # of query tokens
N : # of kv tokens
H : # of heads
D : head dimension

q       : (B, M, H, D)
k, v    : (B, N, H, D)
bias    : (B, H, M, N)

if bias.dtype == bool :
    ignore attention scores for False
if bias.dtype == float:
    added to QK before softmax (i.e., softmax(QK + bias))

Return:
    out : (B, M, H, D)
"""
```
```python
flash_attention_qkvpacked_fn(qkv, bias=None, causal=False, sm_scale=None):
"""
B : batch size
H : # of heads
D : head dimension

qkv     : (B, seqlen, 3, H, D)
bias    : (B, H, seqlen, seqlen)

if bias.dtype == bool :
    ignore attention scores for False
if bias.dtype == float:
    added to QK before softmax (i.e., softmax(QK + bias))

Return:
    out : (B, M, H, D)
"""
```
