# Gemini Research Prompt: vLLM KV Cache Page Size Padding for TurboQuant

**Purpose:** Get Gemini's analysis on edge cases for using `page_size_padded` on `AttentionSpec` in vLLM v1 to handle non-uniform page sizes from TurboQuant KV compression with variable head_dim models.

**Copy the prompt below into Gemini CLI or Gemini web:**

---

I'm working on a vLLM fork called vllm-turboquant that adds TurboQuant KV cache compression (from Google's TurboQuant paper, arxiv.org/abs/2504.19874). I need your help analyzing edge cases for a specific fix.

## The Problem

Google's Gemma 4 model has mixed attention head dimensions:
- 28 sliding attention layers: head_dim=256
- 7 full attention layers: head_dim=512

TurboQuant KV compression computes a `packed_dim` (bytes per KV entry per head) that depends on head_dim:
- head_dim=256 → packed_dim=120 → page_size=30,720 bytes
- head_dim=512 → packed_dim=232 → page_size=59,392 bytes

vLLM v1's `unify_kv_cache_spec_page_size()` in `kv_cache_utils.py` requires `max_page_size % min_page_size == 0` to unify block allocation. Since 59,392 % 30,720 = 28,672 (not zero), it raises `NotImplementedError`.

## My Proposed Fix (Two Files)

### File 1: kv_cache_utils.py
Replace the `NotImplementedError` with `page_size_padded=max_page_size`:
```python
# When max_page_size % layer_page_size != 0:
new_spec = replace(layer_spec, page_size_padded=max_page_size)
```

This uses the existing `page_size_padded` field on `AttentionSpec` (kv_cache_interface.py:86) which separates allocation size from data size.

### File 2: gpu_model_runner.py
In `_reshape_kv_cache_tensors`, slice padded tensors before reshaping:
```python
real_page_size = kv_cache_spec.real_page_size_bytes
if (kv_cache_spec.page_size_padded is not None
        and kv_cache_spec.page_size_padded > real_page_size):
    raw_tensor = (raw_tensor
        .view(num_blocks, kv_cache_spec.page_size_bytes)
        [:, :real_page_size]
        .contiguous()
        .view(-1))
```

This is needed because `.view(uint8).view(kv_cache_shape)` fails when the padded tensor (59,392 bytes per block) doesn't match the shape product (30,720 bytes for packed_dim=120).

## Key Context

- TurboQuant KV maps to `torch.uint8` dtype (not bfloat16)
- `packed_dim` values are in bytes
- The `page_size_padded` field is already used in production for Mamba-to-attention padding in hybrid models, but NOT for attention-to-attention padding
- Mamba uses `torch.as_strided` for reshape (different path); attention uses `.view(dtype).view(shape)`

## Questions for Gemini

1. **Reshape safety:** Is slicing `raw_tensor.view(num_blocks, page_size)[:, :real_page_size].contiguous()` safe for GPU memory? Does `.contiguous()` allocate a new tensor (doubling memory) or can we use `as_strided` like Mamba does to avoid the copy?

2. **Prefix caching:** vLLM v1 uses block hashing for prefix caching. If two layers share the same block but one has padding bytes, could the hash include padding bytes and produce incorrect cache hits? Specifically, does `hash_block` operate on the raw block bytes or on the reshaped tensor?

3. **Block table consistency:** The KVCacheManager's block tables track block indices. With padding, the 256-head layers and 512-head layers use the same block indices but different amounts of data per block. Are block tables per-group or shared? If shared, does the scheduler correctly handle groups with different real data sizes per block?

4. **`torch.as_strided` alternative:** Instead of slice+contiguous, could we use `torch.as_strided` (like the Mamba path) to create a view into the padded tensor without copying? Something like:
```python
torch.as_strided(
    raw_tensor.view(torch.uint8),
    size=(num_blocks, 2, block_size, num_kv_heads, packed_dim),
    stride=(page_size_padded, block_size*num_kv_heads*packed_dim, num_kv_heads*packed_dim, packed_dim, 1),
)
```
Would this be correct and avoid the memory copy?

5. **Any other edge cases** in vLLM v1's KV cache pipeline (scheduler, prefix caching, speculative decoding, pipeline parallelism) that would break with padded attention page sizes?

---

**After getting the response, paste it back into the Claude Code session so I can incorporate the findings into the implementation plan.**
