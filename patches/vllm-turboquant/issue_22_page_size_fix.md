# TurboQuant KV Page Size Fix (Issue #22) — Applied Patches

**Date:** 2026-04-16
**Scope:** Enable TurboQuant KV cache (`kv_cache_dtype='turboquant35'`) on
models with mixed attention head dimensions (Gemma 4 E2B: head_dim=256
sliding + head_dim=512 full).

**Verification:** Gemma 4 E2B + BNB_INT4 + CPU offload 9.9 GB +
turboquant35 KV loads and runs end-to-end on RTX A2000 4 GB (WSL2). Qwen3
AWQ + turboquant35 regression passes.

**Coverage:** 28/35 Gemma 4 layers compressed via TurboQuant (sliding
window, head_dim=256). The 7 full-attention layers (head_dim=512) fall
back to bf16 — no calibration metadata exists for that head size.

---

## Patch 1 — `vllm/v1/core/kv_cache_utils.py`

Allow `page_size_padded` fallback when page sizes are not evenly
divisible (variable head_dim models).

```python
def unify_kv_cache_spec_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> dict[str, KVCacheSpec]:
    from vllm.v1.kv_cache_interface import AttentionSpec
    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        return kv_cache_spec

    max_page_size = max(page_sizes)
    new_kv_cache_spec = {}
    for layer_name, layer_spec in kv_cache_spec.items():
        if layer_spec.page_size_bytes == max_page_size:
            new_kv_cache_spec[layer_name] = layer_spec
        else:
            layer_page_size = layer_spec.page_size_bytes
            if max_page_size % layer_page_size == 0:
                ratio = max_page_size // layer_page_size
                new_block_size = layer_spec.block_size * ratio
                new_spec = replace(layer_spec, block_size=new_block_size)
            elif isinstance(layer_spec, AttentionSpec):
                new_spec = replace(layer_spec, page_size_padded=max_page_size)
            else:
                raise NotImplementedError(
                    "The page size of the layer is not divisible by the "
                    "maximum page size. Cannot unify by adjusting block_size."
                )
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
    return new_kv_cache_spec
```

---

## Patch 2 — `vllm/v1/worker/gpu_model_runner.py::_reshape_kv_cache_tensors`

(a) Honor the spec's `cache_dtype_str` (per-layer) over the global
`cache_config.cache_dtype` when computing `kv_cache_shape`.

(b) For padded layers, use `torch.as_strided` to create a zero-copy view
whose block stride jumps over padding bytes while the shape covers only
the real data.

```python
spec_cache_dtype = getattr(kv_cache_spec, "cache_dtype_str", None)
kv_cache_shape = attn_backend.get_kv_cache_shape(
    kernel_num_blocks,
    kernel_block_size,
    kv_cache_spec.num_kv_heads,
    kv_cache_spec.head_size,
    cache_dtype_str=(
        spec_cache_dtype
        if spec_cache_dtype is not None
        else self.cache_config.cache_dtype
    ),
)
# ... compute stride_order, inv_order ...

real_page_size = kv_cache_spec.real_page_size_bytes
padded_page_size = kv_cache_spec.page_size_bytes
is_padded = padded_page_size > real_page_size

if is_padded:
    dtype_size = get_dtype_size(dtype)
    assert padded_page_size % dtype_size == 0
    viewed = kv_cache_raw_tensors[layer_name].view(dtype)
    element_strides = [1]
    for dim_size in reversed(kv_cache_shape[1:]):
        element_strides.insert(0, element_strides[0] * dim_size)
    padded_block_stride = padded_page_size // dtype_size
    all_strides = (padded_block_stride, *element_strides[1:])
    kv_cache = torch.as_strided(viewed, size=kv_cache_shape, stride=all_strides)
    kv_caches[layer_name] = kv_cache.permute(*inv_order)
else:
    kv_caches[layer_name] = (
        kv_cache_raw_tensors[layer_name]
        .view(dtype)
        .view(kv_cache_shape)
        .permute(*inv_order)
    )
```

---

## Patch 3 — `vllm/v1/attention/ops/triton_prefill_attention.py::get_block_size`

Drop prefill BLOCK from 128 → 64 for `head_dim > 128` on GPUs with
<160 KB opt-in shared memory (RTX A2000 SM_86 = 99 KB, Blackwell SM_121 =
99 KB). A100 SM_80 = 160 KB keeps BLOCK=128.

```python
def get_block_size(dtype, head_dim=None, device=None) -> int:
    if dtype == torch.float32:
        return 32
    if (head_dim is not None and head_dim > 128
        and device is not None and device.type == "cuda"):
        props = torch.cuda.get_device_properties(device)
        max_shared = getattr(
            props, "shared_memory_per_block_optin",
            props.shared_memory_per_block,
        )
        if max_shared < 163840:
            return 64
    if current_platform.is_cuda_alike() and current_platform.has_device_capability(80):
        return 128
    return 64
```

---

## Patch 4 — `vllm/model_executor/layers/attention/attention.py`

(a) After creating the `self.impl`, propagate any TurboQuant
graceful-skip (`impl.kv_cache_dtype` changed to `"auto"` on head_size
mismatch) back to the parent `Attention` layer.

(b) In `get_kv_cache_spec`, build the spec from `self.kv_cache_dtype`
(layer-level, possibly overridden) instead of the global
`vllm_config.cache_config.cache_dtype`, so bf16-downgraded layers
allocate bf16 storage.

```python
# After self.impl = impl_cls(...)
impl_kv_cache_dtype = getattr(self.impl, "kv_cache_dtype", None)
if impl_kv_cache_dtype is not None and impl_kv_cache_dtype != kv_cache_dtype:
    self.kv_cache_dtype = impl_kv_cache_dtype
    self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
        impl_kv_cache_dtype, vllm_config.model_config
    )

# In get_kv_cache_spec:
cache_dtype_str = self.kv_cache_dtype
if cache_dtype_str is None:
    cache_dtype_str = vllm_config.cache_config.cache_dtype
# ... use cache_dtype_str in FullAttentionSpec / SlidingWindowSpec
```
