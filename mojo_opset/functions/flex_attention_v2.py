"""FlexAttention v2 interface with explicit grouped-query attention support."""

from .flex_attention import FlexAttentionFunction


def flex_attention_v2(
    q,
    k,
    v,
    score_mod=None,
    block_mask=None,
    scale=None,
    enable_gqa=False,
    return_lse=False,
    *,
    implementation=None,
    **kwargs,
):
    """Torch-compatible call shape with explicit enable_gqa; score_mod/LSE are unsupported."""
    return FlexAttentionFunction.apply(
        q, k, v, block_mask, scale, implementation, "flex_attention_v2", enable_gqa, score_mod, return_lse
    )
