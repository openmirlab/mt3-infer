"""Adapt the frozen YourMT3 T5 cache contract to Transformers 5.

The bundled YourMT3 decoder uses legacy per-layer key/value tuples. Newer
Transformers T5 layers use Cache objects instead. Translate at the boundary
of this model instance; do not alter process-wide Transformers behavior.
"""

from __future__ import annotations

import inspect
from types import MethodType

from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.models.t5.modeling_t5 import T5LayerCrossAttention, T5LayerSelfAttention

from mt3_infer.models.yourmt3.model.perceiver_mod import PerceiverTFEncoder
from mt3_infer.models.yourmt3.model.t5mod import T5StackYMT3


def _empty_head_mask(self, head_mask, num_layers: int):
    if head_mask is not None:
        raise ValueError("YourMT3 inference does not support head masks with Transformers 5")
    return [None] * num_layers


def _self_attention(self, hidden_states, attention_mask=None, position_bias=None,
                    layer_head_mask=None, past_key_value=None, use_cache=False,
                    output_attentions=False):
    if layer_head_mask is not None:
        raise ValueError("YourMT3 inference does not support head masks with Transformers 5")
    cache = DynamicCache([past_key_value]) if past_key_value is not None else DynamicCache()
    normed = self.layer_norm(hidden_states)
    attention = self.SelfAttention(
        normed, mask=attention_mask, position_bias=position_bias,
        past_key_values=cache if use_cache else None,
        output_attentions=output_attentions,
    )
    hidden_states = hidden_states + self.dropout(attention[0])
    present = (cache.layers[0].keys, cache.layers[0].values) if use_cache else None
    return (hidden_states, present) + attention[1:]


def _cross_attention(self, hidden_states, key_value_states,
                     attention_mask=None, position_bias=None, layer_head_mask=None,
                     past_key_value=None, use_cache=False, query_length=None,
                     output_attentions=False):
    if layer_head_mask is not None:
        raise ValueError("YourMT3 inference does not support head masks with Transformers 5")
    cross = DynamicCache([past_key_value]) if past_key_value is not None else DynamicCache()
    cache = EncoderDecoderCache(DynamicCache(), cross)
    normed = self.layer_norm(hidden_states)
    attention = self.EncDecAttention(
        normed, key_value_states=key_value_states, mask=attention_mask,
        position_bias=position_bias, past_key_values=cache if use_cache else None,
        output_attentions=output_attentions,
    )
    hidden_states = hidden_states + self.dropout(attention[0])
    present = (cross.layers[0].keys, cross.layers[0].values) if use_cache else None
    return (hidden_states, present) + attention[1:]


def adapt_yourmt3_transformers(model) -> None:
    """Translate legacy attention calls on this loaded model only."""
    modern_attention = "past_key_values" in inspect.signature(T5LayerSelfAttention.forward).parameters
    for module in model.modules():
        if isinstance(module, (T5StackYMT3, PerceiverTFEncoder)) and not hasattr(module, "get_head_mask"):
            module.get_head_mask = MethodType(_empty_head_mask, module)
        if not modern_attention:
            continue
        if isinstance(module, T5LayerSelfAttention):
            module.SelfAttention.layer_idx = 0
            module.forward = MethodType(_self_attention, module)
        elif isinstance(module, T5LayerCrossAttention):
            module.EncDecAttention.layer_idx = 0
            module.forward = MethodType(_cross_attention, module)
