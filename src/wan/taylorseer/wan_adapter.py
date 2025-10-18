import torch
import functools
import unittest.mock as mock

from wan.taylorseer.taylor_seer_utils import CachedTransformerBlocks

def apply_cache_on_transformer(transformer):
    if getattr(transformer, "_is_cached", False):
        return transformer

    cached_blocks = torch.nn.ModuleList(
        [
            CachedTransformerBlocks(
                transformer_blocks=transformer.blocks,
                transformer=transformer,
                return_hidden_states_only=True
            )
        ]
    )

    original_forward = transformer.forward

    @functools.wraps(transformer.forward)
    def new_forward(self, *args, **kwargs):
        with mock.patch.object(self, "blocks", cached_blocks):
            return original_forward(*args, **kwargs)

    transformer.forward = new_forward.__get__(transformer)
    transformer._is_cached = True
    print(f"Successfully applied TaylorSeer cache to {transformer.__class__.__name__}.")
    return transformer
