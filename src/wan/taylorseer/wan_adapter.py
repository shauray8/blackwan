# wan_adapter.py

import torch
import functools
import unittest.mock as mock

# Assuming your WanModel class is accessible or imported here if needed
# from your_model_file import WanModel
# For now, we'll assume it has a `.blocks` attribute like a standard transformer.

# We also need the CachedTransformerBlocks from your utility file
from wan.taylor_seer_utils import CachedTransformerBlocks

def apply_cache_on_transformer(transformer):
    """
    Applies the caching mechanism to a WanModel transformer.
    """
    if getattr(transformer, "_is_cached", False):
        return transformer

    # This is the core logic: wrap the original transformer blocks
    # in our CachedTransformerBlocks class.
    cached_blocks = torch.nn.ModuleList(
        [
            CachedTransformerBlocks(
                transformer_blocks=transformer.blocks,
                transformer=transformer,
                # For WanModel, the forward pass of a block returns only the hidden_states tensor.
                return_hidden_states_only=True
            )
        ]
    )

    original_forward = transformer.forward

    @functools.wraps(transformer.forward)
    def new_forward(self, *args, **kwargs):
        # Temporarily replace the 'blocks' attribute with our cached version
        # during the forward pass.
        with mock.patch.object(self, "blocks", cached_blocks):
            return original_forward(*args, **kwargs)

    transformer.forward = new_forward.__get__(transformer)
    transformer._is_cached = True
    print(f"Successfully applied TaylorSeer cache to {transformer.__class__.__name__}.")
    return transformer