# llm_rl_scripts/maze/models/local_patch_encoder.py

from typing import Optional
import jax
import jax.numpy as jnp
import flax.linen as nn


class LocalPatchCNN(nn.Module):
    """
    Encodes [B, H, W] integer-valued maze patches into
    [B, num_visual_tokens, gpt_hidden_size].
    Values expected:
      0 = free
      1 = wall
      2 = agent
    """
    num_cell_types: int = 3
    conv_dim: int = 64
    num_visual_tokens: int = 4
    gpt_hidden_size: int = 768

    @nn.compact
    def __call__(self, patches: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        # patches: [B, H, W]
        x = jax.nn.one_hot(patches.astype(jnp.int32), self.num_cell_types)  # [B,H,W,C]
        x = x.astype(jnp.float32)

        x = nn.Conv(features=self.conv_dim, kernel_size=(2, 2), strides=(1, 1))(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.conv_dim, kernel_size=(2, 2), strides=(1, 1))(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.conv_dim)(x)
        x = nn.tanh(x)

        x = nn.Dense(self.num_visual_tokens * self.gpt_hidden_size)(x)
        x = x.reshape((x.shape[0], self.num_visual_tokens, self.gpt_hidden_size))
        return x