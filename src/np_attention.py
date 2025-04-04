import math
import numpy as np
import torch
import torch.nn as nn


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    Note that softmax is shift invariant,
    hence our implementation shifts by the max logits to mitigate numerical instability.
    """
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads, rng):
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads

        # For simplicity, all the dimensionalities are assumed to be the same
        self.params_q = rng.randn(embed_dim, embed_dim)
        self.params_k = rng.randn(embed_dim, embed_dim)
        self.params_v = rng.randn(embed_dim, embed_dim) # Technically this is concatenating embed_dim / num_heads value matrices for each head
        # Dim for each head is (embed_dim, embed_dim / num_heads)
        self.param_out = rng.randn(embed_dim, embed_dim) # This is the concatenation of the output matrices for each head
        # Dim for each head is (embed_dim, embed_dim / num_heads)

    def __call__(self, q, k, v):
        """
        The forward call of an attention layer.
        Inputs:
        - q: the query
        - k: the key
        - v: the value

        NOTE: For simplicity, we assume that all inputs have the same shape:
            (batch_size, context_length, embed_dim)

        NOTE: self.params_{q, k, v} of shape (embed_dim, embed_dim) can be interpreted as
             ________ ________ _______ ________
            |        |        |       |        |
            |        |        |       |        |
            |        |        |       |        |
            | head_1 | head_2 |  ...  | head_H |
            |        |        |       |        |
            |        |        |       |        |
            |________|________|_______|________|
            where head_h is shaped (embed_dim, embed_dim / H),
        """
        batch_size, context_length, embed_dim = q.shape

        # each attention head dimension
        head_dim = embed_dim // self.num_heads

        # the sqrt(d) scaling in attention head
        scale = 1.0 / math.sqrt(head_dim)
        out = None

        # ========================================================
        # TODO: Compute forward call of a multihead attention, without masking
        # Step 1: Transform Q, K, V into embedding
        q = q @ self.params_q # (batch_size, context_length, embed_dim)
        k = k @ self.params_k # (batch_size, context_length, embed_dim)
        v = v @ self.params_v # (batch_size, context_length, embed_dim)
        

        # Step 2: Reshape for multihead attention
        q = q.reshape(batch_size, context_length, self.num_heads, head_dim) # (batch_size, context_length, num_heads, head_dim)
        k = k.reshape(batch_size, context_length, self.num_heads, head_dim) # (batch_size, context_length, num_heads, head_dim)
        v = v.reshape(batch_size, context_length, self.num_heads, head_dim) # (batch_size, context_length, num_heads, head_dim)

        # Step 3: Apply softmax attention
        q = q.transpose(0, 2, 1, 3) # (batch_size, num_heads, context_length, head_dim)
        k = k.transpose(0, 2, 1, 3) # (batch_size, num_heads, context_length, head_dim)
        v = v.transpose(0, 2, 1, 3) # (batch_size, num_heads, context_length, head_dim)
        # Step 4: Apply softmax attention
        qk = q @ k.transpose(0, 1, 3, 2) # (batch_size, num_heads, context_length, context_length)
        qk = qk * scale
        qk = softmax(qk)
        out = qk @ v # (batch_size, num_heads, context_length, head_dim)
        out = out.transpose(0, 2, 1, 3) # (batch_size, context_length, num_heads, head_dim)
        out = out.reshape(batch_size, context_length, embed_dim) # (batch_size, context_length, embed_dim)
        out = out @ self.param_out # (batch_size, context_length, embed_dim)

        
        # ========================================================

        # NOTE: out is shaped (batch_size, context_length, embed_dim)
        return out


if __name__ == "__main__":
    seed = 42
    rng = np.random.RandomState(seed)

    batch_size = 5
    context_length = 4
    embed_dim = 8
    num_heads = 4

    np_attention = MultiHeadAttention(embed_dim, num_heads, rng)
    torch_attention = nn.MultiheadAttention(
        embed_dim,
        num_heads,
        bias=False,
        batch_first=True,
    )

    # Use parameters from the NumPy attention model
    state_dict = {
        "in_proj_weight": torch.from_numpy(np.concatenate((
            np_attention.params_q.T,
            np_attention.params_k.T,
            np_attention.params_v.T,
        ), axis=0)).float(),
        "out_proj.weight": torch.from_numpy(
            np_attention.param_out.T
        ).float(),
    }
    torch_attention.load_state_dict(state_dict)

    # Sample random query, key, and value
    q = rng.randn(batch_size, context_length, embed_dim)
    k = rng.randn(batch_size, context_length, embed_dim)
    v = rng.randn(batch_size, context_length, embed_dim)

    np_out = np_attention(q, k, v)
    torch_out = torch_attention(
        torch.from_numpy(q).float(),
        torch.from_numpy(k).float(),
        torch.from_numpy(v).float(),
        need_weights=True,
    )[0]

    print("Error should be very small, perhaps around 1e-5.")
    print("Maximum error: {}".format(
        np.max(np.abs(np_out - torch_out.detach().numpy())))
    )
