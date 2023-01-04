import torch
from torch import nn, einsum

from einops import rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# classes

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads =  heads
        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        context = None
    ):
        h, is_cross_attn = self.heads, exists(context)
        kv_input = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1))

        q = q * self.scale

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
