import torch
import torch.nn.functional as F
from torch import nn, einsum

from typing import Callable, Optional, List

from einops import rearrange, repeat

from beartype import beartype

from muse_pytorch.vqgan_vae import VQGanVAE

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# classes

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class GEGLU(nn.Module):
    """ https://arxiv.org/abs/2002.05202 """

    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return gate * F.gelu(x)

def FeedForward(dim, mult = 4):
    """ https://arxiv.org/abs/2110.09456 """

    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        GEGLU(),
        LayerNorm(inner_dim),
        nn.Linear(inner_dim, dim, bias = False)
    )

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_context = None,
        dim_head = 64,
        heads = 8,
        cross_attend = False
    ):
        super().__init__()
        dim_context = default(dim_context, dim)

        self.scale = dim_head ** -0.5
        self.heads =  heads
        inner_dim = dim_head * heads

        self.cross_attend = cross_attend
        self.norm = LayerNorm(dim)

        self.null_kv = nn.Parameter(torch.randn(2, heads, 1, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        context = None,
        context_mask = None
    ):
        assert not (exists(context) ^ self.cross_attend)

        h, is_cross_attn = self.heads, exists(context)

        x = self.norm(x)

        kv_input = context if self.cross_attend else x

        q, k, v = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1))

        q = q * self.scale

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        nk, nv = self.null_kv
        nk, nv = map(lambda t: repeat(t, 'h 1 d -> b h 1 d', b = x.shape[0]), (nk, nv))

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        if exists(context_mask):
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            context_mask = F.pad(context_mask, (1, 0), value = True)

            mask_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(~context_mask, mask_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerBlocks(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_context = None,
        dim_head = 64,
        heads = 8,
        ff_mult = 4
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads),
                Attention(dim = dim, dim_head = dim_head, heads = heads, dim_context = dim_context, cross_attend = True),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = LayerNorm(dim)

    def forward(self, x, context = None, context_mask = None):
        for attn, cross_attn, ff in self.layers:
            x = attn(x) + x

            x = cross_attn(x, context = context, context_mask = context_mask) + x

            x = ff(x) + x

        return self.norm(x)

# transformer - it's all we need

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        seq_len,
        **kwargs
    ):
        super().__init__()
        self.mask_id = num_tokens

        self.token_emb = nn.Embedding(num_tokens + 1, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.seq_len = seq_len

        self.transformer = TransformerBlocks(dim = dim, **kwargs)
        self.norm = LayerNorm(dim)

        self.to_logits = nn.Linear(dim, num_tokens, bias = False)

    def forward(
        self,
        x,
        return_embed = False,
        labels = None,
        ignore_index = 0
    ):
        device, n = x.device, x.shape[1]
        assert n <= self.seq_len

        x = self.token_emb(x)
        x = x + self.pos_emb(torch.arange(n, device = device))

        x = self.transformer(x)

        if return_embed:
            return x

        logits = self.to_logits(x)

        if not exists(labels):
            return logits

        logits = rearrange(logits, 'b n c -> b c n')
        return F.cross_entropy(logits, labels, ignore_index = ignore_index)

# classifier free guidance functions

def uniform(shape, min = 0, max = 1, device = None):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return uniform(shape, device = device) < prob

# noise schedules

# in original maskgit paper, they claimed cosine schedule had best results
# always assumed it was simply cos(t * pi / 2)
# but this paper had a section that seems to suggest it is (2 / pi) * arccos(x) ?
# if anyone knows the answer to this, would greatly appreciate assistance!

def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)

def arccosine_schedule(t):
    return 2 * math.pi * torch.arccos(t)

# main maskgit classes

@beartype
class MaskGit(nn.Module):
    def __init__(
        self,
        transformer: Transformer,
        noise_schedule: Callable = arccosine_schedule,
        vae: Optional[VQGanVAE] = None
    ):
        super().__init__()
        self.vae = vae.copy_for_eval()
        self.transformer = transformer
        self.noise_schedule = noise_schedule

    def generate(self):
        raise NotImplementedError

    def forward(
        self,
        images_or_ids: torch.Tensor,
        ignore_index = -1
    ):
        batch, seq_len, device = *ids.shape, ids.device

        # tokenize if needed

        if images_or_ids.dtype == torch.float:
            assert exists(self.vae), 'vqgan vae must be passed in if training from raw images'
            with torch.no_grad():
                _, ids, _ = self.vae.encode(images_or_ids)
        else:
            ids = images_or_ids

        # prepare mask

        rand_time = uniform((batch,), device = device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (seq_len * rand_mask_probs).round().clamp(min = 1)

        batch_randperm = torch.rand_like(ids).argsort(dim = -1)
        mask = batch_randperm < rearrange(num_token_masked, 'b -> b 1')

        mask_id = self.transformer.mask_id

        x = torch.where(mask, mask_id, ids)
        labels = torch.where(mask, ids, ignore_index)

        # get loss

        ce_loss = self.transformer(
            ids,
            labels = labels,
            ignore_index = ignore_index
        )

        return ce_loss

# final Muse class

@beartype
class Muse(nn.Module):
    def __init__(
        self,
        base_maskgit: MaskGit,
        superres_maskgit: MaskGit
    ):
        super().__init__()
        self.base_maskgit = base_maskgit
        self.superres_maskgit = superres_maskgit

    def forward(
        self,
        texts: List[str]
    ):
        return None
