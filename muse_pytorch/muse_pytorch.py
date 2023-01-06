from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, einsum

from typing import Callable, Optional, List

from einops import rearrange, repeat

from beartype import beartype

from muse_pytorch.vqgan_vae import VQGanVAE
from muse_pytorch.t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME

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
        dim_head = 64,
        heads = 8,
        cross_attend = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads =  heads
        inner_dim = dim_head * heads

        self.cross_attend = cross_attend
        self.norm = LayerNorm(dim)

        self.null_kv = nn.Parameter(torch.randn(2, heads, 1, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
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
        dim_head = 64,
        heads = 8,
        ff_mult = 4
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads),
                Attention(dim = dim, dim_head = dim_head, heads = heads, cross_attend = True),
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

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 3.,
        **kwargs
    ):
        logits = self.forward(self, *args, cond_drop_prob = 0., **kwargs)
        if cond_scale == 1:
            return logits

        null_logits = self.forward(self, *args, cond_drop_prob = 1., **kwargs)

        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        return_embed = False,
        labels = None,
        ignore_index = 0,
        cond_drop_prob = 0.,
        conditioning_token_ids: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[torch.Tensor] = None
    ):
        device, n = x.device, x.shape[1]
        assert n <= self.seq_len

        # prepare texts

        assert exists(texts) ^ exists(text_embeds)

        if exists(texts):
            text_embeds = self.encode_text(texts)
            contexts = self.text_embed_proj(text_embeds)

        context_mask = torch.any(text_embeds == 0, dim = -1)

        # classifier free guidance

        if self.training and cond_drop_prob > 0.:
            mask = prob_mask_like((batch, 1), 1. - cond_drop_prob)
            text_mask = text_mask & mask

        # concat conditioning image token ids if needed

        if exists(conditioning_token_ids):
            cond_token_emb = self.token_emb(conditioning_token_ids)
            contexts = torch.cat((context, cond_token_emb), dim = -2)
            context_mask = F.pad(context_mask, (0, conditioning_token_ids.shape[-1]), value = True)

        # embed tokens

        x = self.token_emb(x)
        x = x + self.pos_emb(torch.arange(n, device = device))

        x = self.transformer(x, context = context, context_mask = context_mask)

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

# sampling helpers

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    return probs.scatter(1, ind, val)

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
        image_size,
        transformer: Transformer,
        noise_schedule: Callable = arccosine_schedule,
        vae: Optional[VQGanVAE] = None,
        cond_vae: Optional[VQGanVAE] = None,
        cond_image_size = None,
        resize_image_for_cond_image = False,
        t5_name = DEFAULT_T5_NAME,
        cond_drop_prob = 0.5
    ):
        super().__init__()
        self.vae = vae.copy_for_eval() if exists(val) else None

        if exists(cond_vae):
            self.cond_vae = cond_vae.eval()
        else:
            self.cond_vae = self.vae

        assert not (exists(cond_vae) and not exists(cond_image_size)), 'cond_image_size must be specified if conditioning'

        self.image_size = image_size
        self.cond_image_size = cond_image_size
        self.resize_image_for_cond_image = resize_image_for_cond_image

        self.encode_text = partial(t5_encode_text, name = t5_name)

        dim = transformer.dim
        text_embed_dim = get_encoded_dim(t5_name)

        self.text_embed_proj = nn.Linear(text_embed_dim, dim, bias = False) if text_embed_dim != dim else nn.Identity() 

        self.cond_drop_prob = cond_drop_prob

        self.transformer = transformer
        self.mask_id = transformer.mask_id
        self.noise_schedule = noise_schedule

    def generate(
        self,
        texts: List[str],
        fmap_size = None,
        temperature = 1.,
        topk_thres = 0.9,
        timesteps = 18,  # ideal number of steps is 18 in maskgit paper
        cond_scale = 3,
    ):
        fmap_size = default(fmap_size, self.vae.get_encoded_fmap_size(self.image_size))

        # begin with all image token ids masked


        seq_len = fmap_size ** 2

        batch_size = len(texts)

        shape = (batch_size, seq_len)

        ids = torch.full(shape, self.mask_id, dtype = torch.long)
        scores = torch.zeros(shape, dtype = torch.float32)

        starting_temperature = temperature

        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps), reversed(range(self.steps))):

            rand_mask_prob = self.noise_schedule(timestep)
            num_token_masked = max(rand_mask_prob.item(), 1)

            masked_indices = scores.topk(num_token_masked, dim = -1).indices

            ids = ids.scatter(1, masked_indices, self.mask_id)

            logits = self.transformer.forward_with_cond_scale(
                ids,
                texts = texts,
                cond_scale = cond_scale
            )

            filtered_logits = top_k(logits, topk_thres)

            temperature = starting_temperature * (steps_until_x0 / self.steps) # temperature is annealed

            pred_ids = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)

            ids = torch.where(
                ids == self.mask_id,
                pred_ids,
                ids
            )

            scores = logits.gather(2, pred_ids)

        # get ids

        ids = rearrange(ids, 'b (i j), -> b i j', i = fmap_size, j = fmap_size)

        if not exists(self.vae):
            return ids

        codes = self.vae.codebook[ids]
        images = self.vae.decode(codes)
        return images

    def forward(
        self,
        images_or_ids: torch.Tensor,
        ignore_index = -1,
        cond_images_or_ids: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[torch.Tensor] = None,
        cond_drop_prob = None
    ):
        batch, seq_len, device, cond_drop_prob = *ids.shape, ids.device, default(cond_drop_prob, self.cond_drop_prob)

        # tokenize if needed

        if images_or_ids.dtype == torch.float:
            assert exists(self.vae), 'vqgan vae must be passed in if training from raw images'
            assert all(height_or_width == self.image_size for height_or_width in images_or_ids.shape[-2:])

            with torch.no_grad():
                _, ids, _ = self.vae.encode(images_or_ids)
        else:
            assert not self.resize_image_for_cond_image, 'you cannot pass in raw image token ids if you want the framework to autoresize image for conditioning super res transformer'
            ids = images_or_ids

        # take care of conditioning image if specified

        if self.resize_image_for_cond_image:
            cond_images_or_ids = F.interpolate(images_or_ids, self.cond_image_size, mode = 'nearest')

        # tokenize conditional images if needed

        cond_ids = None

        if exists(cond_images_or_ids):
            if cond_images_or_ids.dtype == torch.float:
                assert exists(self.cond_vae), 'cond vqgan vae must be passed in'
                assert all(height_or_width == self.cond_image_size for height_or_width in cond_images_or_ids[-2:])

                with torch.no_grad():
                    _, cond_ids, _ = self.cond_vae.encode(cond_images_or_ids)
            else:
                cond_ids = cond_image_or_ids

        # prepare mask

        rand_time = uniform((batch,), device = device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (seq_len * rand_mask_probs).round().clamp(min = 1)

        mask_id = self.mask_id
        batch_randperm = torch.rand_like(ids).argsort(dim = -1)
        mask = batch_randperm < rearrange(num_token_masked, 'b -> b 1')

        mask_id = self.transformer.mask_id

        x = torch.where(mask, mask_id, ids)
        labels = torch.where(mask, ids, ignore_index)

        # get loss

        ce_loss = self.transformer(
            ids,
            texts = texts,
            text_embeds = text_embeds,
            conditioning_token_ids = cond_ids,
            labels = labels,
            cond_drop_prob = cond_drop_prob,
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

        assert superres_maskgit.resize_image_for_cond_image
        self.superres_maskgit = superres_maskgit

    def forward(
        self,
        texts: List[str],
        cond_scale = 3.,
        temperature = 1.,
        timesteps = 18,
        superres_timesteps = None,
        return_lowres = False
    ):
        lowres_image = self.base_maskgit.generate(
            texts = texts,
            cond_scale = cond_scale,
            temperature = temperature,
            timesteps = timesteps
        )

        superres_image = self.superres_maskgit.generate(
            texts = texts,
            cond_scale = cond_scale,
            cond_images_or_ids = lowres_image,
            temperature = temperature,
            timesteps = default(superres_timesteps, timesteps)
        )
        
        if not return_lowres:
            return superres_image

        return superres_image, lowres_image
