from pathlib import Path
import copy
import math
from math import sqrt
from functools import partial, wraps

from vector_quantize_pytorch import VectorQuantize as VQ

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.autograd import grad as torch_grad

import torchvision

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

# constants

MList = nn.ModuleList

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# decorators

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def remove_vgg(fn):
    @wraps(fn)
    def inner(self, *args, **kwargs):
        has_vgg = hasattr(self, '_vgg')
        if has_vgg:
            vgg = self._vgg
            delattr(self, '_vgg')

        out = fn(self, *args, **kwargs)

        if has_vgg:
            self._vgg = vgg

        return out
    return inner

# keyword argument helpers

def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, string_input):
    return string_input.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

# tensor helper functions

def log(t, eps = 1e-10):
    return torch.log(t + eps)

def gradient_penalty(images, output, weight = 10):
    batch_size = images.shape[0]

    gradients = torch_grad(
        outputs = output,
        inputs = images,
        grad_outputs = torch.ones(output.size(), device = images.device),
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((gradients.norm(2, dim = 1) - 1) ** 2).mean()

def leaky_relu(p = 0.1):
    return nn.LeakyReLU(0.1)

def safe_div(numer, denom, eps = 1e-8):
    return numer / denom.clamp(min = eps)

# gan losses

def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()

def hinge_gen_loss(fake):
    return -fake.mean()

def bce_discr_loss(fake, real):
    return (-log(1 - torch.sigmoid(fake)) - log(torch.sigmoid(real))).mean()

def bce_gen_loss(fake):
    return -log(torch.sigmoid(fake)).mean()

def grad_layer_wrt_loss(loss, layer):
    return torch_grad(
        outputs = loss,
        inputs = layer,
        grad_outputs = torch.ones_like(loss),
        retain_graph = True
    )[0].detach()

# vqgan vae

class LayerNormChan(nn.Module):
    def __init__(
        self,
        dim,
        eps = 1e-5
    ):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * var.clamp(min = self.eps).rsqrt() * self.gamma

# discriminator

class Discriminator(nn.Module):
    def __init__(
        self,
        dims,
        channels = 3,
        groups = 16,
        init_kernel_size = 5
    ):
        super().__init__()
        dim_pairs = zip(dims[:-1], dims[1:])

        self.layers = MList([nn.Sequential(nn.Conv2d(channels, dims[0], init_kernel_size, padding = init_kernel_size // 2), leaky_relu())])

        for dim_in, dim_out in dim_pairs:
            self.layers.append(nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 4, stride = 2, padding = 1),
                nn.GroupNorm(groups, dim_out),
                leaky_relu()
            ))

        dim = dims[-1]
        self.to_logits = nn.Sequential( # return 5 x 5, for PatchGAN-esque training
            nn.Conv2d(dim, dim, 1),
            leaky_relu(),
            nn.Conv2d(dim, 1, 4)
        )

    def forward(self, x):
        for net in self.layers:
            x = net(x)

        return self.to_logits(x)

# resnet encoder / decoder

class ResnetEncDec(nn.Module):
    def __init__(
        self,
        dim,
        *,
        channels = 3,
        layers = 4,
        layer_mults = None,
        num_resnet_blocks = 1,
        resnet_groups = 16,
        first_conv_kernel_size = 5
    ):
        super().__init__()
        assert dim % resnet_groups == 0, f'dimension {dim} must be divisible by {resnet_groups} (groups for the groupnorm)'

        self.layers = layers

        self.encoders = MList([])
        self.decoders = MList([])

        layer_mults = default(layer_mults, list(map(lambda t: 2 ** t, range(layers))))
        assert len(layer_mults) == layers, 'layer multipliers must be equal to designated number of layers'

        layer_dims = [dim * mult for mult in layer_mults]
        dims = (dim, *layer_dims)

        self.encoded_dim = dims[-1]

        dim_pairs = zip(dims[:-1], dims[1:])

        append = lambda arr, t: arr.append(t)
        prepend = lambda arr, t: arr.insert(0, t)

        if not isinstance(num_resnet_blocks, tuple):
            num_resnet_blocks = (*((0,) * (layers - 1)), num_resnet_blocks)

        assert len(num_resnet_blocks) == layers, 'number of resnet blocks config must be equal to number of layers'

        for layer_index, (dim_in, dim_out), layer_num_resnet_blocks in zip(range(layers), dim_pairs, num_resnet_blocks):
            append(self.encoders, nn.Sequential(nn.Conv2d(dim_in, dim_out, 4, stride = 2, padding = 1), leaky_relu()))
            prepend(self.decoders, nn.Sequential(nn.ConvTranspose2d(dim_out, dim_in, 4, 2, 1), leaky_relu()))

            for _ in range(layer_num_resnet_blocks):
                append(self.encoders, ResBlock(dim_out, groups = resnet_groups))
                prepend(self.decoders, GLUResBlock(dim_out, groups = resnet_groups))

        prepend(self.encoders, nn.Conv2d(channels, dim, first_conv_kernel_size, padding = first_conv_kernel_size // 2))
        append(self.decoders, nn.Conv2d(dim, channels, 1))

    def get_encoded_fmap_size(self, image_size):
        return image_size // (2 ** self.layers)

    @property
    def last_dec_layer(self):
        return self.decoders[-1].weight

    def encode(self, x):
        for enc in self.encoders:
            x = enc(x)
        return x

    def decode(self, x):
        for dec in self.decoders:
            x = dec(x)
        return x

class GLUResBlock(nn.Module):
    def __init__(self, chan, groups = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan * 2, 3, padding = 1),
            nn.GLU(dim = 1),
            nn.GroupNorm(groups, chan),
            nn.Conv2d(chan, chan * 2, 3, padding = 1),
            nn.GLU(dim = 1),
            nn.GroupNorm(groups, chan),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x

class ResBlock(nn.Module):
    def __init__(self, chan, groups = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.GroupNorm(groups, chan),
            leaky_relu(),
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.GroupNorm(groups, chan),
            leaky_relu(),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x

# main vqgan-vae classes

class VQGanVAE(nn.Module):
    def __init__(
        self,
        *,
        dim,
        channels = 3,
        layers = 4,
        l2_recon_loss = False,
        use_hinge_loss = True,
        vgg = None,
        vq_codebook_dim = 256,
        vq_codebook_size = 512,
        vq_decay = 0.8,
        vq_commitment_weight = 1.,
        vq_kmeans_init = True,
        vq_use_cosine_sim = True,
        use_vgg_and_gan = True,
        discr_layers = 4,
        **kwargs
    ):
        super().__init__()
        vq_kwargs, kwargs = groupby_prefix_and_trim('vq_', kwargs)
        encdec_kwargs, kwargs = groupby_prefix_and_trim('encdec_', kwargs)

        self.channels = channels
        self.codebook_size = vq_codebook_size
        self.dim_divisor = 2 ** layers

        enc_dec_klass = ResnetEncDec

        self.enc_dec = enc_dec_klass(
            dim = dim,
            channels = channels,
            layers = layers,
            **encdec_kwargs
        )

        self.vq = VQ(
            dim = self.enc_dec.encoded_dim,
            codebook_dim = vq_codebook_dim,
            codebook_size = vq_codebook_size,
            decay = vq_decay,
            commitment_weight = vq_commitment_weight,
            accept_image_fmap = True,
            kmeans_init = vq_kmeans_init,
            use_cosine_sim = vq_use_cosine_sim,
            **vq_kwargs
        )

        # reconstruction loss

        self.recon_loss_fn = F.mse_loss if l2_recon_loss else F.l1_loss

        # turn off GAN and perceptual loss if grayscale

        self._vgg = None
        self.discr = None
        self.use_vgg_and_gan = use_vgg_and_gan

        if not use_vgg_and_gan:
            return

        # preceptual loss

        if exists(vgg):
            self._vgg = vgg

        # gan related losses

        layer_mults = list(map(lambda t: 2 ** t, range(discr_layers)))
        layer_dims = [dim * mult for mult in layer_mults]
        dims = (dim, *layer_dims)

        self.discr = Discriminator(dims = dims, channels = channels)

        self.discr_loss = hinge_discr_loss if use_hinge_loss else bce_discr_loss
        self.gen_loss = hinge_gen_loss if use_hinge_loss else bce_gen_loss

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def vgg(self):
        if exists(self._vgg):
            return self._vgg

        vgg = torchvision.models.vgg16(pretrained = True)
        vgg.classifier = nn.Sequential(*vgg.classifier[:-2])
        self._vgg = vgg.to(self.device)
        return self._vgg

    @property
    def encoded_dim(self):
        return self.enc_dec.encoded_dim

    def get_encoded_fmap_size(self, image_size):
        return self.enc_dec.get_encoded_fmap_size(image_size)

    def copy_for_eval(self):
        device = next(self.parameters()).device
        vae_copy = copy.deepcopy(self.cpu())

        if vae_copy.use_vgg_and_gan:
            del vae_copy.discr
            del vae_copy._vgg

        vae_copy.eval()
        return vae_copy.to(device)

    @remove_vgg
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    @remove_vgg
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        state_dict = torch.load(str(path))
        self.load_state_dict(state_dict)

    @property
    def codebook(self):
        return self.vq.codebook

    def encode(self, fmap):
        fmap = self.enc_dec.encode(fmap)
        fmap, indices, commit_loss = self.vq(fmap)
        return fmap, indices, commit_loss

    def decode_from_ids(self, ids):
        codes = self.codebook[ids]
        fmap = self.vq.project_out(codes)
        fmap = rearrange(fmap, 'b h w c -> b c h w')
        return self.decode(fmap)

    def decode(self, fmap):
        return self.enc_dec.decode(fmap)

    def forward(
        self,
        img,
        return_loss = False,
        return_discr_loss = False,
        return_recons = False,
        add_gradient_penalty = True
    ):
        batch, channels, height, width, device = *img.shape, img.device

        for dim_name, size in (('height', height), ('width', width)):
            assert (size % self.dim_divisor) == 0, f'{dim_name} must be divisible by {self.dim_divisor}'

        assert channels == self.channels, 'number of channels on image or sketch is not equal to the channels set on this VQGanVAE'

        fmap, indices, commit_loss = self.encode(img)

        fmap = self.decode(fmap)

        if not return_loss and not return_discr_loss:
            return fmap

        assert return_loss ^ return_discr_loss, 'you should either return autoencoder loss or discriminator loss, but not both'

        # whether to return discriminator loss

        if return_discr_loss:
            assert exists(self.discr), 'discriminator must exist to train it'

            fmap.detach_()
            img.requires_grad_()

            fmap_discr_logits, img_discr_logits = map(self.discr, (fmap, img))

            discr_loss = self.discr_loss(fmap_discr_logits, img_discr_logits)

            if add_gradient_penalty:
                gp = gradient_penalty(img, img_discr_logits)
                loss = discr_loss + gp

            if return_recons:
                return loss, fmap

            return loss

        # reconstruction loss

        recon_loss = self.recon_loss_fn(fmap, img)

        # early return if training on grayscale

        if not self.use_vgg_and_gan:
            if return_recons:
                return recon_loss, fmap

            return recon_loss

        # perceptual loss

        img_vgg_input = img
        fmap_vgg_input = fmap

        if img.shape[1] == 1:
            # handle grayscale for vgg
            img_vgg_input, fmap_vgg_input = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c = 3), (img_vgg_input, fmap_vgg_input))

        img_vgg_feats = self.vgg(img_vgg_input)
        recon_vgg_feats = self.vgg(fmap_vgg_input)
        perceptual_loss = F.mse_loss(img_vgg_feats, recon_vgg_feats)

        # generator loss

        gen_loss = self.gen_loss(self.discr(fmap))

        # calculate adaptive weight

        last_dec_layer = self.enc_dec.last_dec_layer

        norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(p = 2)
        norm_grad_wrt_perceptual_loss = grad_layer_wrt_loss(perceptual_loss, last_dec_layer).norm(p = 2)

        adaptive_weight = safe_div(norm_grad_wrt_perceptual_loss, norm_grad_wrt_gen_loss)
        adaptive_weight.clamp_(max = 1e4)

        # combine losses

        loss = recon_loss + perceptual_loss + commit_loss + adaptive_weight * gen_loss

        if return_recons:
            return loss, fmap

        return loss
