import io
import sys
import os
import requests
import PIL
import warnings
import hashlib
import urllib
import yaml
from pathlib import Path
from tqdm import tqdm
from math import sqrt, log
from packaging import version

from omegaconf import OmegaConf
import importlib

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from taming.models.vqgan import VQModel, GumbelVQ
from dalle_pytorch import distributed_utils

# constants

CACHE_PATH = os.path.expanduser("~/.cache/dalle")

OPENAI_VAE_ENCODER_PATH = 'https://cdn.openai.com/dall-e/encoder.pkl'
OPENAI_VAE_DECODER_PATH = 'https://cdn.openai.com/dall-e/decoder.pkl'

VQGAN_VAE_PATH = 'https://heibox.uni-heidelberg.de/f/140747ba53464f49b476/?dl=1'
VQGAN_VAE_CONFIG_PATH = 'https://heibox.uni-heidelberg.de/f/6ecf2af6c658432c8298/?dl=1'

# helpers methods

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


def download(url, filename = None, root = CACHE_PATH, is_distributed=None, backend=None):
    filename = default(filename, os.path.basename(url))

    download_target = os.path.join(root, filename)
    download_target_tmp = os.path.join(root, f'tmp.{filename}')

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")


    if os.path.isfile(download_target):
        return download_target

    with urllib.request.urlopen(url) as source, open(download_target_tmp, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    os.rename(download_target_tmp, download_target)
    if (
            distributed_utils.is_distributed
            and distributed_utils.backend.is_local_root_worker()
    ):
        distributed_utils.backend.local_barrier()
    return download_target


# VQGAN from Taming Transformers paper
# https://arxiv.org/abs/2012.09841

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

class VQGanVAE(nn.Module):
    def __init__(self, vqgan_model_path=None, vqgan_config_path=None):
        super().__init__()

        if vqgan_model_path is None:
            model_filename = 'vqgan.1024.model.ckpt'
            config_filename = 'vqgan.1024.config.yml'
            download(VQGAN_VAE_CONFIG_PATH, config_filename)
            download(VQGAN_VAE_PATH, model_filename)
            config_path = str(Path(CACHE_PATH) / config_filename)
            model_path = str(Path(CACHE_PATH) / model_filename)
        else:
            model_path = vqgan_model_path
            config_path = vqgan_config_path

        config = OmegaConf.load(config_path)

        model = instantiate_from_config(config["model"])

        state = torch.load(model_path, map_location = 'cpu')['state_dict']
        model.load_state_dict(state, strict = False)

        print(f"Loaded VQGAN from {model_path} and {config_path}")

        self.model = model

        # f as used in https://github.com/CompVis/taming-transformers#overview-of-pretrained-models
        f = config.model.params.ddconfig.resolution / config.model.params.ddconfig.attn_resolutions[0]
        # var: codebook_size
        # fn: get_encoded_fmap_size
        # fn: decode_from_ids
        
        self.num_layers = int(log(f)/log(2))
        self.channels = 3
        self.image_size = 256
        self.num_tokens = config.model.params.n_embed
        self.is_gumbel = isinstance(self.model, GumbelVQ)

        self.encode = self.model.encode


    @torch.no_grad()
    def get_codebook_indices(self, img):
        b = img.shape[0]
        img = (2 * img) - 1
        _, _, [_, _, indices] = self.model.encode(img)
        if self.is_gumbel:
            return rearrange(indices, 'b h w -> b (h w)', b=b)
        return rearrange(indices, '(b n) -> b n', b = b)

    def decode(self, img_seq):
        b, n = img_seq.shape
        one_hot_indices = F.one_hot(img_seq, num_classes = self.num_tokens).float()
        z = one_hot_indices @ self.model.quantize.embed.weight if self.is_gumbel \
            else (one_hot_indices @ self.model.quantize.embedding.weight)

        z = rearrange(z, 'b (h w) c -> b c h w', h = int(sqrt(n)))
        img = self.model.decode(z)

        img = (img.clamp(-1., 1.) + 1) * 0.5
        return img

    def forward(self, img):
        raise NotImplemented

