
from pathlib import Path
from shutil import rmtree

from beartype import beartype

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from muse_maskgit_pytorch.vqgan_vae import VQGanVAE

from einops import rearrange

from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs

from ema_pytorch import EMA

from muse_maskgit_pytorch.muse_maskgit_pytorch import MaskGit
from muse_maskgit_pytorch.trainers.base_accelerated_trainer import BaseAcceleratedTrainer
from muse_maskgit_pytorch.t5 import t5_encode_text_from_encoded
import torch.nn.functional as F
def noop(*args, **kwargs):
    pass

def exists(val):
    return val is not None

class MaskGitTrainer(BaseAcceleratedTrainer):
    def __init__(
        self,
        maskgit: MaskGit,
        dataloader,
        valid_dataloader,
        accelerator,
        *,
        current_step,
        num_train_steps,
        batch_size,
        gradient_accumulation_steps=1,
        max_grad_norm=None,
        save_results_every=100,
        save_model_every=1000,
        results_dir="./results",
        logging_dir="./results/logs",
        apply_grad_penalty_every=4,
        accelerate_kwargs: dict = dict(),
        lr=3e-4,
        use_ema=True,
        ema_beta=0.995,
        ema_update_after_step=0,
        ema_update_every=1,
        validation_prompt="a photo of a dog"
    ):
        super().__init__(dataloader, valid_dataloader, current_step=current_step, num_train_steps=num_train_steps, batch_size=batch_size,\
                        gradient_accumulation_steps=gradient_accumulation_steps, max_grad_norm=max_grad_norm, save_results_every=save_results_every, \
                        save_model_every=save_model_every, results_dir=results_dir, logging_dir=logging_dir, apply_grad_penalty_every=apply_grad_penalty_every, \
                        accelerate_kwargs=accelerate_kwargs)

        # maskgit
        self.model = maskgit
        self.model.vae.requires_grad_(False)

        all_parameters = set(maskgit.parameters())
        # don't train the vae

        vae_parameters = set(self.model.vae.parameters())
        transformer_parameters = all_parameters - vae_parameters

        # optimizers

        self.optim = Adam(transformer_parameters, lr=lr)

        # prepare with accelerator

        (
            self.model,
            self.optim,
            self.dl,
            self.valid_dl,
        ) = self.prepare(
            self.model, self.optim, self.dl, self.valid_dl
        )

        self.use_ema = use_ema
        self.validation_prompt = validation_prompt
        if use_ema:
            self.ema_model = EMA(
                maskgit,
                update_after_step=ema_update_after_step,
                update_every=ema_update_every,
            )
            self.ema_model = self.prepare(self.ema_model)
    def log_validation_images(self, validation_prompt, step, cond_image=None, cond_scale=3, temperature=1):
        image = self.model.generate([validation_prompt], cond_images=cond_image, cond_scale=cond_scale, temperature=temperature)
        super().log_validation_images([image], step, validation_prompt)
    def train_step(self):
        device = self.device
        steps = int(self.steps.item())
        apply_grad_penalty = not (steps % self.apply_grad_penalty_every)


        if self.use_ema:
            ema_model = self.ema_model.module if self.is_distributed else self.ema_model
        self.model.train()
        # logs
        train_loss = 0
        with self.accelerator.accumulate(self.model):
            imgs, input_ids, attn_mask = next(self.dl_iter)
            text_embeds = t5_encode_text_from_encoded(input_ids, attn_mask, self.model.t5, device)
            imgs = imgs.to(device)
            loss = self.model(
                imgs,
                text_embeds=text_embeds,
                add_gradient_penalty = apply_grad_penalty,
                return_loss = True
            )
            avg_loss = self.accelerator.gather(loss.repeat(self.batch_size)).mean()
            train_loss += avg_loss.item() / self.gradient_accumulation_steps
            self.accelerator.backward(loss)
            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optim.step()
            self.optim.zero_grad()
        if self.accelerator.sync_gradients:
            self.steps += 1
            if self.use_ema:
                ema_model.update()
            logs = {"loss": train_loss}

            self.accelerator.log(logs, step=self.steps)

            if steps % self.save_model_every:
                state_dict = self.accelerator.unwrap_model(self.model).state_dict()
                maskgit_save_name = 'maskgit_superres' if self.model.cond_image_size else 'maskgit'
                model_path = str(self.results_dir / f'{maskgit_save_name}.{steps}.pt')
                self.accelerator.save(state_dict, model_path)

                if self.use_ema:
                    ema_state_dict = self.accelerator.unwrap_model(self.ema_model).state_dict()
                    model_path = str(self.results_dir / f'{maskgit_save_name}.{steps}.ema.pt')
                    self.accelerator.save(ema_state_dict, model_path)

                self.print(f'{steps}: saving model to {str(self.results_dir)}')
            if steps % self.log_model_every:
                cond_image = None
                if self.model.cond_image_size:
                    cond_image =F.interpolate(imgs[0], 256)
                self.log_validation_images(self.validation_prompt, self.steps, cond_image=cond_image)
        return logs