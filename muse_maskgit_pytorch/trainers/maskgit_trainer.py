from pathlib import Path
from shutil import rmtree

from beartype import beartype

import torch
from torch import nn

from torch.optim import Adam, AdamW
from lion_pytorch import Lion

from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from PIL import Image
from muse_maskgit_pytorch.vqgan_vae import VQGanVAE
import bitsandbytes as bnb

from einops import rearrange

from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs

from ema_pytorch import EMA
from diffusers.optimization import get_scheduler

from muse_maskgit_pytorch.muse_maskgit_pytorch import MaskGit
from muse_maskgit_pytorch.trainers.base_accelerated_trainer import (
    BaseAcceleratedTrainer,
)
from muse_maskgit_pytorch.t5 import t5_encode_text_from_encoded
import torch.nn.functional as F
import os


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
        lr=3e-4,
        lr_scheduler_type="constant",
        lr_warmup_steps=500,
        use_ema=True,
        ema_beta=0.995,
        ema_update_after_step=0,
        ema_update_every=1,
        validation_prompts=["a photo of a dog"],
        clear_previous_experiments=False,
        validation_image_scale=1,
        only_save_last_checkpoint=False,
        optimizer="Lion",
        weight_decay=0.0,
        use_8bit_adam=False
    ):
        super().__init__(
            dataloader,
            valid_dataloader,
            accelerator,
            current_step=current_step,
            num_train_steps=num_train_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            save_results_every=save_results_every,
            save_model_every=save_model_every,
            results_dir=results_dir,
            logging_dir=logging_dir,
            apply_grad_penalty_every=apply_grad_penalty_every,
            clear_previous_experiments=clear_previous_experiments,
            validation_image_scale=validation_image_scale,
            only_save_last_checkpoint=only_save_last_checkpoint,
        )
        self.save_results_every = save_results_every
        self.batch_size = batch_size
        # maskgit
        self.model = maskgit
        self.model.vae.requires_grad_(False)
        self.model.transformer.t5.requires_grad_(False)

        all_parameters = set(maskgit.parameters())
        # don't train the vae

        vae_parameters = set(self.model.vae.parameters())
        t5_parameters = set(self.model.transformer.t5.parameters())
        transformer_parameters = all_parameters - vae_parameters - t5_parameters

        # optimizers
        if optimizer == "Adam":
            if use_8bit_adam:
                self.optim = bnb.optim.Adam8bit(transformer_parameters, lr=lr)
            else:
                self.optim = Adam(transformer_parameters, lr=lr, weight_decay=weight_decay)
        elif optimizer == "AdamW":
            if use_8bit_adam:
                self.optim = bnb.optim.AdamW8bit(transformer_parameters, lr=lr)
            else:
                self.optim = AdamW(transformer_parameters, lr=lr, weight_decay=weight_decay)
        elif optimizer == "Lion":
            self.optim = Lion(transformer_parameters, lr=lr, weight_decay=weight_decay)
            if use_8bit_adam:
                print("8bit is not supported with the Lion optimiser, Using standard Lion instead.")
        else:
            print(f"{optimizer} optimizer not supported yet.")

        self.lr_scheduler = get_scheduler(
            lr_scheduler_type,
            optimizer=self.optim,
            num_warmup_steps=lr_warmup_steps * self.gradient_accumulation_steps,
            num_training_steps=self.num_train_steps * self.gradient_accumulation_steps,
        )

        # prepare with accelerator

        (
            self.model,
            self.optim,
            self.dl,
            self.valid_dl,
            self.lr_scheduler,
        ) = self.prepare(
            self.model, self.optim, self.dl, self.valid_dl, self.lr_scheduler
        )

        self.use_ema = use_ema
        self.validation_prompts = validation_prompts
        if use_ema:
            self.ema_model = EMA(
                maskgit,
                update_after_step=ema_update_after_step,
                update_every=ema_update_every,
            )
            self.ema_model = self.prepare(self.ema_model)

    def log_validation_images(
        self, validation_prompts, step, cond_image=None, cond_scale=3, temperature=1
    ):
        images = self.model.generate(
            validation_prompts,
            cond_images=cond_image,
            cond_scale=cond_scale,
            temperature=temperature,
        )
        step = int(step.item())
        save_file = str(self.results_dir / f"MaskGit" / f"maskgit_{step}.png")
        os.makedirs(str(self.results_dir / f"MaskGit"), exist_ok=True)

        save_image(images, save_file)
        super().log_validation_images(
            [Image.open(save_file)], step, [" ".join(validation_prompts)]
        )

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
            imgs, input_ids, attn_mask = (
                imgs.to(device),
                input_ids.to(device),
                attn_mask.to(device),
            )
            text_embeds = t5_encode_text_from_encoded(
                input_ids, attn_mask, self.model.transformer.t5, device
            )
            loss = self.model(imgs, text_embeds=text_embeds)
            avg_loss = self.accelerator.gather(loss.repeat(self.batch_size)).mean()
            train_loss += avg_loss.item() / self.gradient_accumulation_steps
            self.accelerator.backward(loss)
            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
            self.lr_scheduler.step()
            self.optim.step()
            self.optim.zero_grad()
        if self.accelerator.sync_gradients:
            self.steps += 1
            if self.use_ema:
                ema_model.update()
            logs = {"loss": train_loss, "lr": self.lr_scheduler.get_last_lr()[0]}
            self.print(
                f"{steps}: maskgit loss: {logs['loss']} - lr: {self.lr_scheduler.get_last_lr()[0]}"
            )
            self.accelerator.log(logs, steps)
            if steps % self.save_model_every == 0:
                state_dict = self.accelerator.unwrap_model(self.model).state_dict()
                maskgit_save_name = (
                    "maskgit_superres" if self.model.cond_image_size else "maskgit"
                )
                file_name = (
                    f"{maskgit_save_name}.{steps}.pt"
                    if not self.only_save_last_checkpoint
                    else f"{maskgit_save_name}.pt"
                )

                model_path = str(self.results_dir / file_name)
                self.accelerator.save(state_dict, model_path)

                if self.use_ema:
                    ema_state_dict = self.accelerator.unwrap_model(
                        self.ema_model
                    ).state_dict()
                    file_name = (
                        f"{maskgit_save_name}.{steps}.ema.pt"
                        if not self.only_save_last_checkpoint
                        else f"{maskgit_save_name}.ema.pt"
                    )
                    model_path = str(self.results_dir / file_name)
                    self.accelerator.save(ema_state_dict, model_path)

                self.print(f"{steps}: saving model to {str(self.results_dir)}")
            if steps % self.save_results_every == 0:
                cond_image = None
                if self.model.cond_image_size:
                    self.print(
                        "With conditional image training, we recommend keeping the validation prompts to empty strings"
                    )
                    cond_image = F.interpolate(imgs[0], 256)

                self.log_validation_images(
                    self.validation_prompts, self.steps, cond_image=cond_image
                )
                self.print(f"{steps}: saving to {str(self.results_dir)}")

            return logs
