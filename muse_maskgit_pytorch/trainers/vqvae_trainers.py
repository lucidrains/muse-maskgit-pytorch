
from pathlib import Path
from shutil import rmtree
from datetime import datetime

from beartype import beartype
from PIL import Image
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
import numpy as np
from muse_maskgit_pytorch.trainers.base_accelerated_trainer import BaseAcceleratedTrainer
from diffusers.optimization import get_scheduler

def noop(*args, **kwargs):
    pass

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.0)
        log[key] = old_value + new_value
    return log
def exists(val):
    return val is not None

class VQGanVAETrainer(BaseAcceleratedTrainer):
    def __init__(
        self,
        vae: VQGanVAE,
        dataloader,
        valid_dataloader,
        accelerator,
        *,
        current_step,
        num_train_steps,
        gradient_accumulation_steps=1,
        max_grad_norm=None,
        save_results_every=100,
        save_model_every=1000,
        results_dir="./results",
        logging_dir="./results/logs",
        apply_grad_penalty_every=4,
        lr=3e-4,
        lr_scheduler='constant',
        lr_warmup_steps= 500,           
        discr_max_grad_norm=None,
        use_ema=True,
        ema_beta=0.995,
        ema_update_after_step=0,
        ema_update_every=1,
        clear_previous_experiments=False
    ):
        super().__init__(dataloader, valid_dataloader, accelerator, current_step=current_step, num_train_steps=num_train_steps,\
                        gradient_accumulation_steps=gradient_accumulation_steps, max_grad_norm=max_grad_norm, save_results_every=save_results_every, \
                        save_model_every=save_model_every, results_dir=results_dir, logging_dir=logging_dir, apply_grad_penalty_every=apply_grad_penalty_every,\
                        clear_previous_experiments=clear_previous_experiments)

        # vae
        self.model = vae

        all_parameters = set(vae.parameters())
        discr_parameters = set(vae.discr.parameters())
        vae_parameters = all_parameters - discr_parameters

        # optimizers
        self.optim = Adam(vae_parameters, lr=lr)
        self.discr_optim = Adam(discr_parameters, lr=lr)
        
        self.lr_scheduler_optim = get_scheduler(
         lr_scheduler,
                optimizer=self.optim,
                num_warmup_steps=lr_warmup_steps * self.gradient_accumulation_steps,
                num_training_steps=self.num_train_steps * self.gradient_accumulation_steps,
        )
    
        self.lr_scheduler_discr_optim = get_scheduler(
            lr_scheduler,
            optimizer=self.discr_optim,
            num_warmup_steps=lr_warmup_steps * self.gradient_accumulation_steps,
            num_training_steps=self.num_train_steps * self.gradient_accumulation_steps,
        )        

        self.discr_max_grad_norm = discr_max_grad_norm

        # prepare with accelerator

        (
            self.model,
            self.optim,
            self.discr_optim,
            self.dl,
            self.valid_dl,
        ) = self.prepare(
            self.model, self.optim, self.discr_optim, self.dl, self.valid_dl
        )
        self.model.train()

        self.use_ema = use_ema

        if use_ema:
            self.ema_model = EMA(
                vae,
                update_after_step=ema_update_after_step,
                update_every=ema_update_every,
            )
            self.ema_model = self.prepare(self.ema_model)

    def load(self, path):
        pkg = super().load(path)
        self.discr_optim.load_state_dict(pkg["discr_optim"])
    def save(self, path):
        if not self.is_local_main_process:
            return

        pkg = dict(
            model=self.get_state_dict(self.model),
            optim=self.optim.state_dict(),
            discr_optim=self.discr_optim.state_dict(),
        )
        torch.save(pkg, path)
    def log_validation_images(self, models_to_evaluate, logs, steps):
        log_imgs = []
        for model, filename in models_to_evaluate:
            model.eval()

            valid_data = next(self.valid_dl_iter)
            valid_data = valid_data.to(self.device)

            recons = model(valid_data, return_recons = True)

            # else save a grid of images

            imgs_and_recons = torch.stack((valid_data, recons), dim = 0)
            imgs_and_recons = rearrange(imgs_and_recons, 'r b ... -> (b r) ...')

            imgs_and_recons = imgs_and_recons.detach().cpu().float().clamp(0., 1.)
            grid = make_grid(imgs_and_recons, nrow = 2, normalize = True, value_range = (0, 1))

            logs['reconstructions'] = grid
            save_file = str(self.results_dir / f'{filename}.png')
            save_image(grid, save_file)
            log_imgs.append(np.asarray(Image.open(save_file)))
        super().log_validation_images(log_imgs, steps)


    def train_step(self):
        device = self.device

        steps = int(self.steps.item())
        apply_grad_penalty = (steps % self.apply_grad_penalty_every) == 0

        self.model.train()
        discr = self.model.module.discr if self.is_distributed else self.model.discr
        if self.use_ema:
            ema_model = self.ema_model.module if self.is_distributed else self.ema_model

        # logs

        logs = {}

        # update vae (generator)

        for _ in range(self.gradient_accumulation_steps):
            img = next(self.dl_iter)
            img = img.to(device)

            with self.accelerator.autocast():
                loss = self.model(
                    img,
                    add_gradient_penalty = apply_grad_penalty,
                    return_loss = True
                )

            self.accelerator.backward(loss / self.gradient_accumulation_steps)

            accum_log(logs, {'Train/vae_loss': loss.item() / self.gradient_accumulation_steps})

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        self.lr_scheduler_optim.step()
        self.lr_scheduler_discr_optim.step()
        self.optim.step()
        self.optim.zero_grad()

        # update discriminator

        if exists(discr):
            self.discr_optim.zero_grad()

            for _ in range(self.gradient_accumulation_steps):
                img = next(self.dl_iter)
                img = img.to(device)

                loss = self.model(img, return_discr_loss = True)

                self.accelerator.backward(loss / self.gradient_accumulation_steps)

                accum_log(logs, {'Train/discr_loss': loss.item() / self.gradient_accumulation_steps})

            if exists(self.discr_max_grad_norm):
                self.accelerator.clip_grad_norm_(discr.parameters(), self.discr_max_grad_norm)

            self.discr_optim.step()

        # log
        
        accum_log(logs, {'lr': self.lr_scheduler_optim.get_last_lr()[0]})

        # self.print(f"{steps}: vae loss: {logs['Train/vae_loss']} - discr loss: {logs['Train/discr_loss']}")
        self.print(f"{steps}: vae loss: {logs['Train/vae_loss']} - discr loss: {logs['Train/discr_loss']} - lr: {self.lr_scheduler_optim.get_last_lr()[0]}")
        self.accelerator.log(logs, step=steps)

        # update exponential moving averaged generator

        if self.use_ema:
            ema_model.update()

        # sample results every so often

        if (steps % self.save_results_every) == 0:
            vaes_to_evaluate = ((self.model, str(steps)),)

            if self.use_ema:
                vaes_to_evaluate = ((ema_model.ema_model, f'{steps}.ema'),) + vaes_to_evaluate

            self.log_validation_images(vaes_to_evaluate, logs, steps)
            self.print(f'{steps}: saving to {str(self.results_dir)}')


        # save model every so often
        self.accelerator.wait_for_everyone()
        if self.is_main and (steps % self.save_model_every) == 0:
            state_dict = self.accelerator.unwrap_model(self.model).state_dict()
            model_path = str(self.results_dir / f'vae.{steps}.pt')
            self.accelerator.save(state_dict, model_path)

            if self.use_ema:
                ema_state_dict = self.accelerator.unwrap_model(self.ema_model).state_dict()
                model_path = str(self.results_dir / f'vae.{steps}.ema.pt')
                self.accelerator.save(ema_state_dict, model_path)

            self.print(f'{steps}: saving model to {str(self.results_dir)}')

        self.steps += 1
        return logs
