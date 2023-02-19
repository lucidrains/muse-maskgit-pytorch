import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from pathlib import Path
from datasets import load_dataset
import os
from muse_maskgit_pytorch import (
    VQGanVAE,
    VQGanVAETrainer,
    MaskGit,
    MaskGitTransformer,
    Muse,
)
from accelerate import Accelerator


import argparse

def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precisoin",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Precision to train on."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Path to save the training samples and checkpoints",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="results/logs",
        help="Path to log the losses and LR",
    )

    # vae_trainer args
    parser.add_argument(
        "--resume_from",
        type=str,
        default="",
        help="Path to the vae model. eg. 'results/vae.steps.pt'",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default="data/datasets/INE/data",
        help="Dataset folder where your input images for training are.",
    )
    parser.add_argument(
        "--num_train_steps",
        type=int,
        default=50000,
        help="Total number of steps to train for. eg. 50000.",
    )
    parser.add_argument("--dim", type=int, default=128, help="Model dimension.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch Size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning Rate.")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1, help="Gradient Accumulation."
    )
    parser.add_argument(
        "--save_results_every",
        type=int,
        default=100,
        help="Save results every this number of steps.",
    )
    parser.add_argument(
        "--save_model_every",
        type=int,
        default=500,
        help="Save the model every this number of steps.",
    )
    parser.add_argument("--vq_codebook_size", type=int, default=256, help="Image Size.")
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Image size. You may want to start with small images, and then curriculum learn to larger ones, but because the vae is all convolution, it should generalize to 512 (as in paper) without training on it",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]',
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )

    # base args
    parser.add_argument(
        "--base_texts",
        type=list,
        default=[
            "a whale breaching from afar",
            "young girl blowing out candles on her birthday cake",
            "fireworks with blue and green sparkles",
            "waking up to a psychedelic landscape",
        ],
        help="List of Prompts to use.",
    )
    parser.add_argument(
        "--base_resume_from",
        type=str,
        default="",
        help="Path to the vae model. eg. 'results/vae.steps.pt'",
    )
    parser.add_argument(
        "--base_num_tokens",
        type=int,
        default=256,
        help="must be same as vq_codebook_size.",
    )
    parser.add_argument(
        "--base_seq_len",
        type=int,
        default=1024,
        help="must be equivalent to fmap_size ** 2 in vae.",
    )
    parser.add_argument("--base_dim", type=int, default=128, help="Model dimension.")
    parser.add_argument("--base_depth", type=int, default=2, help="Depth.")
    parser.add_argument(
        "--base_dim_head", type=int, default=64, help="Attention head dimension."
    )
    parser.add_argument("--base_heads", type=int, default=8, help="Attention heads.")
    parser.add_argument(
        "--base_ff_mult", type=int, default=4, help="Feedforward expansion factor"
    )
    parser.add_argument(
        "--base_t5_name", type=str, default="t5-small", help="Name of your T5 model."
    )
    parser.add_argument("--base_vq_codebook_size", type=int, default=256, help="")
    parser.add_argument("--base_image_size", type=int, default=512, help="")
    parser.add_argument(
        "--base_cond_drop_prob",
        type=float,
        default=0.25,
        help="Conditional dropout, for Classifier Free Guidance",
    )
    parser.add_argument(
        "--base_cond_scale",
        type=int,
        default=3,
        help="Conditional for Classifier Free Guidance",
    )
    parser.add_argument(
        "--base_timesteps",
        type=int,
        default=20,
        help="Time Steps to use for the generation.",
    )

    # superres args
    parser.add_argument(
        "--superres_texts",
        type=list,
        default=[
            "a whale breaching from afar",
            "young girl blowing out candles on her birthday cake",
            "fireworks with blue and green sparkles",
            "waking up to a psychedelic landscape",
        ],
        help="List of Prompts to use.",
    )
    parser.add_argument(
        "--superres_resume_from",
        type=str,
        default="",
        help="Path to the vae model. eg. 'results/vae.steps.pt'",
    )
    parser.add_argument(
        "--superres_num_tokens",
        type=int,
        default=256,
        help="must be same as vq_codebook_size.",
    )
    parser.add_argument(
        "--superres_seq_len",
        type=int,
        default=1024,
        help="must be equivalent to fmap_size ** 2 in vae.",
    )
    parser.add_argument(
        "--superres_dim", type=int, default=128, help="Model dimension."
    )
    parser.add_argument("--superres_depth", type=int, default=2, help="Depth.")
    parser.add_argument(
        "--superres_dim_head", type=int, default=64, help="Attention head dimension."
    )
    parser.add_argument(
        "--superres_heads", type=int, default=8, help="Attention heads."
    )
    parser.add_argument(
        "--superres_ff_mult", type=int, default=4, help="Feedforward expansion factor"
    )
    parser.add_argument(
        "--superres_t5_name", type=str, default="t5-small", help="name of your T5"
    )
    parser.add_argument("--superres_vq_codebook_size", type=int, default=256, help="")
    parser.add_argument("--superres_image_size", type=int, default=512, help="")
    parser.add_argument(
        "--superres_timesteps",
        type=int,
        default=20,
        help="Time Steps to use for the generation.",
    )

    # generate args
    parser.add_argument(
        "--prompt",
        type=list,
        default=[
            "a whale breaching from afar",
            "young girl blowing out candles on her birthday cake",
            "fireworks with blue and green sparkles",
            "waking up to a psychedelic landscape",
        ],
        help="List of Prompts to use for the generation.",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="",
        help="Path to the base vae model. eg. 'results/vae.steps.base.pt'",
    )
    parser.add_argument(
        "--superres_maskgit",
        type=str,
        default="",
        help="Path to the superres vae model. eg. 'results/vae.steps.superres.pt'",
    )
    parser.add_argument(
        "--generate_timesteps",
        type=int,
        default=20,
        help="Time Steps to use for the generation.",
    )
    parser.add_argument(
        "--generate_cond_scale",
        type=int,
        default=3,
        help="Conditional for Classifier Free Guidance",
    )

    # Parse the argument
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=args.logging_dir,
    )
    dataset = ImageDataset(args.folder, args.image_size)
    vae = VQGanVAE(dim=args.dim, vq_codebook_size=args.vq_codebook_size)

    trainer = VQGanVAETrainer(
        vae,
        folder=args.data_folder,
        current_step=current_step,
        num_train_steps=args.num_train_steps,
        batch_size=args.batch_size,
        image_size=args.image_size,  # you may want to start with small images, and then curriculum learn to larger ones, but because the vae is all convolution, it should generalize to 512 (as in paper) without training on it
        lr=args.lr,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=None,
        discr_max_grad_norm=None,
        save_results_every=args.save_results_every,
        save_model_every=args.save_model_every,
        results_dir=args.results_dir,
        logging_dir=args.logging_dir,
        valid_frac=0.05,
        random_split_seed=42,
        use_ema=True,
        ema_beta=0.995,
        ema_update_after_step=1,
        ema_update_every=1,
        apply_grad_penalty_every=4,
        accelerate_kwargs={
            'mixed_precision': args.mixed_precisionWW
        },
    )

    trainer.train()



if __name__ == "__main__":
    main()