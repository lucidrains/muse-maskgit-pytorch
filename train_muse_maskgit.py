import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from pathlib import Path

import os
from muse_maskgit_pytorch import (
    VQGanVAE,
    VQGanVAETrainer,
    MaskGit,
    MaskGitTransformer,
    Muse,
)

import argparse

def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_
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



if __name__ == "__main__":
    main()