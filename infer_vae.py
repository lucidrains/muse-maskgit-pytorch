import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from pathlib import Path
from datasets import load_dataset
import os
from muse_maskgit_pytorch import (
    VQGanVAE,
    VQGanVAETaming,
    MaskGitTrainer,
    MaskGit,
    MaskGitTransformer,
    get_accelerator,
)
from muse_maskgit_pytorch.dataset import (
    get_dataset_from_dataroot,
    ImageDataset,
    split_dataset_into_dataloaders,
)

import argparse


def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no_center_crop",
        action="store_true",
        help="Don't do center crop.",
    )
    parser.add_argument(
        "--no_flip",
        action="store_true",
        help="Don't flip image.",
    )
    parser.add_argument(
        "--dataset_save_path",
        type=str,
        default="dataset",
        help="Path to save the dataset if you are making one from a directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument(
        "--valid_frac", type=float, default=0.05, help="validation fraction."
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Precision to train on.",
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
        "--vae_path",
        type=str,
        default=None,
        help="Path to the vae model. eg. 'results/vae.steps.pt'",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Name of the huggingface dataset used.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="Dataset folder where your input images for training are.",
    )
    parser.add_argument(
        "--num_train_steps",
        type=int,
        default=50000,
        help="Total number of steps to train for. eg. 50000.",
    )
    parser.add_argument("--dim", type=int, default=128, help="Model dimension.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch Size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate.")
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
    parser.add_argument('--taming_model_path', type=str, default = None,
                        help='path to your trained VQGAN weights. This should be a .ckpt file. (only valid when taming option is enabled)')

    parser.add_argument('--taming_config_path', type=str, default = None,
                        help='path to your trained VQGAN config. This should be a .yaml file. (only valid when taming option is enabled)')
    # Parse the argument
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = get_accelerator(
        mixed_precision=args.mixed_precision,
        logging_dir=args.logging_dir,
    )

    if args.train_data_dir:
        dataset = get_dataset_from_dataroot(
            args.train_data_dir,
            image_column=args.image_column,
            caption_column=args.caption_column,
            save_path=args.dataset_save_path,
        )
    elif args.dataset_name:
        dataset = load_dataset(args.dataset_name)["train"]
    if args.vae_path and args.taming_model_path:
        raise Exception("You can't pass vae_path and taming args at the same time.")

    if args.vae_path:
        accelerator.print("Loading Muse VQGanVAE")
        vae = VQGanVAE(dim=args.dim, vq_codebook_size=args.vq_codebook_size).to(
            accelerator.device
        )

        accelerator.print("Resuming VAE from: ", args.vae_path)
        vae.load(
            args.vae_path
        )  # you will want to load the exponentially moving averaged VAE

    elif args.taming_model_path:
        print("Loading Taming VQGanVAE")
        vae = VQGanVAETaming(vqgan_model_path=args.taming_model_path, vqgan_config_path=args.taming_config_path)
        args.num_tokens = vae.codebook_size
        args.seq_len = vae.get_encoded_fmap_size(args.image_size) ** 2
    vae = vae.to(accelerator.device)
    # then you plug the vae and transformer into your MaskGit as so

    dataset = ImageDataset(
        dataset,
        args.image_size,
        image_column=args.image_column,
        center_crop=not args.no_center_crop,
        flip=not args.no_flip,
    )
    save_image(dataset[0], "input.png")
    _, ids, _ = vae.encode(dataset[0][None].to(accelerator.device))
    recon = vae.decode_from_ids(ids)
    save_image(dataset[0], "output.png")



if __name__ == "__main__":
    main()