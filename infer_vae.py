import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from pathlib import Path
from datasets import load_dataset
import os, random
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
        "--random_image",
        action="store_true",
        help="Get a random image from the dataset to use for the reconstruction.",
    )
    parser.add_argument(
        "--dataset_save_path",
        type=str,
        default="dataset",
        help="Path to save the dataset if you are making one from a directory",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for reproducibility. If set to -1 a random seed will be generated.")
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
        "--caption_column",
        type=str,
        default="caption",
        help="The column of the dataset containing a caption or a list of captions.",
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

def seed_to_int(s):
    if type(s) is int:
        return s
    if s is None or s == '':
        return random.randint(0, 2**32 - 1)

    if ',' in s:
        s = s.split(',')

    if type(s) is list:
        seed_list = []
        for seed in s:
            if seed is None or seed == '':
                seed_list.append(random.randint(0, 2**32 - 1))
            else:
                seed_list = s

        return seed_list

    n = abs(int(s) if s.isdigit() else random.Random(s).randint(0, 2**32 - 1))
    while n >= 2**32:
        n = n >> 32
    return n

def main():
    args = parse_args()
    accelerator = get_accelerator(
        mixed_precision=args.mixed_precision,
        logging_dir=args.logging_dir,
    )

    # set pytorch seed for reproducibility 
    torch.manual_seed(seed_to_int(args.seed))    

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
    
    image_id = 0 if not args.random_image else random.randint(0, len(dataset))
   
    os.makedirs(args.results_dir, exist_ok=True)

    save_image(dataset[image_id], f"{args.results_dir}/input.png")
    
    _, ids, _ = vae.encode(dataset[image_id][None].to(accelerator.device))
    recon = vae.decode_from_ids(ids)
    save_image(recon, f"{args.results_dir}/output.png")

if __name__ == "__main__":
    main()