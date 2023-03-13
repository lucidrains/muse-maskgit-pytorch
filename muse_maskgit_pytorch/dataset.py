from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import ImageFile
from pathlib import Path
from muse_maskgit_pytorch.t5 import MAX_LENGTH
import datasets
from datasets import Image, load_from_disk
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os, time, sys
from tqdm import tqdm
from threading import Thread

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset(Dataset):
    def __init__(
        self, dataset, image_size, image_column="image", flip=True, center_crop=True
    ):
        super().__init__()
        self.dataset = dataset
        self.image_column = image_column
        transform_list = [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize(image_size),
        ]
        if flip:
            transform_list.append(T.RandomHorizontalFlip())
        if center_crop:
            transform_list.append(T.CenterCrop(image_size))
        transform_list.append(T.ToTensor())
        self.transform = T.Compose(transform_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][self.image_column]
        return self.transform(image)-0.5


class ImageTextDataset(ImageDataset):
    def __init__(
        self,
        dataset,
        image_size,
        tokenizer,
        image_column="image",
        caption_column=None,
        flip=True,
        center_crop=True,
    ):
        super().__init__(
            dataset,
            image_size=image_size,
            image_column=image_column,
            flip=flip,
            center_crop=center_crop,
        )
        self.caption_column = caption_column
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        image = self.dataset[index][self.image_column]
        descriptions = self.dataset[index][self.caption_column]
        if self.caption_column == None or descriptions == None:
            text = ""
        elif isinstance(descriptions, list):
            if len(descriptions) == 0:
                text = ""
            else:
                text = random.choice(descriptions)
        else:
            text = descriptions
        # max length from the paper
        encoded = self.tokenizer.batch_encode_plus(
            [str(text)],
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
        )

        input_ids = encoded.input_ids
        attn_mask = encoded.attention_mask
        return self.transform(image), input_ids[0], attn_mask[0]

def get_directory_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

def save_dataset_with_progress(dataset, save_path):
    # Estimate the total size of the dataset in bytes
    total_size = sys.getsizeof(dataset)

    # Start saving the dataset in a separate thread
    save_thread = Thread(target=dataset.save_to_disk, args=(save_path,))
    save_thread.start()

    # Create a tqdm progress bar and update it periodically
    with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
        while save_thread.is_alive():
            if os.path.exists(save_path):
                size = get_directory_size(save_path)
                # Update the progress bar based on the current size of the saved file
                pbar.update(size - pbar.n)  # Update by the difference between current and previous size
            time.sleep(1)

def get_dataset_from_dataroot(
    data_root, image_column="image", caption_column="caption", save_path="dataset"
):
    # Check if data_root is a symlink and resolve it to its target location if it is
    if os.path.islink(data_root):
        data_root = os.path.realpath(data_root)

    if os.path.exists(save_path):
        # if the data_root folder is newer than the save_path we reload the
        if os.stat(save_path).st_mtime - os.stat(data_root).st_mtime > 1:
            return load_from_disk(save_path)
        else:
            print ("The data_root folder has being updated recently. Removing previously saved dataset and updating it.")
            os.removedirs(save_path)
    
    
    extensions = ["jpg", "jpeg", "png", "webp"]
    image_paths = []
    
    for ext in extensions:
        image_paths.extend(list(Path(data_root).rglob(f"*.{ext}")))    
    
    random.shuffle(image_paths)
    data_dict = {image_column: [], caption_column: []}
    for image_path in tqdm(image_paths):
        # check image size and ignore images with 0 byte.
        if os.path.getsize(image_path) == 0:
            continue
        caption_path = image_path.with_suffix(".txt")
        if os.path.exists(str(caption_path)):
            captions = caption_path.read_text(encoding="utf-8").split("\n")
            captions = list(filter(lambda t: len(t) > 0, captions))
        else:
            captions = []
        image_path = str(image_path)
        data_dict[image_column].append(image_path)
        data_dict[caption_column].append(captions)
    dataset = datasets.Dataset.from_dict(data_dict)
    dataset = dataset.cast_column(image_column, Image())
    #dataset.save_to_disk(save_path)
    save_dataset_with_progress(dataset, save_path)
    return dataset


def split_dataset_into_dataloaders(dataset, valid_frac=0.05, seed=42, batch_size=1):
    if valid_frac > 0:
        train_size = int((1 - valid_frac) * len(dataset))
        valid_size = len(dataset) - train_size
        dataset, validation_dataset = random_split(
            dataset,
            [train_size, valid_size],
            generator=torch.Generator().manual_seed(seed),
        )
        print(
            f"training with dataset of {len(dataset)} samples and validating with randomly splitted {len(validation_dataset)} samples"
        )
    else:
        validation_dataset = dataset
        print(
            f"training with shared training and valid dataset of {len(dataset)} samples"
        )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    validation_dataloader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=True
    )
    return dataloader, validation_dataloader
