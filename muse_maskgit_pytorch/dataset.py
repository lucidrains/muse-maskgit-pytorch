from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image, ImageFile
from pathlib import Path
from muse_maskgit_pytorch.t5 import MAX_LENGTH
import datasets
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageDataset(Dataset):
    def __init__(self, dataset, image_size, image_column="image"):
        super().__init__()
        self.dataset = dataset
        self.image_column = image_column
        self.transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(image_size),
                T.RandomHorizontalFlip(),
                T.CenterCrop(image_size),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image= self.dataset[index][self.image_column]
        return self.transform(image)
    
class ImageTextDataset(ImageDataset):
    def __init__(self, dataset, image_size, tokenizer, image_column="image", caption_column="caption"):
        super().__init__(dataset, image_size=image_size, image_column=image_column)
        self.caption_column = caption_column
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        image = self.dataset[index][self.image_column]
        descriptions = self.dataset[index][self.caption_column]
        text = random.choice(descriptions)
        
        if self.caption_column == None:
            text = ""
            
        encoded = self.tokenizer.batch_encode_plus(
            [text],
            return_tensors="pt",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )

        input_ids = encoded.input_ids
        attn_mask = encoded.attention_mask
        return self.transform(image), input_ids[0], attn_mask[0]

def get_dataset_from_dataroot(data_root, args):
    image_paths = list(Path(data_root).rglob("*.[jJ][pP][gG]"))
    random.shuffle(image_paths)
    data_dict = {args.image_column: [], args.caption_column: []}
    dataset = datasets.Dataset.from_dict(data_dict)
    for image_path in image_paths:
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        feature = datasets.Image(decode=False)
        
        caption_path = image_path.with_suffix(".txt")
        captions = caption_path.read_text().split('\n')
        captions = list(filter(lambda t: len(t) > 0, captions))
        
        dataset = dataset.add_item({args.image_column: feature.encode_example(image), args.caption_column: captions})

    dataset = dataset.cast_column(args.image_column, datasets.Image())
    return dataset

def split_dataset_into_dataloaders(dataset, valid_frac=0.05, seed=42, batch_size=1):
    if valid_frac > 0:
        train_size = int((1 - valid_frac) * len(dataset))
        valid_size = len(dataset) - train_size
        dataset, validation_dataset = random_split(dataset, [train_size, valid_size], generator = torch.Generator().manual_seed(seed))
        print(f'training with dataset of {len(dataset)} samples and validating with randomly splitted {len(validation_dataset)} samples')
    else:
        validation_dataset = dataset
        print(f'training with shared training and valid dataset of {len(dataset)} samples')
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size = batch_size,
        shuffle = True
    )
    return dataloader, validation_dataloader