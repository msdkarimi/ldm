import json
import os

from torch.utils.data import Dataset
import torch
import random
from PIL import Image, ImageOps
from ast import literal_eval


class DareDataset(Dataset):
    def __init__(self, root_path, mode, transform=None):
        self.root_path = root_path
        self.mode = mode
        self.transform = transform
        self.dataset = self.read_dataset()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        _raw = self.dataset[idx].split('|')
        _file_name = str(_raw[0]).zfill(12) + '.jpg'
        _captions = literal_eval(_raw[-1])
        # _caption_idx = random.randint(0, len(literal_eval(_captions[-1])) - 1)
        _caption_idx = random.randint(0, len(_captions) - 1)
        caption = _captions[_caption_idx]
        image = Image.open(os.path.join(self.root_path, self.mode, _file_name)).convert('RGB')
        image = ImageOps.exif_transpose(image)
        image = self.transform(image) if self.transform is not None else image
        return {
            'image': image,
            'caption': str(caption),
            'image_name': _file_name
        }

    def read_dataset(self):
        _dataset = []
        with open(os.path.join(self.root_path, f'{self.mode}.txt'), 'r') as f:
            for line in f:
                _dataset.append(line)
        return _dataset
