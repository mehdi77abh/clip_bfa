import torch
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from torchvision import datasets, transforms
from torch.utils.data import random_split
import os
import numpy as np
from PIL import Image
import os.path
import pickle
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

class my_ImageDataset(datasets.DatasetFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = datasets.folder.default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        processor: Optional[Callable] = None,
        use_ir = False,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.processor = processor
        
        self.use_ir = use_ir
        self.normalize = transforms.Normalize(mean=self.processor.image_mean,std=self.processor.image_std)
        
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.processor is not None:
            sample = self.processor(images=sample, return_tensors="pt", do_normalize=False)['pixel_values'].squeeze(0)
            if not self.use_ir:
                sample = self.normalize(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

def load_split_ImageNet1k_valid(valdir, aux_num=512, seed=0, processor=None, batch_size=64, shuffle=False, num_workers=1, pin_memory=False,split_ratio=0.,use_ir=False,use_defalut_valid_batchsize=False):
    val_dataset = my_ImageDataset(valdir, processor=processor, use_ir=use_ir)
    total = len(val_dataset)
    val_dataset, aux_dataset = random_split(
        dataset=val_dataset,
        lengths=[total-aux_num, aux_num],
        generator=torch.Generator().manual_seed(seed)
    )
    valid_batch_size = 128 if use_defalut_valid_batchsize else batch_size
    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=valid_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    aux_loader = torch.utils.data.DataLoader(
            aux_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    if split_ratio == 0.:
        return val_loader, aux_loader
    elif split_ratio > 0. and split_ratio <= 1.:
        total = len(val_dataset)
        small_val_dataset, _ = random_split(
            dataset=val_dataset,
            lengths=[int(total*split_ratio), total-int(total*split_ratio)],
            generator=torch.Generator().manual_seed(seed)
        )
        small_val_loader = torch.utils.data.DataLoader(
            small_val_dataset,
            batch_size=valid_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        return val_loader, aux_loader, small_val_loader
    else:
        raise ValueError
    
class my_cifar10(datasets.VisionDataset):
    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        processor=None,
        use_ir=False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        
        self.processor = processor
        self.use_ir = use_ir
        self.normalize = transforms.Normalize(mean=self.processor.image_mean,std=self.processor.image_std)
        
        self.datair = self.processor(images=self.data, return_tensors="pt", do_normalize=False)['pixel_values']
        self.data = self.normalize(self.datair)

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.datair[index] if self.use_ir else self.data[index], self.targets[index]
    

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        for filename, md5 in self.train_list + self.test_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

def load_split_cifar10_valid(valdir, aux_num=512, seed=0, processor=None, batch_size=64, shuffle=False, num_workers=1, pin_memory=False,use_ir=False,use_defalut_valid_batchsize=False,target_transform=None):
    val_dataset = my_cifar10(valdir, train=False, processor=processor, use_ir=use_ir, target_transform=target_transform)
    total = len(val_dataset)
    val_dataset, aux_dataset = random_split(
        dataset=val_dataset,
        lengths=[total-aux_num, aux_num],
        generator=torch.Generator().manual_seed(seed)
    )
    valid_batch_size = 256 if use_defalut_valid_batchsize else batch_size
    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=valid_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    aux_loader = torch.utils.data.DataLoader(
            aux_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    return val_loader, aux_loader