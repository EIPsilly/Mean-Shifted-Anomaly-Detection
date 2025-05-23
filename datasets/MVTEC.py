import argparse
import random
from PIL import Image, ImageOps, ImageEnhance
import glob
import torch
import logging
import os
import numpy as np
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from collections import Counter

with open("../domain-generalization-for-anomaly-detection/config.yml", 'r', encoding="utf-8") as f:
    import yaml
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
domain_to_idx = config["MVTEC_domain_to_idx"]
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
        level: Level of the operation that will be between [0, `PARAMETER_MAX`].
        maxval: Maximum value that the operation can have. This will be scaled to
            level/PARAMETER_MAX.
    Returns:
        An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.
    Args:
        level: Level of the operation that will be between [0, `PARAMETER_MAX`].
        maxval: Maximum value that the operation can have. This will be scaled to
            level/PARAMETER_MAX.
    Returns:
        A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.Resampling.BILINEAR)


def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)

# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


def augmvtec(image, preprocess, severity=3, width=3, depth=-1, alpha=1.):
    aug_list = [ autocontrast, equalize, posterize, solarize, color, sharpness]
    severity = random.randint(0, severity)

    ws = np.float32(np.random.dirichlet([1] * width))
    m = np.float32(np.random.beta(alpha, alpha))
    preprocess_img = preprocess(image)
    mix = torch.zeros_like(preprocess_img)
    for i in range(width):
        image_aug = image.copy()
        depth = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess_img + m * mix
    return mixed


class MVTEC_Dataset(Dataset):
    def __init__(self, args, x, y, transform=None, target_transform=None, augment_transform = None):
        self.image_paths = x
        self.labels = y
        self.transform = transform
        self.target_transform = target_transform
        self.augment_transform = augment_transform
        
        self.img_list = []

        if "input_img_size" in args:
            image_size = args.input_img_size
        else:
            image_size = 256
        resize_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            ])
        self.domain_labels = np.empty_like(self.labels)

        for idx, img_path in enumerate(self.image_paths):
            self.domain_labels[idx] = domain_to_idx[img_path.split("/")[1].replace("mvtec_", "")]
            img = Image.open(config["mvtec_ood_root"] + img_path).convert('RGB')
            img = resize_transform(img)
            self.img_list.append(img)
        
        self.normal_idx = np.where(self.labels==0)[0]
        self.outlier_idx = np.where(self.labels==1)[0]

        self.domain_idx = []
        for i in range(args.domain_cnt):
            self.domain_idx.append(np.where(self.domain_labels == i)[0])
        self.args = args
        self.gray_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean,
                                    std=std),
                transforms.Grayscale(3)
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self.img_list[idx]
        label = self.labels[idx]
        
        augimg = img

        if self.augment_transform is not None:
            augimg = augmvtec(augimg, self.augment_transform)
            # augimg = self.augment_transform(augimg)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if ("gray" in self.args) and (self.args.gray == 1):
            gray_img = self.gray_transform(self.img_list[idx])
            return idx, img, augimg, gray_img, label, self.domain_labels[idx]
        else:
            return idx, img, augimg, label, self.domain_labels[idx]
    
class MVTEC_Data():

    def __init__(self, args):
        if not "supervised" in args:
            args.supervised = "un"
            
        train_path = f'../domain-generalization-for-anomaly-detection/data/mvtec/{args.supervised}supervised/{args.domain_cnt}domain/20240412-MVTEC-{args.checkitew}.npz'
        data = np.load(train_path, allow_pickle=True)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        if "input_img_size" in args:
            image_size = args.input_img_size
        else:
            image_size = 256

        train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
        ])
        
        augment_transform = train_transform
        
        test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop(image_size),
                transforms.Normalize(
                        mean=torch.tensor(mean),
                        std=torch.tensor(std))
        ])
        
        self.train_data = MVTEC_Dataset(args, data["train_set_path"], data["train_labels"], transform=train_transform, target_transform=None, augment_transform = augment_transform)
        unlabeled_idx = np.where(data["train_labels"] == 0)[0]
        self.unlabeled_data = MVTEC_Dataset(args, data["train_set_path"][unlabeled_idx], data["train_labels"][unlabeled_idx], transform=train_transform, target_transform=None, augment_transform = augment_transform)
        self.val_data = MVTEC_Dataset(args, data["val_set_path"], data["val_labels"], transform=train_transform, target_transform=None, augment_transform = augment_transform)

        logging.info("y_train\t" + str(dict(sorted(Counter(data["train_labels"]).items()))))
        logging.info("y_val\t" + str(dict(sorted(Counter(data["val_labels"]).items()))))
        print("y_train\t" + str(dict(sorted(Counter(data["train_labels"]).items()))))
        print("y_val\t" + str(dict(sorted(Counter(data["val_labels"]).items()))))
        self.test_dict = {}
        for domain in ["origin", "brightness", "contrast", "defocus_blur", "gaussian_noise"]:
            self.test_dict[domain] = MVTEC_Dataset(args, data[f"test_{domain}"], data[f"test_{domain}_labels"], transform=test_transform, target_transform=None, augment_transform = augment_transform)
            logging.info(domain + "\ty_test\t" + str(dict(sorted(Counter(data[f"test_{domain}_labels"]).items()))))
            print(domain + "\ty_test\t" + str(dict(sorted(Counter(data[f"test_{domain}_labels"]).items()))))
            
        
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 8, drop_last_train = True):

        self.train_loader = DataLoader(dataset=self.train_data, batch_size=batch_size, shuffle=shuffle_train,
                                                            num_workers=num_workers, drop_last=drop_last_train)

        self.val_loader = DataLoader(dataset=self.val_data, batch_size=batch_size, shuffle=shuffle_train,
                                                            num_workers=num_workers, drop_last=drop_last_train)

        self.test_loader_dict = {}
        for domain_type, test_set in self.test_dict.items():
            self.test_loader_dict[domain_type] = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers, drop_last=False)
        
        self.unlabeled_loader = DataLoader(dataset=self.unlabeled_data, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers, drop_last=False)

        return self.train_loader, self.val_loader, self.test_loader_dict
    
    

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:    %(message)s', datefmt='%Y-%m-%d %H:%M:%S ')
    logging.getLogger().setLevel(logging.INFO)

    args = argparse.ArgumentParser()
    args.add_argument("--data_name", type=str, default="MVTEC")
    args.add_argument("--checkitew", type=str, default="bottle")
    args.add_argument("--domain_cnt", type=int, default=4)

    args.add_argument("--contamination_rate", type=float ,default=0)
    args.add_argument("--labeled_rate", type=float, default=0.02)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--train_binary", type=bool, default=True)
    args.add_argument("--in_domain_type", nargs="+", type=str, default=["origin", "brightness", "contrast", "defocus_blur"], choices=["origin", "brightness", "contrast", "defocus_blur", "gaussian_noise"])
    args = args.parse_args()
    

    data = MVTEC_Data(args)

    train_loader, val_loader, test_loader_dict = data.loaders(batch_size = 16)

    label_list = []
    for batch in train_loader:
        idx, img, aug, label = batch
        label_list.append(label)
    
    logging.info("train_loader\t" + str(dict(sorted(Counter(np.concatenate(label_list)).items()))))
    

    label_list = []
    for batch in val_loader:
        idx, img, aug, label = batch
        label_list.append(label)
    
    logging.info("val_loader\t" + str(dict(sorted(Counter(np.concatenate(label_list)).items()))))
    

    for domain_type, test_loader in test_loader_dict.items():
        label_list = []
        for batch in test_loader:
            idx, img, aug, label = batch
            label_list.append(label)
        
        logging.info(domain_type + "\ttest_loader\t" + str(dict(sorted(Counter(np.concatenate(label_list)).items()))))