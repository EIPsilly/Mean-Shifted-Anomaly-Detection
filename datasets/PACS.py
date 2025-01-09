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
class_to_idx = config["PACS_class_to_idx"]
domain_to_idx = config["PACS_domain_to_idx"]

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


def augpacs(image, preprocess, severity=3, width=3, depth=-1, alpha=1.):
    aug_list = [
            autocontrast, equalize, posterize, solarize, color, contrast, brightness, sharpness
    ]
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

class PACS_Dataset_with_domain_label(Dataset):
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
            self.domain_labels[idx] = domain_to_idx[img_path.split("/")[2]]
            img = Image.open(config["PACS_root"] + img_path).convert('RGB')
            img = resize_transform(img)
            self.img_list.append(img)
        
        self.semi_domain_labels = self.domain_labels.copy()
        if "domain_label_ratio" in args:
            from sklearn.model_selection import train_test_split
            mask_set, unmask_set, mask_idx, unmask_idx = train_test_split(self.semi_domain_labels, np.arange(self.semi_domain_labels.shape[0]), test_size=max(3 / self.semi_domain_labels.shape[0], args.domain_label_ratio), random_state=42, stratify=self.semi_domain_labels)
            self.semi_domain_labels[mask_idx] = -1
        
        self.normal_idx = np.where(self.labels==0)[0]
        self.outlier_idx = np.where(self.labels==1)[0]

        self.domain_idx = []
        for i in range(args.domain_cnt):
            self.domain_idx.append(np.where(self.domain_labels == i)[0])
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self.img_list[idx]
        label = self.labels[idx]
        
        augimg = img

        if self.augment_transform is not None:
            augimg = augpacs(augimg, self.augment_transform)
            # augimg = self.augment_transform(augimg)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return idx, img, augimg, label, self.domain_labels[idx]
    
class PACS_Data():

    def __init__(self, args):
            
        normal_class = "".join(list(map(str,args.normal_class)))
        anomaly_class = "".join(list(map(str,args.anomaly_class)))
        if not "supervised" in args:
            args.supervised = "un"
        
        if args.domain_cnt == 3:
            train_path = f'../domain-generalization-for-anomaly-detection/data/pacs/{args.supervised}supervised/{args.domain_cnt}domain/20240412-PACS-{normal_class}-{anomaly_class}.npz'
        elif args.domain_cnt == 1:
            train_path = f'../domain-generalization-for-anomaly-detection/data/pacs/{args.supervised}supervised/{args.domain_cnt}domain/20241124-PACS-{normal_class}-{anomaly_class}.npz'

        if ("contamination_rate" in args == False) or (args.contamination_rate == 0):
            pass
        else:
            train_path = f'../domain-generalization-for-anomaly-detection/data/contamination/pacs/unsupervised/{args.domain_cnt}domain/20240412-PACS-{normal_class}-{anomaly_class}-{args.contamination_rate}.npz'

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
        
        self.train_data = PACS_Dataset_with_domain_label(args, data["train_set_path"], data["train_labels"], transform=train_transform, target_transform=None, augment_transform = augment_transform)
        unlabeled_idx = np.where(data["train_labels"] == 0)[0]
        self.unlabeled_data = PACS_Dataset_with_domain_label(args, data["train_set_path"][unlabeled_idx], data["train_labels"][unlabeled_idx], transform=train_transform, target_transform=None, augment_transform = augment_transform)
        self.val_data = PACS_Dataset_with_domain_label(args, data["val_set_path"], data["val_labels"], transform=train_transform, target_transform=None, augment_transform = augment_transform)

        logging.info("y_train\t" + str(dict(sorted(Counter(data["train_labels"]).items()))))
        logging.info("y_val\t" + str(dict(sorted(Counter(data["val_labels"]).items()))))
        print("y_train\t" + str(dict(sorted(Counter(data["train_labels"]).items()))))
        print("y_val\t" + str(dict(sorted(Counter(data["val_labels"]).items()))))
        self.test_dict = {}
        for domain in ["photo", "art_painting", "cartoon", "sketch"]:
            self.test_dict[domain] = PACS_Dataset_with_domain_label(args, data[f"test_{domain}"], data[f"test_{domain}_labels"], transform=test_transform, target_transform=None, augment_transform = augment_transform)
            logging.info(domain + "\ty_test\t" + str(dict(sorted(Counter(data[f"test_{domain}_labels"]).items()))))
            print(domain + "\ty_test\t" + str(dict(sorted(Counter(data[f"test_{domain}_labels"]).items()))))
            
        # for domain in os.listdir(f'{config["PACS_root"]}/test'):
        #     # if domain in self.in_domain_type:
        #     #     continue
        #     test_img_path_list = []
        #     test_label = []
        #     for _class_ in os.listdir(f'{config["PACS_root"]}/test/{domain}'):
        #         img_paths = glob.glob(f'{config["PACS_root"]}/test/{domain}/{_class_}/*.[jp][pn]g')
        #         img_paths = [path.split("PACS")[1] for path in img_paths]
        #         test_img_path_list += img_paths
        #         test_label += [int(class_to_idx[_class_] in args.anomaly_class)] * len(img_paths)
        #     logging.info(domain + "\ty_test\t" + str(dict(sorted(Counter(test_label).items()))))
        #     self.test_dict[domain] = PACS_Dataset_with_domain_label(test_img_path_list, test_label, transform=test_transform, target_transform=None, augment_transform = augment_transform)


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
    args.add_argument("--data_name", type=str, default="PACS")
    args.add_argument("--normal_class", nargs="+", type=int, default=[0,1,2,3])
    args.add_argument("--anomaly_class", nargs="+", type=int, default= [4,5,6])
    args.add_argument("--contamination_rate", type=float ,default=0)
    args.add_argument("--labeled_rate", type=float, default=0.02)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--train_binary", type=bool, default=True)
    args.add_argument("--in_domain_type", nargs="+", type=str, default=["photo", "art_painting", "cartoon"], choices=["art_painting", "cartoon", "photo", "sketch"])
    

    args = args.parse_args(["--data_name", "PACS",
                            "--normal_class", "0", "1", "2", "3",
                            "--anomaly_class", "4", "5", "6",
                            "--train_binary", "True",
                            "--in_domain_type", "photo", "art_painting", "cartoon",
                            ])

    data = PACS_Data(args)