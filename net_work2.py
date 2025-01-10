import torch
import faiss
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torchvision.models as models
import torch.nn.functional as F
from PIL import ImageFilter
import random
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


transform_color = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_resnet18 = transforms.Compose([
    transforms.Resize(224, interpolation=BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


moco_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class Transform:
    def __init__(self):
        self.moco_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __call__(self, x):
        x_1 = self.moco_transform(x)
        x_2 = self.moco_transform(x)
        return x_1, x_2

def mixstyle_test(input_image, normal_image, lamda = 0.5):
    input_x = input_image


    normal_x = normal_image


    B, C, W, H = input_x.size(0), input_x.size(1), input_x.size(2), input_x.size(3)


    mu = input_x.mean(dim=[2, 3], keepdim=True)
    var = input_x.var(dim=[2, 3], keepdim=True)
    sig = (var + 1e-6).sqrt()
    mu, sig = mu.detach(), sig.detach()
    x_normed = (input_x - mu) / sig

    mu2 = normal_x.mean(dim=[2, 3], keepdim=True)
    var2 = normal_x.var(dim=[2, 3], keepdim=True)
    sig2 = (var2 + 1e-6).sqrt()
    mu_mix = mu * lamda + mu2 * (1 - lamda)
    sig_mix = sig * lamda + sig2 * (1 - lamda)

    new_input_x = x_normed * sig_mix + mu_mix

    return new_input_x

def EFDM_test(input_image, normal_image, lamda = 0.5):

    lamda = 1-lamda
    input_x = input_image
    normal_x = normal_image
    B, C, W, H = input_x.size(0), input_x.size(1), input_x.size(2), input_x.size(3)
    input_x_view = input_x.view(B, C, -1)
    normal_x_view = normal_x.view(B, C, -1)

    value_input_x, index_input_x = torch.sort(input_x_view)
    value_normal_x, index_normal_x = torch.sort(normal_x_view)
    new_input_x = value_input_x + (value_normal_x - value_input_x) * lamda
    inverse_index = index_input_x.argsort(-1)
    new_input_x = new_input_x.gather(-1, inverse_index)
    new_input_x = new_input_x.view(B, C, W, H)
    # new_input_x = new_input_x.cpu().detach().numpy()
    # new_input_x = torch.from_numpy(new_input_x)

    return new_input_x

def sample_align(input, target, lamda = 0.5):

    lamda = 1-lamda
    B, C, W, H = input.size(0), input.size(1), input.size(2), input.size(3)
    B2 = target.shape[0]

    random_index = torch.randint(0, target.shape[0], (B, C, H, W))
    sampled_values = target[random_index, torch.arange(C)[None, :, None, None], torch.arange(H)[None, None, :, None], torch.arange(W)[None, None, None, :]]

    new_input_x = input + (sampled_values - input) * lamda
    # new_input_x = new_input_x.cpu().detach().numpy()
    # new_input_x = torch.from_numpy(new_input_x)

    return new_input_x

class Align_Test_Model(torch.nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        backbone = args.backbone
        if (backbone == "resnet152") or (backbone == 152):
            self.backbone = models.resnet152(pretrained=True)
        elif backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=True)
        elif backbone == "wide_resnet50_2":
            self.backbone = models.wide_resnet50_2(pretrained=True)
        self.backbone.fc = torch.nn.Identity()
        freeze_parameters(self.backbone, backbone, train_fc=False)
        if ("freeze_m" in args) and (args.freeze_m == 1):
            for k, v in self.backbone.named_parameters():
                if not ('layer4' in k):
                    v.requires_grad = False
    
    def forward(self, x):
        z1 = self.backbone(x)
        z_n = F.normalize(z1, dim=-1)
        return z_n
    
    def normal_feature_sample(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x1 = self.backbone.layer1(x)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)

        x = self.backbone.avgpool(x4)
        x = torch.flatten(x, 1)
        z1 = self.backbone.fc(x)

        # z1 = self.backbone(x)
        
        return x1, x2

    def inference(self, x, feature1_list, feature2_list):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x1 = self.backbone.layer1(x)
        
        if self.args.test_type == "sample_align":
            x1 = sample_align(x1, feature1_list)

        x2 = self.backbone.layer2(x1)
        if self.args.test_type == "sample_align":
            x2 = sample_align(x2, feature2_list)

        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)

        x = self.backbone.avgpool(x4)
        x = torch.flatten(x, 1)
        z1 = self.backbone.fc(x)

        # z1 = self.backbone(x)
        invariant_feature = F.normalize(z1, dim=-1)

        return invariant_feature

class Multi_Scale_Model(torch.nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        backbone = args.backbone
        if (backbone == "resnet152") or (backbone == 152):
            self.backbone = models.resnet152(pretrained=True)
        elif backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=True)
        elif backbone == "wide_resnet50_2":
            self.backbone = models.wide_resnet50_2(pretrained=True)
        self.backbone.fc = torch.nn.Identity()
        freeze_parameters(self.backbone, backbone, train_fc=False)
        if ("freeze_m" in args) and (args.freeze_m == 1):
            for k, v in self.backbone.named_parameters():
                if not ('layer4' in k):
                    v.requires_grad = False
        
        from utils_DGAD_method15 import _BN_layer, AttnBottleneck
        kwargs['width_per_group'] = 64 * 2
        self.specfic_conv = _BN_layer(args, "wide_resnet50_2", AttnBottleneck, [3, 4, 6, 3], True, True, **kwargs)
        self.args = args

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x1 = self.backbone.layer1(x)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)

        x = self.backbone.avgpool(x4)
        x = torch.flatten(x, 1)
        z1 = self.backbone.fc(x)

        # z1 = self.backbone(x)
        invariant_feature = F.normalize(z1, dim=-1)
        
        if self.args.conv_layer == 1:
            specific_feature = x1
        elif self.args.conv_layer == 2:
            specific_feature = x2
        elif self.args.conv_layer == 3:
            specific_feature = x3
        elif self.args.conv_layer == 4:
            specific_feature = x4

        specific_feature = self.specfic_conv(specific_feature)
        specific_feature = F.normalize(torch.flatten(specific_feature, 1), dim=-1)
        
        return specific_feature, invariant_feature

class Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        backbone = args.backbone
        if (backbone == "resnet152") or (backbone == 152):
            self.backbone = models.resnet152(pretrained=True)
        elif backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=True)
        elif backbone == "wide_resnet50_2":
            self.backbone = models.wide_resnet50_2(pretrained=True)
        self.backbone.fc = torch.nn.Identity()
        freeze_parameters(self.backbone, backbone, train_fc=False)
        if ("freeze_m" in args) and (args.freeze_m == 1):
            for k, v in self.backbone.named_parameters():
                if not ('layer4' in k):
                    v.requires_grad = False

    def forward(self, x):
        z1 = self.backbone(x)
        z_n = F.normalize(z1, dim=-1)
        return z_n

class MLP(nn.Module):
    def __init__(self, args, dims):
        super(MLP, self).__init__()
        self.args = args
        
        layers = [nn.Linear(dims[i - 1], dims[i], bias=False) for i in range(1, len(dims))]
        self.hidden = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.hidden:
            x = F.leaky_relu(layer(x))
        return x
    
class ScoreNet(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.score_net = MLP(args, [2048, 512, 64, 1])

    def forward(self, x):
        return self.score_net(x)


def freeze_parameters(model, backbone, train_fc=False):
    if not train_fc:
        for p in model.fc.parameters():
            p.requires_grad = False
    if backbone == 152:
        for p in model.conv1.parameters():
            p.requires_grad = False
        for p in model.bn1.parameters():
            p.requires_grad = False
        for p in model.layer1.parameters():
            p.requires_grad = False
        for p in model.layer2.parameters():
            p.requires_grad = False

def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)

def get_loaders(dataset, label_class, batch_size, backbone):
    if dataset == "cifar10":
        ds = torchvision.datasets.CIFAR10
        transform = transform_color if backbone == 152 else transform_resnet18
        coarse = {}
        trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
        testset = ds(root='data', train=False, download=True, transform=transform, **coarse)
        trainset_1 = ds(root='data', train=True, download=True, transform=Transform(), **coarse)
        idx = np.array(trainset.targets) == label_class
        testset.targets = [int(t != label_class) for t in testset.targets]
        trainset.data = trainset.data[idx]
        trainset.targets = [trainset.targets[i] for i, flag in enumerate(idx, 0) if flag]
        trainset_1.data = trainset_1.data[idx]
        trainset_1.targets = [trainset_1.targets[i] for i, flag in enumerate(idx, 0) if flag]
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2,
                                                   drop_last=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2,
                                                  drop_last=False)
        train_loader_1 = torch.utils.data.DataLoader(trainset_1, batch_size=batch_size, shuffle=True, num_workers=2,
                                                     drop_last=False)
        return train_loader, test_loader, train_loader_1
    else:
        print('Unsupported Dataset')
        exit()