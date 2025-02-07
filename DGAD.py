import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
import torch.optim as optim
import argparse
import net_work
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from datasets.PACS import PACS_Data
from datasets.MVTEC import MVTEC_Data
from datasets.MNIST import MNIST_Data

def contrastive_loss(out_1, out_2):
    out_1 = F.normalize(out_1, dim=-1)
    out_2 = F.normalize(out_2, dim=-1)
    bs = out_1.size(0)
    temp = 0.25
    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temp)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * bs, device=sim_matrix.device)).bool()
    # [2B, 2B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * bs, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temp)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss


def train_model(model, train_loader, val_loader, test_loader, device, args):
    model.eval()
    roc, prc, feature_space = get_score(model, device, train_loader, val_loader)
    print('Epoch: {}, AUROC is: {}, AUPRC is :{}'.format(0, roc, prc))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005)
    center = torch.FloatTensor(feature_space).mean(dim=0)
    if args.angular:
        center = F.normalize(center, dim=-1)
    center = center.to(device)
    
    val_max_metric = {"AUROC": -1,
                      "AUPRC": -1}
    val_results_loss = []
    val_AUROC_list = []
    val_AUPRC_list = []
    train_results_loss = []
    test_results_list = []

    for epoch in range(args.epochs):
        running_loss = run_epoch(model, train_loader, optimizer, center, device, args.angular)
        train_results_loss.append(running_loss)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
        roc, prc, _ = get_score(model, device, train_loader, val_loader)
        print('Epoch: {}, val_AUROC is: {}, val_AUPRC is :{}'.format(epoch + 1, roc, prc))
        if prc > val_max_metric["AUPRC"]:
            val_max_metric["AUPRC"] = prc
            val_max_metric["AUROC"] = roc
            val_max_metric["epoch"] = epoch
            torch.save(model.state_dict(), os.path.join(args.experiment_dir, filename + ".pt"))
        val_AUROC_list.append(roc)
        val_AUPRC_list.append(prc)
        test_metric = None
        if (args.test_epoch != 0) and ((epoch == 0) or (epoch % args.test_epoch == 0)):
            test_metric = test(model, train_loader, test_loader)
        
        test_results_list.append(test_metric)

    
    model.load_state_dict(torch.load(os.path.join(args.experiment_dir, filename + ".pt")))
    test_metric = test(model, train_loader, test_loader)
    val_max_metric["metric"] = test_metric
    print(f'results{args.results_save_path}/{filename}.npz')
    np.savez(f'results{args.results_save_path}/{filename}.npz',
             val_max_metric = np.array(val_max_metric),
             train_results_loss = np.array(train_results_loss),
            #  sub_train_results_loss = np.array(sub_train_results_loss),
             val_results_loss = np.array(val_results_loss),
             val_AUROC_list = np.array(val_AUROC_list),
             val_AUPRC_list = np.array(val_AUPRC_list),
             test_results_list = np.array(test_results_list),
             test_metric = np.array(test_metric),
             args = np.array(args.__dict__),)
    os.remove(os.path.join(args.experiment_dir, filename + ".pt"))

def test(model, train_loader, test_loader):
    train_feature_space = []
    with torch.no_grad():
        for (idx, imgs, augimg, label, _) in tqdm(train_loader, desc='Train set feature extracting'):
            imgs = imgs.cuda()
            features = model(imgs)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()

    test_metric = {}
    for key in test_loader:
        print(key)
        test_feature_space = []
        test_labels = []
        with torch.no_grad():
            for (idx, imgs, augimg, labels, _) in tqdm(test_loader[key], desc='Test set feature extracting'):
                imgs = imgs.cuda()
                features = model(imgs)
                test_feature_space.append(features)
                test_labels.append(labels)
            test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
            test_labels = torch.cat(test_labels, dim=0).cpu().numpy()

        distances = net_work.knn_score(train_feature_space, test_feature_space)

        roc = roc_auc_score(test_labels, distances)
        precision, recall, threshold = precision_recall_curve(test_labels, distances)
        prc = auc(recall, precision)

        test_metric[key] = {
            "test_loss_list": None,
            "AUROC": roc,
            "AUPRC": prc,
            "pred":distances,
            "target":test_labels
        }
    return test_metric

def run_epoch(model, train_loader, optimizer, center, device, is_angular):
    total_loss, total_num = 0.0, 0
    train_start = time.time()
    for (idx, img1, img2, label, _) in tqdm(train_loader, desc='Train...'):
        
        img1, img2 = img1.to(device), img2.to(device)

        optimizer.zero_grad()

        out_1 = model(img1)
        out_2 = model(img2)
        out_1 = out_1 - center
        out_2 = out_2 - center

        loss = contrastive_loss(out_1, out_2)

        if is_angular:
            loss += ((out_1 ** 2).sum(dim=1).mean() + (out_2 ** 2).sum(dim=1).mean())

        loss.backward()

        optimizer.step()

        total_num += img1.size(0)
        total_loss += loss.item() * img1.size(0)
    
    train_end = time.time()
    print("training_time", train_end - train_start)
    return total_loss / (total_num)


def get_score(model, device, train_loader, test_loader):
    train_feature_space = []
    with torch.no_grad():
        for (idx, imgs, augimg, label, _) in tqdm(train_loader, desc='Train set feature extracting'):
            imgs = imgs.to(device)
            features = model(imgs)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
    test_feature_space = []
    test_labels = []
    with torch.no_grad():
        for (idx, imgs, augimg, labels, _) in tqdm(test_loader, desc='Test set feature extracting'):
            imgs = imgs.to(device)
            features = model(imgs)
            test_feature_space.append(features)
            test_labels.append(labels)
        test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        test_labels = torch.cat(test_labels, dim=0).cpu().numpy()

    distances = net_work.knn_score(train_feature_space, test_feature_space)

    roc = roc_auc_score(test_labels, distances)
    precision, recall, threshold = precision_recall_curve(test_labels, distances)
    prc = auc(recall, precision)
    
    return roc, prc, train_feature_space

def build_dataloader(args, **kwargs):

    if args.dataset == "PACS":
        data = PACS_Data(args)
        
    if args.dataset == "MVTEC":
        data = MVTEC_Data(args)
    
    if args.dataset == "MNIST":
        data = MNIST_Data(args)
    
    train_set = data.train_data
    train_loader = DataLoader(train_set, batch_size=args.batch_size, **kwargs)
    val_data = data.val_data
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = {}
    for key in data.test_dict:
        test_loader[key] = DataLoader(data.test_dict[key], batch_size=args.batch_size, shuffle=False, **kwargs)
    
    unlabeled_data = data.unlabeled_data
    unlabeled_loader = DataLoader(unlabeled_data, batch_size=args.batch_size, **kwargs)
    
    return train_loader, val_loader, test_loader, unlabeled_loader

def main(args):
    print('Dataset: {}, Normal Label: {}, LR: {}'.format(args.dataset, args.label, args.lr))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = net_work.Model(args)
    model = model.to(device)

    # train_loader, test_loader, train_loader_1 = utils.get_loaders(dataset=args.dataset, label_class=args.label, batch_size=args.batch_size, backbone=args.backbone)
    kwargs = {'num_workers': args.workers}
    _, val_loader, test_loader, train_loader = build_dataloader(args, **kwargs)
    
    train_model(model, train_loader, val_loader, test_loader, device, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='MVTEC')
    parser.add_argument("--contamination_rate", type=float ,default=0)
    parser.add_argument("--checkitew", type=str, default="bottle")
    parser.add_argument("--normal_class", nargs="+", type=int, default=[0])
    parser.add_argument("--anomaly_class", nargs="+", type=int, default=[1,2,3,4,5,6])
    parser.add_argument('--epochs', default=2, type=int, metavar='epochs', help='number of epochs')
    parser.add_argument('--label', default=0, type=int, help='The normal class')
    parser.add_argument('--lr', type=float, default=1e-5, help='The initial learning rate.')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--backbone', default='wide_resnet50_2', type=str, help='ResNet 18/152')
    parser.add_argument('--angular', action='store_true', help='Train with angular center loss')
    parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
    parser.add_argument('--experiment_dir', type=str, default='/experiment', help="experiment dir root")
    parser.add_argument("--results_save_path", type=str, default="/DEBUG")
    parser.add_argument("--test_epoch", type=int, default=5)
    parser.add_argument("--domain_cnt", type=int, default=4)
    parser.add_argument("--in_domain_type", nargs="+", type=str, default=["MNIST", "MNIST_M", "SVHN"], choices=["MNIST", "MNIST_M", "SYN", "SVHN"])
    parser.add_argument("--label_discount", type=float, default=1.0)
    parser.add_argument("--cnt", type=int, default=0)
    
    args = parser.parse_args()
    # args = parser.parse_args(["--dataset", "MVTEC", "--domain_cnt", "4"])

    args.label_discount = int(8 * 27 / args.label_discount)
    args.experiment_dir = f"experiment{args.results_save_path}"

    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    if not os.path.exists(f"results{args.results_save_path}"):
        os.makedirs(f"results{args.results_save_path}")

    if args.dataset == "PACS":
        filename = f'dataset={args.dataset},normal_class={args.normal_class},anomaly_class={args.anomaly_class},epochs={args.epochs},lr={args.lr},batch_size={args.batch_size},backbone={args.backbone},contamination_rate={args.contamination_rate},cnt={args.cnt}'
    if args.dataset == "MVTEC":
        filename = f'dataset={args.dataset},checkitew={args.checkitew},epochs={args.epochs},lr={args.lr},batch_size={args.batch_size},backbone={args.backbone},domain_cnt={args.domain_cnt},cnt={args.cnt}'
    if args.dataset == "MNIST":
        filename = f'dataset={args.dataset},normal_class={args.normal_class},anomaly_class={args.anomaly_class},epochs={args.epochs},lr={args.lr},batch_size={args.batch_size},backbone={args.backbone},label_discount={args.label_discount},cnt={args.cnt}'
    main(args)