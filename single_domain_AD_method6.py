import os
from torch.utils.data import Sampler
from datasets.base_dataset import BaseADDataset
import math
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, average_precision_score
import torch.optim as optim
import argparse
import net_work2
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from datasets.PACS import PACS_Data
from datasets.MVTEC import MVTEC_Data
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
import torch.optim.lr_scheduler as lr_scheduler

def contrastive_loss(out_1, out_2):
    out_1 = F.normalize(out_1, dim=-1)
    out_2 = F.normalize(out_2, dim=-1)
    bs = out_1.size(0)
    temperature = args.temperature
    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * bs, device=sim_matrix.device)).bool()
    # [2B, 2B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * bs, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss

def aucPerformance(predict, labels, prt=True):
    roc_auc = roc_auc_score(labels, predict)
    ap = average_precision_score(labels, predict)
    if prt:
        print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap


def calc_score(score_net, cluster_centers, dataloader, domain_key):
    model.eval()
    score_net.eval()
    feature1_list = []
    feature2_list = []
    for sample in train_loader:
        # image, target = sample['image'], sample['label']
        idx, image, augimg, _, target, domain_label = sample

        image = image[torch.where(target == 0)[0]].cuda()
        with torch.no_grad():
            feature1, feature2 = model.normal_feature_sample(image)
            
        feature1_list.append(feature1)
        feature2_list.append(feature2)
    
    feature1_list = torch.concat(feature1_list)
    feature2_list = torch.concat(feature2_list)
    
    test_loss = 0.0
    total_pred = np.array([])
    total_target = np.array([])
    loss_list = []
    class_feature_list = []
    texture_feature_list = []
    target_list = []
    domain_label_list = []
    file_name_list = []
    cos_similarity_list = []
    for i, sample in enumerate(dataloader):
        idx, img1, augimg, _, target, domain_label = sample
        img1, target, domain_label = img1.cuda(), target.cuda(), domain_label.cuda()
        with torch.no_grad():
            invariant_feature = model.inference(img1, feature1_list, feature2_list)
            
            # cos_similarity = torch.mm(specific_feature, cluster_centers.T)
            output = score_net(invariant_feature)

            target_list.append(target.cpu().numpy())
            domain_label_list.append(domain_label.cpu().numpy())
            file_name_list.append(dataloader.dataset.image_paths[idx].reshape(-1))
            # cos_similarity_list.append(cos_similarity.cpu().numpy())

        total_pred = np.append(total_pred, output.cpu().numpy())
        total_target = np.append(total_target, target.cpu().numpy())
    file_name_list=np.concatenate(file_name_list)
    roc, pr = aucPerformance(total_pred, total_target)
    
    if args.save_embedding == 1:
        if not os.path.exists(f'./results/intermediate_results/{args.results_save_path}'):
            os.makedirs(f'./results/intermediate_results/{args.results_save_path}')

        np.savez(f"./results/intermediate_results/{args.results_save_path}/{filename},running_epoch={epoch},{domain_key}.npz",
                    # class_feature_list=np.concatenate(class_feature_list),
                    # texture_feature_list=np.concatenate(texture_feature_list),
                    cos_similarity_list=np.concatenate(cos_similarity_list),
                    target_list=np.concatenate(target_list),
                    domain_label_list=np.concatenate(domain_label_list),
                    total_pred=total_pred,
                    total_target=total_target,
                    AUROC=np.array(roc),
                    AUPRC=np.array(pr),
                    file_name_list=file_name_list
                    )

    return loss_list, roc, pr, total_pred, total_target, file_name_list

def test_after_fine_tune(score_net, cluster_centers, test_loader):
    train_feature_space = []
    
    test_metric = {}
    for key in test_loader:
        loss_list, roc, prc, total_pred, total_target, file_name_list = calc_score(score_net, cluster_centers, test_loader[key], key)

        test_metric[key] = {
            "test_loss_list": None,
            "AUROC": roc,
            "AUPRC": prc,
            "pred":total_pred,
            "target":total_target,
            "file_name_list":file_name_list
        }
    return test_metric


def train_model(score_net, device, args):
    
    val_max_metric = {"AUROC": -1,
                      "AUPRC": -1}
    val_results_loss = []
    val_AUROC_list = []
    val_AUPRC_list = []
    train_results_loss = []
    test_results_list = []
    
    warmup_epoch = math.ceil(args.ft_epochs * args.warmup)
    model_optimizer = optim.Adam(model.parameters(), lr = args.ft_lr, weight_decay=1e-5)
    score_optimizer = optim.Adam(score_net.parameters(), lr = args.score_lr, weight_decay=1e-5)
    if args.use_scheduler == 1:
        model_scheduler = lr_scheduler.CosineAnnealingLR(model_optimizer, T_max=args.ft_epochs, eta_min = args.ft_lr * 1e-6)
        score_scheduler = lr_scheduler.SequentialLR(score_optimizer, schedulers=[lr_scheduler.LinearLR(score_optimizer, start_factor=1, end_factor=1, total_iters=warmup_epoch),
                                                                                lr_scheduler.CosineAnnealingLR(score_optimizer, T_max=args.ft_epochs, eta_min = args.score_lr * 1e-6)],
                                                                                milestones=[warmup_epoch])

    sub_train_results_loss = []
    cluster_centers = None
    global epoch
    for epoch in range(args.ft_epochs):
        # torch.cuda.empty_cache()
        normal_score_list = []
        feature1_list = []
        feature2_list = []
        with torch.no_grad():
            score_net.eval()
            model.eval()
            for (_, img1, _, _, label, _) in tqdm(no_drop_loader, desc='init...'):
                img1, label = img1.to(device), label.to(device)

                img1 = img1[torch.where(label == 0)[0]]
                
                invariant_feature = model(img1)
                feature1, feature2 = model.normal_feature_sample(img1)
                
                feature1_list.append(feature1)
                feature2_list.append(feature2)

                scores = score_net(invariant_feature)
                normal_score_list.append(scores)
            
            feature1_list = torch.concat(feature1_list).detach()
            feature2_list = torch.concat(feature2_list).detach()
            normal_score_list = torch.concat(normal_score_list)
            border = torch.quantile(normal_score_list, args.quantile)

        model.train()
        score_net.train()
        total_loss, total_num = 0.0, 0
        train_loss_list = []
        sub_train_loss_list = []
        training_data_loader = balance_loader if args.BalancedBatchSampler == 1 else train_loader
        for (idx, img1, augimg, gray_img, label, _) in tqdm(training_data_loader, desc='Train...'):
            
            img1, augimg, gray_img, label = img1.to(device), augimg.to(device), gray_img.to(device), label.to(device)

            model_optimizer.zero_grad()
            score_optimizer.zero_grad()

            normal_idx = torch.where(label == 0)[0]
            anomaly_idx = torch.where(label == 1)[0]

            invariant_feature = model(img1)
            aug_invariant_feature = model(augimg)
            # gray_feature = model(gray_img)
            align_feature = model.inference(img1, feature1_list, feature2_list)

            L_CL1 = contrastive_loss(invariant_feature, aug_invariant_feature)
            # L_CL2 = contrastive_loss(invariant_feature, gray_feature)
            L_CL2 = contrastive_loss(invariant_feature, align_feature)
            
            scores = score_net(invariant_feature)
            L_normal_score = (scores[normal_idx] - border).clamp_(min=0.).mean()

            L_anomaly_score = (border + args.confidence_margin - scores[anomaly_idx]).clamp_(min=0.).mean()

            if args.lambda1 != 0:
                loss = L_CL1 + args.lambda0 * L_CL2 + min(epoch / warmup_epoch, 1) * (L_normal_score + args.lambda1 * L_anomaly_score)
            else:
                L_classfier = torch.nn.BCELoss()(torch.sigmoid(scores.reshape(-1)), label.to(torch.float32))
                loss = L_CL1 + args.lambda0 * L_CL2 + L_classfier

            loss.backward()

            model_optimizer.step()
            score_optimizer.step()

            total_num += img1.size(0)
            total_loss += loss.item()
            train_loss_list.append(loss.item())
            if args.lambda1 != 0:
                sub_train_loss_list.append([L_CL1.item(), L_CL2.item(), L_normal_score.item(), L_anomaly_score.item()])
            else:
                sub_train_loss_list.append([L_CL1.item(), L_CL2.item(), L_classfier.item()])

        if args.use_scheduler == 1:
            model_scheduler.step()
            print("model_optimizer_lr", model_optimizer.state_dict()['param_groups'][0]['lr'])
            score_scheduler.step()
            print("score_optimizer_lr", score_optimizer.state_dict()['param_groups'][0]['lr'])

        running_loss = total_loss / (total_num)
        train_results_loss.append(running_loss)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
        sub_train_results_loss.append(sub_train_loss_list)


        _, roc, prc, _, _, _ = calc_score(score_net, cluster_centers, val_loader, "val")
        print('Epoch: {}, val_AUROC is: {}, val_AUPRC is :{}'.format(epoch + 1, roc, prc))
        if prc > val_max_metric["AUPRC"]:
            val_max_metric["AUPRC"] = prc
            val_max_metric["AUROC"] = roc
            val_max_metric["epoch"] = epoch
            save_model = {
                'model': model.state_dict(),
                'score_net': score_net.state_dict(),
            }
            torch.save(save_model, os.path.join(args.experiment_dir, filename + ".pt"))
        val_AUROC_list.append(roc)
        val_AUPRC_list.append(prc)
        test_metric = None
        if (epoch == 0) or ((epoch + 1) % args.test_epoch == 0):
            test_metric = test_after_fine_tune(score_net, cluster_centers, test_loader)
        
        test_results_list.append(test_metric)

    load_models = torch.load(os.path.join(args.experiment_dir, filename + ".pt"))
    model.load_state_dict(load_models["model"])
    score_net.load_state_dict(load_models["score_net"])
    test_metric = test_after_fine_tune(score_net, cluster_centers, test_loader)
    val_max_metric["metric"] = test_metric
    print(f'results{args.results_save_path}/{filename}.npz')
    np.savez(f'results{args.results_save_path}/{filename}.npz',
             val_max_metric = np.array(val_max_metric),
             train_results_loss = np.array(train_results_loss),
             sub_train_results_loss = np.array(sub_train_results_loss),
             val_results_loss = np.array(val_results_loss),
             val_AUROC_list = np.array(val_AUROC_list),
             val_AUPRC_list = np.array(val_AUPRC_list),
             test_results_list = np.array(test_results_list),
             test_metric = np.array(test_metric),
             args = np.array(args.__dict__),)
    
    os.remove(os.path.join(args.experiment_dir, filename + ".pt"))

def worker_init_fn_seed(worker_id):
    seed = 10
    seed += worker_id
    np.random.seed(seed)


class BalancedBatchSampler(Sampler):
    def __init__(self,
                 cfg,
                 dataset: BaseADDataset):
        super(BalancedBatchSampler, self).__init__(dataset)
        self.cfg = cfg
        self.dataset = dataset

        self.normal_generator = self.random_generator(self.dataset.normal_idx)
        self.outlier_generator = self.random_generator(self.dataset.outlier_idx)
        if self.cfg.n_anomaly != 0:
            self.n_normal = self.cfg.batch_size // 2
            self.n_outlier = self.cfg.batch_size - self.n_normal
        else:
            self.n_normal = self.cfg.batch_size
            self.n_outlier = 0

    @staticmethod
    def random_generator(idx_list):
        while True:
            random_list = np.random.permutation(idx_list)
            for i in random_list:
                yield i

    def __len__(self):
        return self.cfg.steps_per_epoch
    
    def __iter__(self):
        for _ in range(self.cfg.steps_per_epoch):
            batch = []

            for _ in range(self.n_normal):
                batch.append(next(self.normal_generator))

            for _ in range(self.n_outlier):
                batch.append(next(self.outlier_generator))
            yield batch


def build_dataloader(args, **kwargs):

    if args.dataset == "PACS":
        data = PACS_Data(args)
        
    if args.dataset == "MVTEC":
        data = MVTEC_Data(args)

    train_set = data.train_data
    no_drop_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers = args.workers, drop_last=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers = args.workers, drop_last=True)
    val_data = data.val_data
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = {}
    for key in data.test_dict:
        test_loader[key] = DataLoader(data.test_dict[key], batch_size=args.batch_size, shuffle=False, **kwargs)
    
    unlabeled_data = data.unlabeled_data
    unlabeled_loader = DataLoader(unlabeled_data, batch_size=args.batch_size, num_workers = args.workers, drop_last=True)

    balance_loader = DataLoader(train_set, worker_init_fn=worker_init_fn_seed, batch_sampler=BalancedBatchSampler(args, train_set), **kwargs)
    
    return train_loader, val_loader, test_loader, unlabeled_loader, balance_loader, no_drop_loader

def main(args):
    print('Dataset: {}, Normal Label: {}, LR: {}'.format(args.dataset, args.label, args.lr))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    global model
    model = net_work2.Align_Test_Model(args)
    model = model.to(device)

    # train_loader, test_loader, train_loader_1 = utils.get_loaders(dataset=args.dataset, label_class=args.label, batch_size=args.batch_size, backbone=args.backbone)
    kwargs = {'num_workers': args.workers}
    # _, val_loader, test_loader, train_loader = build_dataloader(args, **kwargs)
    global train_loader, val_loader, test_loader, unlabeled_loader, balance_loader, no_drop_loader
    train_loader, val_loader, test_loader, unlabeled_loader, balance_loader, no_drop_loader = build_dataloader(args, **kwargs)
    
    print("\n===================\ntrain_model\n===================\n")
    score_net = net_work2.ScoreNet(args)
    score_net = score_net.to(device)
    train_model(score_net, device, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='PACS')
    parser.add_argument("--contamination_rate", type=float ,default=0.04)
    parser.add_argument("--checkitew", type=str, default="bottle")
    parser.add_argument("--normal_class", nargs="+", type=int, default=[6])
    parser.add_argument("--anomaly_class", nargs="+", type=int, default=[0,1,2,3,4,5])
    parser.add_argument('--epochs', default=2, type=int, metavar='epochs', help='number of epochs')
    parser.add_argument('--ft_epochs', default=2, type=int, help='number of fine tune epochs')
    parser.add_argument('--label', default=0, type=int, help='The normal class')
    parser.add_argument('--lr', type=float, default=1e-5, help='The initial learning rate.')
    parser.add_argument('--ft_lr', type=float, default=1e-5, help='The fine tune learning rate.')
    parser.add_argument('--score_lr', type=float, default=1e-3, help='The fine tune learning rate.')
    parser.add_argument('--confidence_margin', type=float, default=3, help='confidence_margin.')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--backbone', default='wide_resnet50_2', type=str, help='ResNet 18/152')
    parser.add_argument('--angular', action='store_true', help='Train with angular center loss')
    parser.add_argument('--workers', type=int, default=8, metavar='N', help='dataloader threads')
    parser.add_argument('--experiment_dir', type=str, default='/experiment', help="experiment dir root")
    parser.add_argument("--results_save_path", type=str, default="/DEBUG")
    parser.add_argument("--test_epoch", type=int, default=5)
    parser.add_argument("--n_anomaly", type=int, default=10)
    parser.add_argument("--steps_per_epoch", type=int, default=5, help="the number of batches per epoch")
    parser.add_argument("--k_cluster", type=int, default=3)
    parser.add_argument("--domain_cnt", type=int, default=1)
    parser.add_argument("--cnt", type=int, default=0)
    parser.add_argument("--lambda0", type=int, default=1)
    parser.add_argument("--lambda1", type=int, default=1)
    parser.add_argument("--no_center", type=int, default=1)
    parser.add_argument("--freeze_m", type=int, default=1)
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--quantile", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--supervised", type=str, default="semi-")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default="3")
    parser.add_argument("--save_embedding", type=int, default=0)
    parser.add_argument("--warmup", type=float, default=0.25)
    parser.add_argument("--use_scheduler", type=int, default=0)
    parser.add_argument("--conv_layer", type=int, default=4)
    parser.add_argument("--test_type", type=str, default="sample_align")
    parser.add_argument("--gray", type=int, default=1)
    parser.add_argument("--BalancedBatchSampler", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    
    # args = parser.parse_args(["--ft_epochs", "20" , "--ft_lr", "0.0005", "--score_lr", "0.0005", "--batch_size", "64", "--epochs", "5", "--lr", "0.0001"])
    args = parser.parse_args()

    torch.manual_seed(args.random_seed)
    args.experiment_dir = f"experiment{args.results_save_path}"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    if not os.path.exists(f"results{args.results_save_path}"):
        os.makedirs(f"results{args.results_save_path}")

    if args.dataset == "PACS":
        filename = f'dataset={args.dataset},normal_class={args.normal_class},batch_size={args.batch_size},ft_lr={args.ft_lr},ft_epochs={args.ft_epochs},score_lr={args.score_lr},contamination_rate={args.contamination_rate},lambda0={args.lambda0},lambda1={args.lambda1},freeze_m={args.freeze_m},warmup={args.warmup},alpha={args.alpha},use_scheduler={args.use_scheduler},BalancedBatchSampler={args.BalancedBatchSampler},cnt={args.cnt}'
    if args.dataset == "MVTEC":
        filename = f'dataset={args.dataset},checkitew={args.checkitew},batch_size={args.batch_size},ft_lr={args.ft_lr},ft_epochs={args.ft_epochs},score_lr={args.score_lr},lambda0={args.lambda0},lambda1={args.lambda1},freeze_m={args.freeze_m},warmup={args.warmup},alpha={args.alpha},use_scheduler={args.use_scheduler},BalancedBatchSampler={args.BalancedBatchSampler},cnt={args.cnt}'
    main(args)