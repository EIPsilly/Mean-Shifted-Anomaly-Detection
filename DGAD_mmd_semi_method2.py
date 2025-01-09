import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, average_precision_score
import torch.optim as optim
import argparse
import net_work
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from datasets.PACS import PACS_Data
from datasets.MVTEC import MVTEC_Data
from mmd import MMDLoss

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

def aucPerformance(predict, labels, prt=True):
    roc_auc = roc_auc_score(labels, predict)
    ap = average_precision_score(labels, predict)
    if prt:
        print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap


def calc_score(model, score_net, center, dataloader):
    model.eval()
    score_net.eval()
    test_loss = 0.0
    total_pred = np.array([])
    total_target = np.array([])
    loss_list = []
    class_feature_list = []
    texture_feature_list = []
    target_list = []
    domain_label_list = []
    file_name_list = []
    for i, sample in enumerate(dataloader):
        idx, img1, augimg, target, domain_label = sample
        img1, target, domain_label = img1.cuda(), target.cuda(), domain_label.cuda()
        with torch.no_grad():
            out_1 = model(img1)
            if args.no_center == 0:
                out_1 = out_1 - center
            output = score_net(out_1)

            target_list.append(target.cpu().numpy())
            domain_label_list.append(domain_label.cpu().numpy())
            file_name_list.append(dataloader.dataset.image_paths[idx])

        total_pred = np.append(total_pred, output.cpu().numpy())
        total_target = np.append(total_target, target.cpu().numpy())
    file_name_list=np.concatenate(file_name_list)
    roc, pr = aucPerformance(total_pred, total_target)

    return loss_list, roc, pr, total_pred, total_target, file_name_list

def test_after_fine_tune(model, score_net, center, test_loader):
    train_feature_space = []
    
    test_metric = {}
    for key in test_loader:
        loss_list, roc, prc, total_pred, total_target, file_name_list = calc_score(model, score_net, center, test_loader[key])

        test_metric[key] = {
            "test_loss_list": None,
            "AUROC": roc,
            "AUPRC": prc,
            "pred":total_pred,
            "target":total_target,
            "file_name_list":file_name_list
        }
    return test_metric


def fine_tune_model(center, model, score_net, train_loader, unlabeled_loader, val_loader, test_loader, device, args):
    optimizer = optim.Adam([
        {'params': model.parameters(), "lr": args.ft_lr}, 
        {'params': score_net.parameters(), "lr": args.score_lr},
	], weight_decay=1e-5)
    
    val_max_metric = {"AUROC": -1,
                      "AUPRC": -1}
    val_results_loss = []
    val_AUROC_list = []
    val_AUPRC_list = []
    train_results_loss = []
    test_results_list = []
    
    normal_score_list = []
    with torch.no_grad():
        score_net.eval()
        model.eval()
        for (idx, img1, augimg, label, domain_labels) in tqdm(unlabeled_loader, desc='Get Border...'):
            img1, augimg = img1.to(device), augimg.to(device)
            out_1 = model(img1)
            if args.no_center == 0:
                out_1 = out_1 - center

            scores = score_net(out_1)
            normal_score_list.append(scores[torch.where(label == 0)[0]])
        
        border = torch.quantile(torch.concat(normal_score_list), args.quantile)
    
    sub_train_results_loss = []
    for epoch in range(args.ft_epochs):
        
        model.train()
        score_net.train()
        total_loss, total_num = 0.0, 0
        train_loss_list = []
        sub_train_loss_list = []
        for (idx, img1, augimg, label, domain_labels) in tqdm(train_loader, desc='Train...'):
            
            img1, augimg, label = img1.to(device), augimg.to(device), label.to(device)

            optimizer.zero_grad()

            normal_idx = torch.where(label == 0)[0]
            out_1 = model(img1)
            out_2 = model(augimg)
            if args.no_center == 0:
                out_1 = out_1 - center
                out_2 = out_2 - center

            L_CL = contrastive_loss(out_1[normal_idx], out_2[normal_idx])
            L_mmd = args.lambda0 * mmd(domain_labels[normal_idx], out_1[normal_idx])

            scores = score_net(out_1)
            L_normal_score = (scores[normal_idx] - border).clamp_(min=0.).sum()
            L_anomaly_score = (border + args.confidence_margin - scores[torch.where(label == 1)[0]]).clamp_(min=0.).sum()

            if args.lambda1 == 1:
                loss = L_CL + L_mmd + args.lambda1 * (L_normal_score + L_anomaly_score)
            else:
                L_classfier = torch.nn.BCELoss()(torch.sigmoid(scores.reshape(-1)), label.to(torch.float32))
                loss = L_CL + L_mmd + L_classfier
            # CrossEntropyLoss
            
            if args.angular:
                loss += ((out_1 ** 2).sum(dim=1).mean() + (out_2 ** 2).sum(dim=1).mean())

            loss.backward()

            optimizer.step()

            total_num += img1.size(0)
            total_loss += loss.item()
            train_loss_list.append(loss.item())
            if args.lambda1 == 1:
                sub_train_loss_list.append([L_CL.item(), L_mmd.item(), L_normal_score.item(), L_anomaly_score.item()])
            else:
                sub_train_loss_list.append([L_CL.item(), L_mmd.item(), L_classfier.item()])
            
        running_loss = total_loss / (total_num)
        train_results_loss.append(running_loss)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))
        sub_train_results_loss.append(sub_train_loss_list)


        _, roc, prc, _, _, _ = calc_score(model, score_net, center, val_loader)
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
            test_metric = test_after_fine_tune(model, score_net, center, test_loader)
        
        test_results_list.append(test_metric)

    load_models = torch.load(os.path.join(args.experiment_dir, filename + ".pt"))
    model.load_state_dict(load_models["model"])
    score_net.load_state_dict(load_models["score_net"])
    test_metric = test_after_fine_tune(model, score_net, center, test_loader)
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

def train_model(model, train_loader, val_loader, test_loader, device, args):
    model.eval()
    train_feature_space = []
    with torch.no_grad():
        for (idx, imgs, augimg, label, domain_label) in tqdm(train_loader, desc='Train set feature extracting'):
            imgs = imgs.to(device)
            features = model(imgs)
            train_feature_space.append(features)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005)
    center = torch.FloatTensor(train_feature_space).mean(dim=0)
    if args.angular:
        center = F.normalize(center, dim=-1)
    center = center.to(device)
    train_results_loss = []

    for epoch in range(args.epochs):
        running_loss = run_epoch(model, train_loader, optimizer, center, device, args.angular)
        train_results_loss.append(running_loss)
        print('Epoch: {}, Loss: {}'.format(epoch + 1, running_loss))

    return center

def run_epoch(model, train_loader, optimizer, center, device, is_angular):
    model.train()
    total_loss, total_num = 0.0, 0
    for (idx, img1, augimg, label, domain_labels) in tqdm(train_loader, desc='Train...'):
        
        img1, augimg = img1.to(device), augimg.to(device)

        optimizer.zero_grad()

        out_1 = model(img1)
        out_2 = model(augimg)
        if args.no_center == 0:
            out_1 = out_1 - center
            out_2 = out_2 - center

        L_CL = contrastive_loss(out_1, out_2)
        L_mmd = args.lambda0 * mmd(domain_labels, out_1)

        loss = L_CL + L_mmd
        
        if is_angular:
            loss += ((out_1 ** 2).sum(dim=1).mean() + (out_2 ** 2).sum(dim=1).mean())

        loss.backward()

        optimizer.step()

        total_num += img1.size(0)
        total_loss += loss.item() * img1.size(0)

    return total_loss / (total_num)

def build_dataloader(args, **kwargs):

    if args.dataset == "PACS":
        data = PACS_Data(args)
        
    if args.dataset == "MVTEC":
        data = MVTEC_Data(args)

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = net_work.Model(args)
    global mmd
    mmd = MMDLoss(pooled=True)
    mmd = mmd.to(device)
    model = model.to(device)

    # train_loader, test_loader, train_loader_1 = utils.get_loaders(dataset=args.dataset, label_class=args.label, batch_size=args.batch_size, backbone=args.backbone)
    kwargs = {'num_workers': args.workers}
    # _, val_loader, test_loader, train_loader = build_dataloader(args, **kwargs)
    train_loader, val_loader, test_loader, unlabeled_loader = build_dataloader(args, **kwargs)
    
    center = train_model(model, unlabeled_loader, val_loader, test_loader, device, args)

    print("\n===================\nfine_tune_model\n===================\n")
    score_net = net_work.ScoreNet(args)
    score_net = score_net.to(device)
    fine_tune_model(center, model, score_net, train_loader, unlabeled_loader, val_loader, test_loader, device, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='PACS')
    parser.add_argument("--contamination_rate", type=float ,default=0)
    parser.add_argument("--checkitew", type=str, default="bottle")
    parser.add_argument("--normal_class", nargs="+", type=int, default=[0])
    parser.add_argument("--anomaly_class", nargs="+", type=int, default=[1,2,3,4,5,6])
    parser.add_argument('--epochs', default=1, type=int, metavar='epochs', help='number of epochs')
    parser.add_argument('--ft_epochs', default=1, type=int, help='number of fine tune epochs')
    parser.add_argument('--label', default=0, type=int, help='The normal class')
    parser.add_argument('--lr', type=float, default=1e-5, help='The initial learning rate.')
    parser.add_argument('--ft_lr', type=float, default=1e-5, help='The fine tune learning rate.')
    parser.add_argument('--score_lr', type=float, default=1e-3, help='The fine tune learning rate.')
    parser.add_argument('--confidence_margin', type=float, default=3, help='confidence_margin.')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--backbone', default='wide_resnet50_2', type=str, help='ResNet 18/152')
    parser.add_argument('--angular', action='store_true', help='Train with angular center loss')
    parser.add_argument('--workers', type=int, default=32, metavar='N', help='dataloader threads')
    parser.add_argument('--experiment_dir', type=str, default='/experiment', help="experiment dir root")
    parser.add_argument("--results_save_path", type=str, default="/DEBUG")
    parser.add_argument("--test_epoch", type=int, default=5)
    parser.add_argument("--domain_cnt", type=int, default=3)
    parser.add_argument("--cnt", type=int, default=0)
    parser.add_argument("--lambda0", type=int, default=1)
    parser.add_argument("--lambda1", type=int, default=1)
    parser.add_argument("--no_center", type=int, default=0)
    parser.add_argument("--quantile", type=float, default=1.0)
    parser.add_argument("--supervised", type=str, default="semi-")
    parser.add_argument("--random_seed", type=int, default=42)
    
    args = parser.parse_args()
    torch.manual_seed(args.random_seed)
    args.experiment_dir = f"experiment{args.results_save_path}"

    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)

    if not os.path.exists(f"results{args.results_save_path}"):
        os.makedirs(f"results{args.results_save_path}")

    if args.dataset == "PACS":
        filename = f'dataset={args.dataset},normal_class={args.normal_class},anomaly_class={args.anomaly_class},epochs={args.epochs},lr={args.lr},batch_size={args.batch_size},ft_lr={args.ft_lr},ft_epochs={args.ft_epochs},score_lr={args.score_lr},backbone={args.backbone},contamination_rate={args.contamination_rate},lambda0={args.lambda0},lambda1={args.lambda1},no_center={args.no_center},cnt={args.cnt}'
    if args.dataset == "MVTEC":
        filename = f'dataset={args.dataset},checkitew={args.checkitew},epochs={args.epochs},lr={args.lr},batch_size={args.batch_size},backbone={args.backbone},cnt={args.cnt}'
    main(args)