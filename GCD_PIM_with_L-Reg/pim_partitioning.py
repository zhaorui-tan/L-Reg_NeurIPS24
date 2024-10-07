import os
import yaml
import argparse
from tqdm import tqdm

import numpy as np
# import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from feature_maps_manip import feature_maps_manip as fm
from metrics import partitioning_metrics as p_metrics
from sklearn.cluster import KMeans
from models.clustering_methods import torch_semi_supervised_kmeans as ssKM
from configs import config_K
from params_estim.ssKM_protos_initialization import ssKM_protos_init
from params_estim.automatic_lambda_search import lambda_search
from models.PIM import PIM_partitioner, Affine

class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


def sample_covariance(a, b, invert=False):
    '''
    Sample covariance estimating
    a = [N,m]
    b = [N,m]
    '''
    # assert (a.shape[0] == b.shape[0])
    # assert (a.shape[1] == b.shape[1])
    # m = a.shape[1]
    N = a.shape[0]
    C = torch.matmul(a.T, b) / N
    if invert:
        return torch.linalg.pinv(C)
    else:
        return C




def get_cond_shift(X1, Y1, estimator=sample_covariance):
    # print(matrix1.shape, matrix2.shape)
    m1 = torch.mean(X1, dim=0)
    my1 = torch.mean(Y1, dim=0)
    x1 = X1 - m1
    y1 = Y1 - my1

    c_x1_y = estimator(x1, y1)
    c_y_x1 = estimator(y1, x1)
    # c_x1_x1 = estimator(x1, x1)

    inv_c_y_y = estimator(y1, y1, invert=True)
    # c_x1_given_y1 = c_x1_x1 - torch.matmul(c_x1_y, torch.matmul(inv_c_y_y, c_y_x1))
    # return nn.MSELoss()((c_x1_given_y1) , (c_x2_given_y2))
    shift = torch.matmul(c_x1_y, torch.matmul(inv_c_y_y, c_y_x1))
    # return torch.trace(aa) + torch.trace(bb)
    return nn.MSELoss()(shift, torch.zeros_like(shift))



def from_numpy_to_torch(np_array, torch_device):
    return torch.from_numpy(np_array).to(torch_device)


def from_torch_to_numpy(torch_tensor):
    return torch_tensor.cpu().numpy()


def partitioning_eval(unlab_gt_labs, unlab_preds, seen_mask, dset_name, path_k_strat):
    # GCD metrics (used in GCD arxiv v1)
    accs_v1 = p_metrics.cluster_acc_v1(unlab_gt_labs, unlab_preds, seen_mask)

    # GCD metrics (used in GCD CVPR-22 and GCD arxiv v2)
    accs_v2 = p_metrics.cluster_acc_v2(unlab_gt_labs, unlab_preds, seen_mask)

    # ORCA metrics (used in ORCA ICLR-22)
    orca_accs = p_metrics.orca_all_old_new_ACCs(unlab_preds, unlab_gt_labs, seen_mask)

    print("Classes:                      (All) & (Old) & (New)")

    print("PIM ACC (v1):                 ", np.round(100. * accs_v1[0], 1),
          "\t", np.round(100. * accs_v1[1], 1),
          "\t", np.round(100. * accs_v1[2], 1))

    print("PIM ACC (ORCA metric):        ", np.round(100. * orca_accs[0], 1),
          "\t", np.round(100. * orca_accs[1], 1),
          "\t", np.round(100. * orca_accs[2], 1))

    print("PIM ACC (Official GCD metric):", np.round(100. * accs_v2[0], 1),
          "\t", np.round(100. * accs_v2[1], 1),
          "\t", np.round(100. * accs_v2[2], 1))

    path_accs_v2 = 'params_estim/' + path_k_strat + '/' + dset_name + '/scores'
    if not os.path.exists(path_accs_v2):
        os.makedirs(path_accs_v2)

    accs_v2_file_name = path_accs_v2 + '/ACCs_v2.npy'
    with open(accs_v2_file_name, 'wb') as f:
        np.save(f, accs_v2)
    return orca_accs


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Arguments for GCD challenge")

    parser.add_argument('--dataset', type=str, default='cub',
                        choices=['cifar10', 'cifar100', 'imagenet_100',
                                 'cub', 'scars', 'herbarium'])

    parser.add_argument('--device_name', type=str, default='cuda:0',
                        choices=['cpu', 'cuda:0', 'cuda:1'])

    parser.add_argument('--centroids_init_nbr', type=int, default=100,
                        help="Number of separate centroids initialization.")

    parser.add_argument('--cfg', type=str, default="./configs/config_fm_paths.yml",
                        help="feature maps roots config file")

    parser.add_argument('--epochs', type=int, default=1000,
                        help="Total number of training epoch iterations for PIM partitioner.")

    parser.add_argument('--perform_init_protos', type=bool, default=False,
                        help="Set True if you want to (re)perform prototypes initiliaztion using ssKM.")

    parser.add_argument('--perform_lambda_search', type=bool, default=False,
                        help="Set True if you want to (re)perform automatic lambda search.")

    parser.add_argument('--seed', type=int, default=2024,
                        help="Seed for RandomState.")

    parser.add_argument('--k_strat', type=str, default='ground_truth_K',
                        choices=['ground_truth_K',
                                 'Max_ACC_PIM_Brent'],
                        help="Select a strategy to set the assumed number of classes " +
                             "(ground truth or precomputed using Max-ACC).")


    parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")
    parser.add_argument("--epsilon_sk", default=0.05, type=float, help="epsilon for the Sinkhorn")
    parser.add_argument("--imb-factor", default=1, type=float, help="imbalance factor of the data, default 1")
    parser.add_argument("--threshold", default=0.5, type=float, help="threshold for hard pseudo-labeling")
    parser.add_argument('--alpha', default=0.75, type=float)
    return parser.parse_args()


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    print('nu', nu)
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    print(xy)
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    for v in xy:
        print(len(v))
    return [torch.cat(v, dim=0) for v in xy]


def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor

class SinkhornKnopp(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_iters = args.num_iters_sk
        self.epsilon = args.epsilon_sk
        self.imb_factor = args.imb_factor

    @torch.no_grad()
    def iterate(self, Q):
        Q = shoot_infs(Q)
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / Q.shape[1]  # Samples

        if self.imb_factor > 1:
            # obtain permutation/order from the marginals
            marginals_argsort = torch.argsort(Q.sum(1))
            marginals_argsort = marginals_argsort.detach()
            r = []
            for i in range(Q.shape[0]):  # Classes
                r.append((1 / self.imb_factor) ** (i / (Q.shape[0] - 1.0)))
            r = np.array(r)
            r = r * (Q.shape[1] / Q.shape[0])  # Per-class distribution in the mini-batch
            r = torch.from_numpy(r).cuda(non_blocking=True)
            r[marginals_argsort] = torch.sort(r)[0]  # Sort/permute based on the data order
            r = torch.clamp(r, min=1)  # Clamp the min to have a balance distribution for the tail classes
            r /= r.sum()  # Scaling to make it prob
        else:
            r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]

        for it in range(self.num_iters):
            u = torch.sum(Q, dim=1)
            u = r / u
            u = shoot_infs(u)
            Q *= u.unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    @torch.no_grad()
    def forward(self, logits):
        # get assignments
        q = logits / self.epsilon
        M = torch.max(q)
        q -= M
        q = torch.exp(q).t()
        return self.iterate(q)


def main():
    args = get_arguments()
    with open(args.cfg, "r") as stream:
        try:
            cfg_paths = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    torch_device = torch.device(args.device_name)
    np.random.seed(args.seed)

    ### Get feature map set, ground truth labels, labeled set mask, old-new mask
    print("Dataset:", args.dataset)
    gcd_feature_map_datasets_path = cfg_paths["gcd_feature_map_datasets_path"]
    feature_maps_folder_name = cfg_paths[args.dataset]
    set_path = os.path.join(gcd_feature_map_datasets_path, feature_maps_folder_name, 'train')
    feature_map_preds, gt_labels, nbr_of_classes, mask_old_new, mask_lab = fm.get_fm_preds_and_gt_labels(set_path)
    ###

    ### Get assumed K
    assumed_k, path_k_strat = config_K.get_assumed_nbr_of_classes(args.k_strat, args.dataset)
    print('K strategy:', args.k_strat, '\nAssumed K:', assumed_k, '\n')
    path_params = '/path/for/params' + path_k_strat + '/' + args.dataset
    if not os.path.exists(path_params):
        os.makedirs(path_params)
    ###

    ### Get ssKM centroids to initialize PIM prototypes
    protos_file_name = path_params + '/' + 'protos.npy'
    if ((not os.path.exists(protos_file_name)) or (args.perform_init_protos == True)) and (
            args.k_strat != 'ground_truth_K'):
        # Estimate ssKM centroids
        print("PIM init prototypes estimation using ssKM...")
        prototypes = ssKM_protos_init(feature_map_preds,
                                      assumed_k,
                                      gt_labels[mask_lab],
                                      mask_lab,
                                      mask_old_new,
                                      args.centroids_init_nbr,
                                      args.device_name,
                                      protos_file_name)
    else:
        print("PIM init prototypes already generated \n")
        with open(protos_file_name, 'rb') as f:
            prototypes = np.load(f)
    prototypes = from_numpy_to_torch(np.asarray(prototypes), torch_device)
    ###

    ### Get lambda (automatically estimated)
    path_auto_lambda = path_params + '/auto_lambda_search'
    lambda_search_lab_acc_file_name = path_auto_lambda + '/' + "all_lab_Accs_" + str(assumed_k) + ".npy"
    if (not os.path.exists(lambda_search_lab_acc_file_name)) or (args.perform_lambda_search == True):
        # Auto lambda search
        print("Start automatic lambda search...")
        auto_lambda_val = lambda_search(path_auto_lambda,
                                        feature_map_preds,
                                        args.epochs,
                                        assumed_k,
                                        mask_lab,
                                        args.device_name,
                                        args.dataset,
                                        torch_device,
                                        gt_labels[mask_lab])
    else:
        print("lambda already estimated")
        with open(path_auto_lambda + '/' + 'lambda_vals_list_' + str(assumed_k) + '.npy', 'rb') as f:
            lambda_vals_list = np.load(f)
        with open(path_auto_lambda + '/' + 'all_lab_Accs_' + str(assumed_k) + '.npy', 'rb') as f:
            all_lab_Accs = np.load(f)
        auto_lambda_val = lambda_vals_list[1:][np.argmax(all_lab_Accs[1:])]
    print("Obtained lambda:", auto_lambda_val, "\n")




    ################
    # use modified PIM here. The PIM had an extra linear layer as the encoder
    ################

    pim = PIM_partitioner(num_features=len(feature_map_preds[0]),
                          num_classes=assumed_k,
                          r=1.).to(args.device_name) # imagenet 0.0, cub, 0.5,   others 1

    for name, param in pim.named_parameters():
        if name == "partitioner.weight":
            pim.partitioner.weight.data = prototypes.type_as(param)

    # Optimizer
    criterion = nn.CrossEntropyLoss()

    if args.dataset == 'scars':
        wd = 0.02
    elif args.dataset == 'cub':
        wd = 0.02
    elif args.dataset == 'herbarium':
        wd = 0.02
    elif args.dataset == 'cifar100':
        wd = 0.005
    elif args.dataset == 'cifar10':
        wd = 0.05
    elif args.dataset == 'imagenet_100':
        wd = 0.005
    else:
        wd = 1e-2
    wd /= 2

    optimizer = optim.Adam([
        {'params': pim.parameters()},
                            ], lr=0.001, weight_decay=wd)

    print(f'using weight decay {r}')

    # PIM training
    print('PIM training...')
    mb_size = len(feature_map_preds)

    mb_gt_labels = gt_labels
    mb_lab_mask = mask_lab
    mb_lab_points = mb_gt_labels[mb_lab_mask]
    seen_N = mb_lab_points.max() + 1
    print('no_seen_N', seen_N)

    e = 2.220446049250313e-16


    for epoch in range(args.epochs):  # loop over the feature map set multiple times
        running_loss = 0.0

        for mb_id in range(0, int(len(feature_map_preds) / mb_size)):

            mb_inputs = from_numpy_to_torch(feature_map_preds[mb_id * mb_size:(mb_id + 1) * mb_size],
                                            torch_device).float() # z
            mb_gt_labels = from_numpy_to_torch(gt_labels[mb_id * mb_size:(mb_id + 1) * mb_size], torch_device)
            mb_lab_mask = from_numpy_to_torch(mask_lab[mb_id * mb_size:(mb_id + 1) * mb_size], torch_device)
            mb_lab_points = mb_gt_labels[mb_lab_mask]  # We only use labels information for Z_L subset, y^L
            unseen_lab = mb_gt_labels[~mb_lab_mask]
            optimizer.zero_grad()
            # optimizer_affine.zero_grad()

            # mb_inputs = affine(mb_inputs, mb_inputs)
            feat, mb_logits_outputs = pim(mb_inputs)
            soft_mb_logits_outputs = F.softmax(mb_logits_outputs, dim=1)
            loss = 0.

            soft_mb_logits_outputs_seen_preds = soft_mb_logits_outputs[mb_lab_mask] # bar(logits)^L
            soft_mb_logits_outputs_unseen_preds = soft_mb_logits_outputs[~mb_lab_mask] # bar(logits)^U
            mb_logits_outputs_seen_preds = mb_logits_outputs[mb_lab_mask] # bar(y)^L
            mb_logits_outputs_unseen_preds = mb_logits_outputs[~mb_lab_mask] # bar(y)^U

            h_u_loss = - ((soft_mb_logits_outputs_unseen_preds + e) * torch.log(
                soft_mb_logits_outputs_unseen_preds + e)).sum(1).mean() * auto_lambda_val

            ce_l_loss = criterion(mb_logits_outputs_seen_preds, mb_lab_points.to(torch.int64))
            h_u_l_cond_z = (soft_mb_logits_outputs.mean(0) * torch.log(soft_mb_logits_outputs.mean(0) + e)).sum()
            loss += ce_l_loss + h_u_loss + h_u_l_cond_z

            A = torch.matmul(mb_logits_outputs.permute(1, 0), feat)
            A = F.softmax(A, dim=1)
            A_entropy = (
                    (A.mean(0) + 1e-12)
                    * torch.log(A.mean(0) + 1e-12)
            ).sum()

            A_dim_entropy = (
                -((A + 1e-12)
                  * torch.log(
                            A + 1e-12
                        )
                  )
                .sum(1)
                .mean()
            )

            a = 0.001
            if args.dataset == 'cifar100':
                a = 0.0025
            elif args.dataset == 'cub':
                a = 0.1
            elif args.dataset == 'scars':
                a = 0.001
            elif args.dataset == 'cifar10':
                a = 0.01
            elif args.dataset == 'herbarium':
                a = 0.2
            elif args.dataset == 'imagenet_100':
                a = 0.01
            loss += (A_entropy + A_dim_entropy) * a * auto_lambda_val

            loss.backward()
            optimizer.step()
            running_loss += loss.item()


    print(running_loss)
    # PIM partitioning evaluation
    with torch.no_grad():
        _, outputs = pim(from_numpy_to_torch(feature_map_preds, torch_device).float())
        _, predicted = torch.max(outputs.data, 1)
        unlab_preds = np.asarray(from_torch_to_numpy(predicted)[~mask_lab], dtype=int)
        lab_preds = np.asarray(from_torch_to_numpy(predicted)[mask_lab], dtype=int)

    unlab_gt_labs = np.asarray(gt_labels[~mask_lab], dtype=int)
    seen_mask = mask_old_new[~mask_lab].astype(bool)
    partitioning_eval(unlab_gt_labs, unlab_preds, seen_mask, args.dataset, path_k_strat)

    sum = 0
    for i in range(len(mb_lab_points)):
        if lab_preds[i] == mb_lab_points[i]:
            sum +=1
    print(sum/len(mb_lab_points))
    ###



def sample_for_visual(soft_mb_logits_outputs_seen_preds, soft_mb_logits_outputs_unseen_preds, seen_lab, unseen_lab,
                      visual_class_num=2, visual_sample_num=4):
    soft_mb_logits_outputs_seen_preds, soft_mb_logits_outputs_unseen_preds, seen_lab, unseen_lab = \
        from_torch_to_numpy(soft_mb_logits_outputs_seen_preds.detach()),  \
            from_torch_to_numpy(soft_mb_logits_outputs_unseen_preds.detach()), \
            from_torch_to_numpy(seen_lab.detach()), \
            from_torch_to_numpy(unseen_lab.detach()),

    shared_labels = np.isin(seen_lab, unseen_lab, )
    shared_labels = np.unique(seen_lab[shared_labels])[:visual_class_num]

    unseen_labels = (1 - np.isin(unseen_lab, shared_labels).astype(int)).astype(bool)
    unseen_labels = np.unique(unseen_lab[unseen_labels])[:visual_class_num]
    all_labels = np.concatenate([shared_labels, unseen_labels,], axis=0)
    print(all_labels)

    visual_shared_labels_seen_list = []
    visual_shared_labels_seen_list_label = []
    visual_shared_labels_unseen_list = []
    visual_shared_labels_unseen_list_label= []
    for i in shared_labels:
        visual_shared_labels_seen = np.isin(soft_mb_logits_outputs_seen_preds, [i])[:visual_sample_num]
        visual_shared_labels_seen_list.append(visual_shared_labels_seen)
        visual_shared_labels_seen_list_label.append(i)

    for i in all_labels:
        visual_shared_labels_unseen = np.isin(soft_mb_logits_outputs_unseen_preds, [i])[:visual_sample_num]
        visual_shared_labels_unseen_list.append(visual_shared_labels_unseen)
        visual_shared_labels_unseen_list_label.append(i)

    visual_shared_labels_seen_list = np.vstack(visual_shared_labels_seen_list)
    visual_shared_labels_unseen_list = np.vstack(visual_shared_labels_unseen_list)
    visual_shared_labels_seen_list_label= np.array(visual_shared_labels_seen_list_label)
    visual_shared_labels_unseen_list_label = np.array(visual_shared_labels_unseen_list_label)

    print(visual_shared_labels_seen_list_label)
    print(visual_shared_labels_seen_list.shape, visual_shared_labels_seen_list_label.shape)

    return visual_shared_labels_seen_list, visual_shared_labels_unseen_list, \
        visual_shared_labels_seen_list_label, visual_shared_labels_unseen_list_label

if __name__ == '__main__':
    main()