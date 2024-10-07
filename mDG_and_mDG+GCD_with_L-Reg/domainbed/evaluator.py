import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from domainbed.lib.fast_data_loader import FastDataLoader
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import Iterable
# if torch.cuda.is_available():
#     device = "cuda"
# else:
#     device = "cpu"
device = "cuda"


def cluster_acc_v2(y_true, y_pred, mask,  mask_range=-1):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_sum_assignment(w.max() - w)
    ind = np.vstack(ind).T
    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])

    new_acc /= total_new_instances

    # new_acc_fix = cluster_acc_v2_fix(y_true, y_pred, mask, mask_range=mask_range)
    # return total_acc, old_acc, new_acc, new_acc_fix
    return total_acc, old_acc, new_acc

def cluster_acc_v2_fix(y_true, y_pred, mask, mask_range=-1):
    y_true = y_true.astype(int)
    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_sum_assignment(w.max() - w)
    ind = np.vstack(ind).T
    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        if i not in old_classes_gt:
            new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])

    new_acc /= total_new_instances
    return new_acc

def accuracy_from_loader(algorithm, loader, weights, debug=False,mask_range=-1):
    # if mask_range <0:
    #     return mdg_accuracy_from_loader(algorithm, loader, weights, debug=debug)
    # else:
    return gcd_accuracy_from_loader(algorithm, loader, weights, debug=debug,mask_range=mask_range)

def gcd_accuracy_from_loader(algorithm, loader, weights, debug=False,mask_range=-1):
    correct = 0
    total = 0
    losssum = 0.0
    weights_offset = 0

    algorithm.eval()

    all_pred_y = []
    all_gt_y = []
    all_weights = []
    for i, batch in enumerate(loader):
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        with torch.no_grad():
            logits = algorithm.predict(x)
            loss = F.cross_entropy(logits, y).item()

        B = len(x)
        losssum += loss * B
        if logits.size(1) == 1:
            pred_y = logits.gt(0)
        else:
            pred_y = logits.argmax(1)

        all_pred_y.append(pred_y.cpu().numpy())
        all_gt_y.append(y.cpu().numpy())

    all_pred_y = np.concatenate(all_pred_y, axis=0)
    all_gt_y = np.concatenate(all_gt_y, axis=0)
    mask = np.zeros(len(all_pred_y))
    if mask_range > 0:
        mask[np.where(all_gt_y <= mask_range)] = 1
    else:
        mask = np.ones(len(all_pred_y))
    mask = mask.astype(bool)
    total_acc, old_acc, new_acc = cluster_acc_v2(all_gt_y,all_pred_y, mask, mask_range)
    total = len(all_pred_y)
    algorithm.train()
    loss = losssum / total
    return np.array([total_acc, old_acc, new_acc]), loss


def mdg_accuracy_from_loader(algorithm, loader, weights, debug=False):
    correct = 0
    total = 0
    losssum = 0.0
    weights_offset = 0

    algorithm.eval()


    for i, batch in enumerate(loader):
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        with torch.no_grad():
            logits = algorithm.predict(x)
            loss = F.cross_entropy(logits, y).item()

        B = len(x)
        losssum += loss * B

        if weights is None:
            batch_weights = torch.ones(len(x))
        else:
            batch_weights = weights[weights_offset : weights_offset + len(x)]
            weights_offset += len(x)
        batch_weights = batch_weights.to(device)
        if logits.size(1) == 1:
            correct += (logits.gt(0).eq(y).float() * batch_weights).sum().item()
        else:
            correct += (logits.argmax(1).eq(y).float() * batch_weights).sum().item()
        total += batch_weights.sum().item()

        if debug:
            break

    algorithm.train()

    acc = correct / total
    loss = losssum / total
    return acc, loss


def accuracy(algorithm, loader_kwargs, weights, **kwargs):
    if isinstance(loader_kwargs, dict):
        loader = FastDataLoader(**loader_kwargs)
    elif isinstance(loader_kwargs, FastDataLoader):
        loader = loader_kwargs
    else:
        raise ValueError(loader_kwargs)
    return accuracy_from_loader(algorithm, loader, weights, **kwargs)


class Evaluator:
    def __init__(
        self, test_envs, eval_meta, n_envs, logger, evalmode="fast", debug=False, target_env=None,mask_range=-1
    ):
        all_envs = list(range(n_envs))
        train_envs = sorted(set(all_envs) - set(test_envs))
        self.test_envs = test_envs
        self.train_envs = train_envs
        self.eval_meta = eval_meta
        self.n_envs = n_envs
        self.logger = logger
        self.evalmode = evalmode
        self.debug = debug
        self.mask_range = mask_range

        if target_env is not None:
            self.set_target_env(target_env)

    def set_target_env(self, target_env):
        """When len(test_envs) == 2, you can specify target env for computing exact test acc."""
        self.test_envs = [target_env]

    def evaluate(self, algorithm, ret_losses=False):
        n_train_envs = len(self.train_envs)
        n_test_envs = len(self.test_envs)
        assert n_test_envs == 1
        summaries = collections.defaultdict(float)
        # for key order
        summaries["test_in"] = 0.0
        summaries["test_out"] = 0.0
        summaries["train_in"] = 0.0
        summaries["train_out"] = 0.0
        accuracies = {}
        losses = {}

        for name, loader_kwargs, weights in self.eval_meta:
            # env\d_[in|out]
            env_name, inout = name.split("_")
            env_num = int(env_name[3:])

            skip_eval = self.evalmode == "fast" and inout == "in" and env_num not in self.test_envs
            if skip_eval:
                continue

            is_test = env_num in self.test_envs
            acc, loss = accuracy(algorithm, loader_kwargs, weights, debug=self.debug, mask_range=self.mask_range)
            accuracies[name] = acc
            losses[name] = loss
            if env_num in self.train_envs:
                summaries["train_" + inout] += acc / n_train_envs
                if inout == "out":
                    summaries["tr_" + inout + "loss"] += loss / n_train_envs
            elif is_test:
                summaries["test_" + inout] += acc / n_test_envs

        if ret_losses:
            return accuracies, summaries, losses
        else:
            return accuracies, summaries
