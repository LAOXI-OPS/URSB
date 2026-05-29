import copy
import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class RecDataset(Dataset):
    def __init__(self, df):
        self.seqs = df['seq'].tolist()
        self.nexts = df['next'].tolist()
        self.has_negative_samples = 'negative_samples' in df.columns
        self.negative_samples = df['negative_samples'].tolist() if self.has_negative_samples else None

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        sample = {
            'seq': torch.tensor(self.seqs[idx], dtype=torch.long),
            'next': torch.tensor(self.nexts[idx], dtype=torch.long),
        }
        if self.has_negative_samples:
            sample['negative_samples'] = torch.tensor(self.negative_samples[idx], dtype=torch.long)
        return sample


def build_negative_samples(df, item_num, num_negatives):
    if num_negatives <= 0:
        raise ValueError(f"num_negatives must be > 0, got {num_negatives}")

    if 'negative_samples' in df.columns and df['negative_samples'].notna().all():
        return df

    rng = np.random.default_rng(42)
    out_df = df.copy()
    out_df['negative_samples'] = None
    max_item_id = int(item_num)

    for i in range(len(out_df)):
        pos_item = int(out_df['next'].iloc[i])
        history_items = {int(x) for x in out_df['seq'].iloc[i] if int(x) > 0}
        blocked = history_items | {pos_item}

        candidates = [pos_item]
        while len(candidates) < num_negatives + 1:
            sampled = int(rng.integers(1, max_item_id + 1))
            if sampled not in blocked and sampled not in candidates:
                candidates.append(sampled)

        out_df.at[out_df.index[i], 'negative_samples'] = candidates

    return out_df


def compute_eval_metrics(scores_rec_sb, target, batch, metric_ks, device, eval_mode):
    if eval_mode == 'negative':
        candidate_ids = batch['negative_samples'].to(device, non_blocking=(device == 'cuda'))
        row_indices = torch.arange(candidate_ids.size(0), device=device).unsqueeze(1)
        candidate_scores = scores_rec_sb[row_indices, candidate_ids]
        candidate_labels = torch.zeros((candidate_ids.size(0), 1), dtype=torch.long, device=device)
        return hrs_and_ndcgs_k(candidate_scores, candidate_labels, metric_ks)

    return hrs_and_ndcgs_k(scores_rec_sb, target, metric_ks)


def optimizers(model, args):
    if args.optimizer.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")


def cal_hr(label, predict, ks):
    max_ks = min(max(ks), predict.shape[-1])
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    hr = [hit[:, :ks[i]].sum().item() / label.size()[0] for i in range(len(ks))]
    return hr


def cal_ndcg(label, predict, ks):
    max_ks = min(max(ks), predict.shape[-1])
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = (label == topk_predict).int()
    ndcg = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k - 1), device=predict.device))
        predict_dcg = dcg(hit[:, :k])
        ndcg.append((predict_dcg / max_dcg).mean().item())
    return ndcg


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1, device=hit.device) + 1).unsqueeze(0)
    rel = (hit / log2).sum(dim=-1)
    return rel


def hrs_and_ndcgs_k(scores, labels, ks):
    metrics = {}
    labels_det = labels.detach()
    scores_det = scores.detach()
    ndcg = cal_ndcg(labels_det, scores_det, ks)
    hr = cal_hr(labels_det, scores_det, ks)
    for k, ndcg_temp, hr_temp in zip(ks, ndcg, hr):
        metrics['HR@%d' % k] = hr_temp
        metrics['NDCG@%d' % k] = ndcg_temp
    return metrics


def evaluate_model(eval_data, model_joint, args, logger, forward_flag, split_name="Test"):
    device = args.device
    metric_ks = args.metric_ks
    eval_mode = getattr(args, 'eval_mode', 'full')
    num_negatives = getattr(args, 'num_negatives', 100)

    if eval_mode == 'negative':
        eval_data = build_negative_samples(eval_data, args.item_num, num_negatives)
        logger.info('Evaluation mode: negative sampling, num_negatives=%s', num_negatives)
    else:
        logger.info('Evaluation mode: full ranking')

    pin_memory = device == 'cuda'
    eval_loader = DataLoader(
        RecDataset(eval_data),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=pin_memory,
    )

    model_joint.eval()
    with torch.no_grad():
        eval_metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
        for batch in eval_loader:
            seq = batch['seq'].to(device, non_blocking=pin_memory)
            target = batch['next'].unsqueeze(1).to(device, non_blocking=pin_memory)
            rep_sb, _, _ = model_joint(seq, target, forward_flag, train_flag=False)
            scores_rec_sb = model_joint.sb_rep_pre(rep_sb, forward_flag)
            metrics = compute_eval_metrics(scores_rec_sb, target, batch, metric_ks, device, eval_mode)
            for k, v in metrics.items():
                eval_metrics_dict[k].append(v)

    eval_metrics_mean = {}
    for key, values in eval_metrics_dict.items():
        eval_metrics_mean[key] = round(np.mean(values) * 100, 4)

    print(f"{split_name}------------------------------------------------------")
    logger.info("%s------------------------------------------------------", split_name)
    print(eval_metrics_mean)
    logger.info(eval_metrics_mean)
    return eval_metrics_mean


def model_train(train_data, val_data, test_data, model_joint, args, logger, forward_flag,
                eval_enabled=True, test_enabled=True):
    epochs = args.epochs
    device = args.device
    metric_ks = args.metric_ks
    model_joint = model_joint.to(device)

    is_parallel = args.num_gpu > 1
    if is_parallel:
        model_joint = nn.DataParallel(model_joint)

    best_model = copy.deepcopy(model_joint)
    optimizer = optimizers(model_joint, args)

    best_metrics_dict = {'Best_HR@5': 0, 'Best_NDCG@5': 0, 'Best_HR@10': 0, 'Best_NDCG@10': 0,
                         'Best_HR@20': 0, 'Best_NDCG@20': 0}
    best_epoch = {'Best_epoch_HR@5': 0, 'Best_epoch_NDCG@5': 0, 'Best_epoch_HR@10': 0,
                  'Best_epoch_NDCG@10': 0, 'Best_epoch_HR@20': 0, 'Best_epoch_NDCG@20': 0}
    bad_count = 0

    eval_mode = getattr(args, 'eval_mode', 'full')
    num_negatives = getattr(args, 'num_negatives', 100)
    if eval_enabled or test_enabled:
        if eval_mode == 'negative':
            val_data = build_negative_samples(val_data, args.item_num, num_negatives)
            test_data = build_negative_samples(test_data, args.item_num, num_negatives)
            logger.info('Evaluation mode: negative sampling, num_negatives=%s', num_negatives)
        else:
            logger.info('Evaluation mode: full ranking')

    pin_memory = device == 'cuda'
    train_loader = DataLoader(
        RecDataset(train_data), batch_size=args.batch_size, shuffle=True,
        drop_last=True, pin_memory=pin_memory)
    val_loader = DataLoader(
        RecDataset(val_data), batch_size=args.batch_size, shuffle=False,
        drop_last=True, pin_memory=pin_memory)
    test_loader = DataLoader(
        RecDataset(test_data), batch_size=args.batch_size, shuffle=False,
        drop_last=True, pin_memory=pin_memory)

    for epoch_temp in range(epochs):
        model_joint.train()
        flag_update = 0

        for batch in train_loader:
            optimizer.zero_grad()
            seq = batch['seq'].to(device, non_blocking=pin_memory)
            target = batch['next'].unsqueeze(1).to(device, non_blocking=pin_memory)
            sb_rep, x1, pred_x1 = model_joint(seq, target, forward_flag, train_flag=True)
            loss_sb_value = model_joint.loss_sb_ce(sb_rep, target, forward_flag, x1, pred_x1)
            loss_sb_value.backward()
            optimizer.step()

        print('Epoch: {}'.format(epoch_temp))
        logger.info('Epoch: {}'.format(epoch_temp))

        if eval_enabled and epoch_temp != 0 and epoch_temp % args.eval_interval == 0:
            print('start predicting: ', datetime.datetime.now())
            logger.info('start predicting: {}'.format(datetime.datetime.now()))
            eval_model = model_joint
            eval_model.eval()
            with torch.no_grad():
                metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [],
                                'HR@20': [], 'NDCG@20': []}
                for batch in val_loader:
                    seq = batch['seq'].to(device, non_blocking=pin_memory)
                    target = batch['next'].unsqueeze(1).to(device, non_blocking=pin_memory)
                    rep_sb, _, _ = eval_model(seq, target, forward_flag, train_flag=False)
                    scores_rec_sb = eval_model.sb_rep_pre(rep_sb, forward_flag)
                    metrics = compute_eval_metrics(scores_rec_sb, target, batch, metric_ks, device, eval_mode)
                    for k, v in metrics.items():
                        metrics_dict[k].append(v)

            metrics_dict_mean = {}
            for key, values in metrics_dict.items():
                metrics_dict_mean[key] = round(np.mean(values) * 100, 4)

            compact = ", ".join(f"{k}:{v}" for k, v in metrics_dict_mean.items())
            print(f"Epoch {epoch_temp}: {compact}")
            logger.info("Eval@Epoch %s: %s", epoch_temp, metrics_dict_mean)

            for key, values in metrics_dict.items():
                values_mean = round(np.mean(values) * 100, 4)
                if values_mean > best_metrics_dict['Best_' + key]:
                    flag_update += 1
                    bad_count = 0
                    best_metrics_dict['Best_' + key] = values_mean
                    best_epoch['Best_epoch_' + key] = epoch_temp

            if flag_update == 0:
                bad_count += 1
            else:
                logger.info(best_metrics_dict)
                logger.info(best_epoch)
                if flag_update >= 3:
                    best_model = copy.deepcopy(model_joint)
                    if forward_flag:
                        save_dir = os.path.join('.', 'saved_model', f"{args.s_dataset}_{args.t_dataset}")
                        os.makedirs(save_dir, exist_ok=True)
                        torch.save(model_joint.state_dict(), os.path.join(save_dir, 'model.pth'))
            if bad_count >= args.patience:
                break

    logger.info(best_metrics_dict)
    logger.info(best_epoch)

    best_model = copy.deepcopy(model_joint)

    if forward_flag:
        save_dir = os.path.join('.', 'saved_model', f"{args.s_dataset}_{args.t_dataset}")
        os.makedirs(save_dir, exist_ok=True)
        model_to_save = model_joint.module if hasattr(model_joint, 'module') else model_joint
        torch.save(model_to_save.state_dict(), os.path.join(save_dir, 'model.pth'))

    test_metrics_dict_mean = {}
    if test_enabled:
        with torch.no_grad():
            test_metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [],
                                 'HR@20': [], 'NDCG@20': []}
            for batch in test_loader:
                seq = batch['seq'].to(device, non_blocking=pin_memory)
                target = batch['next'].unsqueeze(1).to(device, non_blocking=pin_memory)
                rep_sb, _, _ = best_model(seq, target, forward_flag, train_flag=False)
                scores_rec_sb = best_model.sb_rep_pre(rep_sb, forward_flag)
                metrics = compute_eval_metrics(scores_rec_sb, target, batch, metric_ks, device, eval_mode)
                for k, v in metrics.items():
                    test_metrics_dict[k].append(v)

        for key, values in test_metrics_dict.items():
            test_metrics_dict_mean[key] = round(np.mean(values) * 100, 4)

        print('Test------------------------------------------------------')
        logger.info('Test------------------------------------------------------')
        compact = ", ".join(f"{k}:{v}" for k, v in test_metrics_dict_mean.items())
        print(compact)
        logger.info(test_metrics_dict_mean)

    logger.info('Best Eval---------------------------------------------------------')
    logger.info(best_metrics_dict)
    logger.info(best_epoch)
    return best_model, test_metrics_dict_mean
