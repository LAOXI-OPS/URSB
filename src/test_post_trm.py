"""验证 post-transformer 独立表征能力：rep_item 不混合 x_t，直接过 post_trm 做训练+预测"""
import argparse
import math
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pandas as pd

from sbrec import TransformerRep, LayerNorm


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


DATASET_ALIASES = {
    'movie': 'movie', 'video': 'video', 'beauty': 'beauty',
    'books': 'books', 'cloth': 'cloth', 'elec': 'elec',
    'electronics': 'elec', 'food': 'food', 'phone': 'phone',
    'music': 'music', 'sports_and_outdoors': 'sports_and_outdoors',
    'sports': 'sports_and_outdoors', 'toys_and_games': 'toys_and_games',
    'toys': 'toys_and_games',
}

TXT_SOURCE_MAP = {
    'beauty': 'Beauty.txt',
    'sports_and_outdoors': 'Sports_and_Outdoors.txt',
    'toys_and_games': 'Toys_and_Games.txt',
}


def convert_iclrec_txt_to_df(txt_path, out_dir, target_length=8):
    rows = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            items = [int(x) for x in parts[1:]]
            if len(items) < 5:
                continue
            if len(items) > target_length + 1:
                items = items[-(target_length + 1):]
            target_item = items[-1]
            input_seq = items[:-1]
            seq_length = len(input_seq)
            if seq_length < target_length:
                input_seq = [0] * (target_length - seq_length) + input_seq
            rows.append((input_seq, seq_length, target_item))
    if not rows:
        raise ValueError(f"No valid rows parsed from {txt_path}")
    np.random.shuffle(rows)
    train_size = int(len(rows) * 0.8)
    val_size = int(len(rows) * 0.1)
    train_data = rows[:train_size]
    val_data = rows[train_size:train_size + val_size]
    test_data = rows[train_size + val_size:]
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_data, columns=['seq', 'len_seq', 'next']).to_pickle(out_dir / 'train.df')
    pd.DataFrame(val_data, columns=['seq', 'len_seq', 'next']).to_pickle(out_dir / 'val.df')
    pd.DataFrame(test_data, columns=['seq', 'len_seq', 'next']).to_pickle(out_dir / 'test.df')


def ensure_dataset_ready(dataset_name, max_len):
    data_dir = Path(__file__).resolve().parent.parent / 'data' / dataset_name
    train_path = data_dir / 'train.df'
    val_path = data_dir / 'val.df'
    test_path = data_dir / 'test.df'
    if train_path.exists() and val_path.exists() and test_path.exists():
        return str(train_path), str(val_path), str(test_path)
    if dataset_name not in TXT_SOURCE_MAP:
        missing = [str(p) for p in (train_path, val_path, test_path) if not p.exists()]
        raise FileNotFoundError(f"Missing dataset files for '{dataset_name}': {missing}")
    txt_name = TXT_SOURCE_MAP[dataset_name]
    txt_path = Path(__file__).resolve().parent.parent.parent / 'datasets' / 'ICLRec_seq_txt' / txt_name
    if not txt_path.exists():
        raise FileNotFoundError(f"Cannot find source txt file: {txt_path}")
    convert_iclrec_txt_to_df(txt_path, data_dir, target_length=max_len)
    return str(train_path), str(val_path), str(test_path)


def infer_item_num(*dfs):
    max_item = 0
    for df in dfs:
        if len(df) == 0:
            continue
        seq_max = max(max(seq) for seq in df['seq'])
        next_max = int(df['next'].max())
        max_item = max(max_item, seq_max, next_max)
    return int(max_item)


class SeqDataset(Dataset):
    def __init__(self, df):
        self.seq = df['seq'].values
        self.nxt = df['next'].values

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return torch.tensor(self.seq[idx]), torch.tensor([self.nxt[idx]])


class PlainPostTrm(nn.Module):
    """rep_item → Post-Transformer → mean pool → pred"""
    def __init__(self, item_num, hidden_size, num_blocks, dropout, emb_dropout):
        super().__init__()
        self.embedding = nn.Embedding(item_num + 1, hidden_size)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.norm_in = LayerNorm(hidden_size)
        self.post_trm = TransformerRep(
            argparse.Namespace(hidden_size=hidden_size, num_blocks=num_blocks, dropout=dropout))
        self.post_norm = LayerNorm(hidden_size)

    def forward(self, seq):
        mask = (seq > 0).float()
        x = self.embedding(seq)
        x = self.emb_dropout(x)
        x = self.norm_in(x)
        x = self.post_trm(x, mask)
        x = self.post_norm(x)
        return x.mean(dim=1)


def evaluate(model, loader, item_embeddings, device, ks=[5, 10, 20]):
    model.eval()
    hr = {k: [] for k in ks}
    ndcg = {k: [] for k in ks}
    with torch.no_grad():
        for seq, target in loader:
            seq, target = seq.to(device), target.to(device)
            rep = model(seq)
            scores = rep @ item_embeddings.weight.t()
            # exclude padding idx 0
            scores[:, 0] = -1e9
            _, topk = scores.topk(max(ks), dim=1)
            target_exp = target.expand(-1, max(ks))
            hits = (topk == target_exp).float()
            for k in ks:
                h = hits[:, :k].sum(dim=1)
                hr[k].append(h.mean().item())
                idx = (h > 0).float()
                rank = hits[:, :k].float().argmax(dim=1).float() + 1
                dcg = idx / torch.log2(rank + 1)
                idcg = 1.0 / torch.log2(torch.tensor(2.0, device=device))
                ndcg[k].append((dcg / idcg).mean().item())
    return {f'HR@{k}': np.mean(hr[k]) * 100 for k in ks}, \
           {f'NDCG@{k}': np.mean(ndcg[k]) * 100 for k in ks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='books')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_blocks', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--emb_dropout', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=8)
    args = parser.parse_args()

    fix_random_seed_as(1997)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_path, val_path, test_path = ensure_dataset_ready(args.dataset, args.max_len)
    train_df = pd.read_pickle(train_path)
    val_df = pd.read_pickle(val_path)
    test_df = pd.read_pickle(test_path)
    item_num = infer_item_num(train_df, val_df, test_df)

    train_loader = DataLoader(SeqDataset(train_df), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SeqDataset(val_df), batch_size=args.batch_size)
    test_loader = DataLoader(SeqDataset(test_df), batch_size=args.batch_size)

    model = PlainPostTrm(item_num, args.hidden_size, args.num_blocks, args.dropout, args.emb_dropout).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Dataset: {args.dataset}, items: {item_num}")
    print(f"Train samples: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # ---- Before training ----
    hr_before, ndcg_before = evaluate(model, test_loader, model.embedding, device)
    print(f"\nBefore training (random init) — Test:")
    print(f"  HR@10={hr_before['HR@10']:.2f}  HR@20={hr_before['HR@20']:.2f}")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for seq, target in train_loader:
            seq, target = seq.to(device), target.to(device)
            rep = model(seq)
            scores = rep @ model.embedding.weight.t()
            loss = loss_fn(scores, target.squeeze(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: loss={total_loss / len(train_loader):.4f}")

    # ---- After training ----
    hr_after, ndcg_after = evaluate(model, test_loader, model.embedding, device)
    print(f"\nAfter {args.epochs} epoch(s) training — Test:")
    print(f"  HR@5={hr_after['HR@5']:.2f}  NDCG@5={ndcg_after['NDCG@5']:.2f}")
    print(f"  HR@10={hr_after['HR@10']:.2f}  NDCG@10={ndcg_after['NDCG@10']:.2f}")
    print(f"  HR@20={hr_after['HR@20']:.2f}  NDCG@20={ndcg_after['NDCG@20']:.2f}")


if __name__ == '__main__':
    main()
