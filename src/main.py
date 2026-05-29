import argparse
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn

from model import create_model_sb, AttSBModel
from trainer import model_train, evaluate_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DATASET_ALIASES = {
    'movie': 'movie',
    'movie_new': 'movie_new',
    'movie_new_5': 'movie_new_5',
    'video': 'video',
    'video_new': 'video_new',
    'beauty': 'beauty',
    'books': 'books',
    'books_5': 'books_5',
    'books_v1': 'books_v1',
    'cloth': 'cloth',
    'elec': 'elec',
    'electronics': 'elec',
    'food': 'food',
    'phone': 'phone',
    'music': 'music',
    'sports_and_outdoors': 'sports_and_outdoors',
    'sports': 'sports_and_outdoors',
    'toys_and_games': 'toys_and_games',
    'toys': 'toys_and_games',
}

TXT_SOURCE_MAP = {
    'beauty': 'Beauty.txt',
    'sports_and_outdoors': 'Sports_and_Outdoors.txt',
    'toys_and_games': 'Toys_and_Games.txt',
}

parser = argparse.ArgumentParser()
parser.add_argument('--s_dataset', default='movie')
parser.add_argument('--t_dataset', default='video')
parser.add_argument('--p', type=float, default=0.3)
parser.add_argument('--w', type=float, default=1.0)
parser.add_argument('--log_file', default='log/')
parser.add_argument('--random_seed', type=int, default=1997)
parser.add_argument('--max_len', type=int, default=8)
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument("--hidden_size", default=128, type=int)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--emb_dropout', type=float, default=0.3)
parser.add_argument('--num_blocks', type=int, default=4)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--ft_epochs', type=int, default=2)
parser.add_argument('--metric_ks', nargs='+', type=int, default=[5, 10, 20])
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--momentum', type=float, default=None)
parser.add_argument('--schedule_sampler_name', type=str, default='lossaware')
parser.add_argument('--diffusion_steps', type=int, default=32)
parser.add_argument('--lambda_uncertainty', type=float, default=0.001)
parser.add_argument('--lambda_x1', type=float, default=0.1)
parser.add_argument('--rescale_timesteps', default=True)
parser.add_argument('--interval', type=int, default=1000)
parser.add_argument('--beta_min', type=float, default=0.01)
parser.add_argument('--beta_max', type=float, default=50)
parser.add_argument('--sample_steps', type=int, default=32)

parser.add_argument('--eval_interval', type=int, default=5)
parser.add_argument('--patience', type=int, default=2)
parser.add_argument('--eval_mode', type=str, default='full', choices=['full', 'negative'])
parser.add_argument('--num_negatives', type=int, default=100)
parser.add_argument('--description', type=str, default='SB_norm_score')
args = parser.parse_args()

args.s_dataset = args.s_dataset.strip().lower()
args.t_dataset = args.t_dataset.strip().lower()
if args.s_dataset not in DATASET_ALIASES:
    supported = ', '.join(sorted(DATASET_ALIASES.keys()))
    raise ValueError(f"Unsupported source dataset '{args.s_dataset}'. Supported values: {supported}")
if args.t_dataset not in DATASET_ALIASES:
    supported = ', '.join(sorted(DATASET_ALIASES.keys()))
    raise ValueError(f"Unsupported target dataset '{args.t_dataset}'. Supported values: {supported}")
args.s_dataset = DATASET_ALIASES[args.s_dataset]
args.t_dataset = DATASET_ALIASES[args.t_dataset]

if not os.path.exists(args.log_file):
    os.makedirs(args.log_file)
if not os.path.exists(args.log_file + args.s_dataset + "_" + args.t_dataset):
    os.makedirs(args.log_file + args.s_dataset + "_" + args.t_dataset)

logging.basicConfig(
    level=logging.INFO,
    filename=args.log_file + args.s_dataset + "_" + args.t_dataset + '/'
    + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.log',
    datefmt='%Y/%m/%d %H:%M:%S',
    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s',
    filemode='w',
)
logger = logging.getLogger(__name__)


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def item_num_create(args, source_item_num, target_item_num):
    args.source_item_num = source_item_num
    args.target_item_num = target_item_num
    args.item_num = source_item_num
    return args


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
        raise FileNotFoundError(
            f"Missing dataset files for '{dataset_name}': {missing}. "
            "Please prepare train.df/val.df/test.df first."
        )

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


def main(args):
    fix_random_seed_as(args.random_seed)

    if args.t_dataset in {"toys_and_games", "sports_and_outdoors"}:
        args.dropout = 0.3
    logger.info(args)

    s_train_path, s_val_path, s_test_path = ensure_dataset_ready(args.s_dataset, args.max_len)
    t_train_path, t_val_path, t_test_path = ensure_dataset_ready(args.t_dataset, args.max_len)

    s_tra_data = pd.read_pickle(s_train_path)
    s_val_data = pd.read_pickle(s_val_path)
    s_test_data = pd.read_pickle(s_test_path)
    t_tra_data = pd.read_pickle(t_train_path)
    t_val_data = pd.read_pickle(t_val_path)
    t_test_data = pd.read_pickle(t_test_path)

    source_item_num = infer_item_num(s_tra_data, s_val_data, s_test_data)
    target_item_num = infer_item_num(t_tra_data, t_val_data, t_test_data)

    args = item_num_create(args, source_item_num, target_item_num)

    sb_rec = create_model_sb(args)
    rec_sb_joint_model = AttSBModel(sb_rec, args)

    forward_flag = True
    model_train(
        s_tra_data, s_val_data, s_test_data, rec_sb_joint_model, args, logger, forward_flag)

    ckpt_path = os.path.join(".", "saved_model", f"{args.s_dataset}_{args.t_dataset}", "model.pth")
    rec_sb_joint_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

    args.item_num = target_item_num
    args.eval_interval = 2
    args.patience = args.ft_epochs
    args.epochs = args.ft_epochs
    forward_flag = False

    fine_tuned_model, _ = model_train(
        t_tra_data, t_val_data, t_test_data, rec_sb_joint_model, args, logger,
        forward_flag, eval_enabled=True, test_enabled=True)

    args.item_num = source_item_num
    forward_flag = True
    evaluate_model(s_val_data, fine_tuned_model, args, logger, forward_flag,
                   split_name="Source Val (Post-Finetune)")
    evaluate_model(s_test_data, fine_tuned_model, args, logger, forward_flag,
                   split_name="Source Test (Post-Finetune)")


if __name__ == '__main__':
    main(args)
