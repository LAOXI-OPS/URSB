import argparse
import random

import numpy as np
import pandas as pd
import torch

from trainer import hrs_and_ndcgs_k


def build_negative_samples(df, item_num):
    if "negative_samples" in df.columns and df["negative_samples"].notna().all():
        return df

    df = df.copy()
    df["negative_samples"] = None
    m = df["seq"].size
    for i in range(m):
        negative_sample = [int(df["next"][i])]
        while len(negative_sample) < 101:
            sample = random.randint(0, item_num - 1)
            if sample not in negative_sample:
                negative_sample.append(sample)
        df.at[i, "negative_samples"] = negative_sample
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["movie", "video", "beauty", "sports_and_outdoors", "toys_and_games"],
        default="movie",
    )
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1997)
    parser.add_argument("--metric_ks", nargs="+", type=int, default=[5, 10, 20])
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_path = f"../data/{args.dataset}/{args.split}.df"
    data_df = pd.read_pickle(data_path)
    item_num = max(max(seq) for seq in data_df["seq"])
    item_num = max(item_num, int(data_df["next"].max()))
    data_df = build_negative_samples(data_df, item_num)

    if len(data_df) < args.num_samples:
        raise ValueError(f"Dataset has only {len(data_df)} rows, smaller than num_samples={args.num_samples}")

    mini_df = data_df.sample(n=args.num_samples, random_state=args.seed).reset_index(drop=True)

    # This emulates scores_rec_sb with same expected shape: [B, item_vocab_size].
    fake_scores_rec_sb = torch.randn(args.num_samples, item_num + 1, dtype=torch.float32)

    candidate_ids = torch.tensor(mini_df["negative_samples"].tolist(), dtype=torch.long)
    row_indices = torch.arange(args.num_samples).unsqueeze(1)
    candidate_scores = fake_scores_rec_sb[row_indices, candidate_ids]

    # Positive item is always at index 0 in each candidate list.
    candidate_labels = torch.zeros((args.num_samples, 1), dtype=torch.long)

    metrics = hrs_and_ndcgs_k(candidate_scores, candidate_labels, args.metric_ks)

    print("Random scores metric test")
    print(f"dataset={args.dataset}, split={args.split}, num_samples={args.num_samples}, seed={args.seed}")
    print("metrics:")
    for k in args.metric_ks:
        print(f"  HR@{k}: {metrics['HR@%d' % k]:.6f}")
        print(f"  NDCG@{k}: {metrics['NDCG@%d' % k]:.6f}")


if __name__ == "__main__":
    main()
