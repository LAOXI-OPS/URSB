"""处理 Amazon CDs_and_Vinyl (Music) 数据集为 SBCDR 项目格式。

用法: python process_music.py
输出: data/music/train.df, val.df, test.df
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

JSONL_PATH = Path("F:/迅雷下载/amazon数据集/CDs_and_Vinyl.jsonl")
OUTPUT_DIR = Path("E:/code/reproduce/my_proj/SBCDR_new/data/music")
TARGET_USERS = 33985
RANDOM_SEED = 42

print(f"[1/4] 读取 JSONL，过滤评分 >= 4...")
user_items = defaultdict(list)
item_set = set()

with open(JSONL_PATH, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj["rating"] >= 4.0:
            user_items[obj["user_id"]].append((obj["asin"], obj["timestamp"]))
            item_set.add(obj["asin"])
        if (i + 1) % 1000000 == 0:
            print(f"  已读取 {(i+1)//1000000}M 行...")

print(f"  评分>=4: {sum(len(v) for v in user_items.values()):,} 条, 用户 {len(user_items):,}, 物品 {len(item_set):,}")

print(f"[2/4] 过滤交互数 >= 5 的用户...")
user_items = {uid: items for uid, items in user_items.items() if len(items) >= 5}
print(f"  保留 {len(user_items):,} 个用户")

print(f"[3/4] 采样 {TARGET_USERS} 个用户，映射物品到连续 ID...")
np.random.seed(RANDOM_SEED)
all_uids = list(user_items.keys())
selected = set(np.random.choice(all_uids, TARGET_USERS, replace=False))
user_items = {uid: user_items[uid] for uid in selected}

filtered_items = set()
for items in user_items.values():
    for aid, _ in items:
        filtered_items.add(aid)
all_items = sorted(filtered_items)
item2id = {aid: i + 1 for i, aid in enumerate(all_items)}
num_items = len(all_items)
print(f"  物品数: {num_items:,}")

print(f"[4/4] 构建序列并划分 8:1:1...")
sequences = []
for uid, items in user_items.items():
    items.sort(key=lambda x: x[1])
    recent = items[-9:]
    item_ids = [item2id[aid] for aid, _ in recent]
    hist = item_ids[:-1]
    target = item_ids[-1]
    n = len(hist)
    seq = [0] * (8 - n) + hist
    sequences.append((seq, n, target))

del user_items, item2id

np.random.seed(RANDOM_SEED)
np.random.shuffle(sequences)
n_total = len(sequences)
train_end = int(n_total * 0.8)
val_end = int(n_total * 0.9)

train_df = pd.DataFrame(sequences[:train_end], columns=['seq', 'len_seq', 'next'])
val_df = pd.DataFrame(sequences[train_end:val_end], columns=['seq', 'len_seq', 'next'])
test_df = pd.DataFrame(sequences[val_end:], columns=['seq', 'len_seq', 'next'])

max_id = max(max(max(s) for s in train_df['seq']), train_df['next'].max(),
             max(max(s) for s in val_df['seq']), val_df['next'].max(),
             max(max(s) for s in test_df['seq']), test_df['next'].max())

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
train_df.to_pickle(OUTPUT_DIR / "train.df")
val_df.to_pickle(OUTPUT_DIR / "val.df")
test_df.to_pickle(OUTPUT_DIR / "test.df")

print(f"  训练: {len(train_df):,}, 验证: {len(val_df):,}, 测试: {len(test_df):,}")
print(f"  物品: {num_items:,}, 最大ID: {max_id:,}")
print(f"  嵌入矩阵: {max_id * 256 * 4 / 1024**3:.2f} GB")
print(f"\n保存到 {OUTPUT_DIR}")
print("完成!")
