"""处理 Amazon Electronics 5-core 数据集为 SBCDR 项目格式。"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

JSON_PATH = Path("E:/Downloads/Elec/Electronics_5.json")
OUTPUT_DIR = Path("E:/code/reproduce/my_proj/SBCDR_new/data/elec")

print("[1/4] 读取 JSON，过滤评分 >= 4...")
user_items = defaultdict(list)
item_set = set()
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj["overall"] >= 4.0:
            user_items[obj["reviewerID"]].append((obj["asin"], obj["unixReviewTime"]))
            item_set.add(obj["asin"])

print(f"  评分>=4: {sum(len(v) for v in user_items.values()):,} 条, 用户 {len(user_items):,}, 物品 {len(item_set):,}")

print("[2/4] 过滤交互数 >= 5 的用户...")
user_items = {uid: items for uid, items in user_items.items() if len(items) >= 5}
print(f"  保留 {len(user_items):,} 个用户")

print("[3/4] 映射物品到连续整数 ID...")
all_items = sorted(item_set)
item2id = {aid: i + 1 for i, aid in enumerate(all_items)}
num_items = len(all_items)
print(f"  物品数: {num_items:,}")

print("[4/4] 构建序列并划分 8:1:1...")
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

np.random.seed(42)
np.random.shuffle(sequences)
n_total = len(sequences)
train_end = int(n_total * 0.8)
val_end = int(n_total * 0.9)

train_df = pd.DataFrame(sequences[:train_end], columns=['seq', 'len_seq', 'next'])
val_df = pd.DataFrame(sequences[train_end:val_end], columns=['seq', 'len_seq', 'next'])
test_df = pd.DataFrame(sequences[val_end:], columns=['seq', 'len_seq', 'next'])

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
train_df.to_pickle(OUTPUT_DIR / "train.df")
val_df.to_pickle(OUTPUT_DIR / "val.df")
test_df.to_pickle(OUTPUT_DIR / "test.df")

max_id = max(max(max(s) for s in train_df['seq']), train_df['next'].max(),
             max(max(s) for s in val_df['seq']), val_df['next'].max(),
             max(max(s) for s in test_df['seq']), test_df['next'].max())

print(f"  训练: {len(train_df):,}, 验证: {len(val_df):,}, 测试: {len(test_df):,}")
print(f"  物品: {num_items:,}, 最大ID: {max_id:,}, 嵌入: {max_id * 256 * 4 / 1024**3:.2f} GB")
print(f"\n保存到 {OUTPUT_DIR}")
print("完成!")
