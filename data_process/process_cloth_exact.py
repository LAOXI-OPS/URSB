"""重新处理 cloth 全量数据，采样精确 127,503 条。"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

JSON_PATH = Path("E:/Downloads/cloth/reviews_Clothing_Shoes_and_Jewelry.json/Clothing_Shoes_and_Jewelry.json")
OUTPUT_DIR = Path("E:/code/reproduce/my_proj/SBCDR_new/data/cloth")
TEMP_DIR = Path("E:/code/reproduce/my_proj/SBCDR_new/data/cloth_full")

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
        if (i + 1) % 2000000 == 0:
            print(f"  {(i+1)//1000000}M 行...")

print(f"  评分>=4: {sum(len(v) for v in user_items.values()):,} 条, 用户 {len(user_items):,}, 物品 {len(item_set):,}")

print("[2/4] 过滤交互数 >= 5...")
user_items = {uid: items for uid, items in user_items.items() if len(items) >= 5}
print(f"  保留 {len(user_items):,} 个用户")

print("[3/4] 构建全量序列（暂存）...")
all_items = sorted(item_set)
item2id = {aid: i + 1 for i, aid in enumerate(all_items)}
del item_set

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
print(f"  构建了 {len(sequences):,} 条序列")

# 采样精确 127,503 条
print(f"[4/4] 采样 127,503 条，划分 8:1:1...")
np.random.seed(42)
indices = np.random.permutation(len(sequences))
sampled_indices = indices[:127503]
sampled = [sequences[i] for i in sampled_indices]
del sequences

# 重映射物品ID
all_ids = set()
for seq, _, _ in sampled:
    for x in seq:
        if x != 0:
            all_ids.add(x)
for _, _, x in sampled:
    all_ids.add(x)

old_to_new = {old: new for new, old in enumerate(sorted(all_ids), start=1)}

rows = []
for seq, n, target in sampled:
    new_seq = [old_to_new.get(x, 0) for x in seq]
    rows.append((new_seq, n, old_to_new[target]))
del sampled

np.random.shuffle(rows)
n_total = len(rows)
train_end = int(n_total * 0.8)
val_end = int(n_total * 0.9)

train_df = pd.DataFrame(rows[:train_end], columns=['seq', 'len_seq', 'next'])
val_df = pd.DataFrame(rows[train_end:val_end], columns=['seq', 'len_seq', 'next'])
test_df = pd.DataFrame(rows[val_end:], columns=['seq', 'len_seq', 'next'])

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
train_df.to_pickle(OUTPUT_DIR / "train.df")
val_df.to_pickle(OUTPUT_DIR / "val.df")
test_df.to_pickle(OUTPUT_DIR / "test.df")

max_id = max(max(max(s) for s in train_df['seq']), train_df['next'].max(),
             max(max(s) for s in val_df['seq']), val_df['next'].max(),
             max(max(s) for s in test_df['seq']), test_df['next'].max())

print(f"  训练: {len(train_df):,}, 验证: {len(val_df):,}, 测试: {len(test_df):,}")
print(f"  物品: {len(all_ids):,}, 最大ID: {max_id:,}, 嵌入: {max_id * 256 * 4 / 1024**3:.2f} GB")
print(f"\n完成!")
