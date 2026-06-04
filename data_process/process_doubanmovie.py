"""处理豆瓣电影评论数据集为 SBCDR 项目格式。
筛选策略：电影按交互次数 top 20% 为热门，用户按热门电影占比排序取前 55,662。
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

CSV_PATH = Path("E:/Downloads/all_movies_with_id.csv")
OUTPUT_DIR = Path("E:/code/reproduce/my_proj/SBCDR_new/data/doubanmovie")
MAX_LEN = 8
TARGET_USERS = 55662

print("[1/6] 分块读取 CSV，统计电影交互次数 + 按用户聚合...")
user_items = defaultdict(list)
movie_counts = Counter()

chunksize = 200000
for i, chunk in enumerate(pd.read_csv(CSV_PATH, usecols=['Movie_Name', 'Username', 'Date'],
                                      encoding='utf-8-sig', chunksize=chunksize)):
    chunk = chunk.dropna(subset=['Username', 'Movie_Name'])
    for _, row in chunk.iterrows():
        movie = str(row['Movie_Name'])
        user_items[row['Username']].append((movie, str(row['Date'])))
        movie_counts[movie] += 1
    if (i + 1) % 10 == 0:
        print(f"  已处理 {(i+1)*chunksize:,} 行, 用户 {len(user_items):,}, 电影 {len(movie_counts):,}")

total_ints = sum(len(v) for v in user_items.values())
print(f"  总交互: {total_ints:,}, 用户: {len(user_items):,}, 电影: {len(movie_counts):,}")

print("[2/6] 过滤交互数 >= 5 的用户...")
user_items = {uid: items for uid, items in user_items.items() if len(items) >= 5}
print(f"  保留 {len(user_items):,} 个用户")

print("[3/6] 按交互次数确定 top 40% 热门电影...")
sorted_movies = sorted(movie_counts.items(), key=lambda x: x[1], reverse=True)
top_n = max(1, int(len(sorted_movies) * 0.4))
popular_movies = {m for m, _ in sorted_movies[:top_n]}
print(f"  电影总数: {len(sorted_movies):,}, top 20% 阈值: {top_n:,}, 热门电影数: {len(popular_movies):,}")
print(f"  热门阈值交互数: >= {sorted_movies[top_n-1][1]}")

print("[4/6] 计算每个用户的热门电影占比并排序...")
user_pop_ratio = {}
for uid, items in user_items.items():
    user_movies = {m for m, _ in items}
    if not user_movies:
        user_pop_ratio[uid] = 0.0
        continue
    pop_count = sum(1 for m in user_movies if m in popular_movies)
    user_pop_ratio[uid] = pop_count / len(user_movies)

sorted_users = sorted(user_pop_ratio.items(), key=lambda x: x[1], reverse=True)
print(f"  占比范围: {sorted_users[-1][1]:.4f} ~ {sorted_users[0][1]:.4f}")
print(f"  top 55,662 阈值: {sorted_users[TARGET_USERS-1][1]:.4f}")

selected_uids = {uid for uid, _ in sorted_users[:TARGET_USERS]}
user_items = {uid: user_items[uid] for uid in selected_uids}

seen_items = set()
for items in user_items.values():
    for aid, _ in items:
        seen_items.add(aid)
print(f"  最终用户: {len(user_items):,}, 最终电影: {len(seen_items):,}")

print("[5/6] 映射物品到连续整数 ID...")
all_items = sorted(seen_items)
item2id = {aid: i + 1 for i, aid in enumerate(all_items)}
num_items = len(all_items)
print(f"  物品数: {num_items:,}")

print("[6/6] 构建序列并划分 8:1:1...")
sequences = []
for uid, items in user_items.items():
    items.sort(key=lambda x: x[1])
    recent = items[-9:]
    item_ids = [item2id[aid] for aid, _ in recent]
    hist = item_ids[:-1]
    target = item_ids[-1]
    n = len(hist)
    seq = [0] * (MAX_LEN - n) + hist
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
print(f"  保存到 {OUTPUT_DIR}")
print("完成!")
