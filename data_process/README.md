# 数据处理工作流

## 处理规则

1. **交互处理**：评分 ≥ 4 视为正反馈（隐式交互）
2. **用户过滤**：保留交互数 ≥ N 的用户（N 由阈值表决定）
3. **序列构建**：每个用户序列固定为 9 个物品（前 8 历史，最后 1 个目标），左补 0
4. **时间排序**：按时间戳排序取最后 9 条
5. **数据划分**：训练:验证:测试 = 8:1:1
6. **物品 ID**：从 1 开始连续编号，0 保留给 padding

## 输出格式

每个数据集生成 `data/{name}/` 目录，包含 3 个 pickle 文件：
- `train.df` — 80% 用户
- `val.df` — 10% 用户
- `test.df` — 10% 用户

每个 DataFrame 含 3 列：`seq`(list[8]), `len_seq`(int), `next`(int)

## 注册数据集

在 `src/main.py` 的 `DATASET_ALIASES` 中加入新名称，即可用 `--s_dataset` / `--t_dataset` 引用。

## 处理流程

### 1. 理解原始数据

先看格式和阈值情况。原始数据分两类：

**CSV 格式**（如 `Books_rating.csv`）：
- `review/score` — 评分
- `User_id` — 用户 ID
- `Title` / `asin` — 物品 ID（可能是书名而非 asin）
- `review/time` — Unix 时间戳

**JSON/JSONL 格式**（每行一个 JSON）：
- `overall` 或 `rating` — 评分
- `reviewerID` 或 `user_id` — 用户 ID
- `asin` — 物品 ID
- `unixReviewTime` 或 `timestamp`（毫秒）— 时间戳

### 2. 统计阈值用户数

```python
import json, csv
from collections import defaultdict

path = '原始文件路径'
user_items = defaultdict(int)

# JSONL 格式
with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        obj = json.loads(line)
        if obj['rating'] >= 4.0:  # CSV 用 'review/score'
            user_items[obj['user_id']] += 1  # JSON 用 'reviewerID'

for th in range(5, 16):
    cnt = sum(1 for v in user_items.values() if v >= th)
    print(f'交互>={th}: {cnt:,}')
```

### 3. 选择合适的阈值和用户数

- 参考现有数据集阈值和规模来决定
- 目标用户数应 ≤ 阈值对应的用户数
- 采样方式：排序后取连续段比随机采样更可控

### 4. 运行处理

参考 `data_process/` 下的现有脚本，修改 `JSONL_PATH`、`OUTPUT_DIR`、`TARGET_USERS`、`MIN_INTERACTIONS` 参数后运行：

```bash
python data_process/process_xxx.py
```

### 5. 加入超参设置

在 `src/main.py` 的 `DATASET_ALIASES` 中添加别名映射。

## 工具脚本

所有处理脚本在 `data_process/` 目录下：
- `process_cloth_exact.py` — Cloth，可调 TARGET_USERS
- `process_elec.py` — Electronics
- `process_toy.py` — Toys_and_Games（已改名为 process_toys.py），可调阈值和用户数
- `process_phone.py` — Phones
- `process_music.py` — CDs_and_Vinyl
- `process_food.py` — Grocery_and_Gourmet_Food
- `process_video.py` — Video_Games（如有）

每个脚本顶部注释说明用途和参数，直接修改脚本中的 `TARGET_USERS` 和 `MIN_INTERACTIONS` 即可复用。

## 注意事项

- JSON 的 `unixReviewTime` 是秒级，JSONL 的 `timestamp` 是毫秒级
- CSV 的 `review/time` 可能有空值或字符串，需用 `pd.to_numeric()` + `try/except` 处理
- 物品 ID 必须连续，避免稀疏 ID 导致嵌入矩阵过大（`max_id × hidden_size × 4` 字节）
- 处理大数据集（14GB+）时注意内存，14GB JSON 约需 30 分钟处理
- 嵌入矩阵估算：`max_item_id × 256 × 4 / (1024^3)` GB
