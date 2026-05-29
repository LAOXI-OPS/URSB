# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SBCDR (Score-Based Cross-Domain Recommendation) — 基于扩散模型的跨域推荐。先在源域训练，再在目标域微调。核心思想：用 Schrödinger Bridge 扩散模型学习域不变的用户序列表示。

## Run Commands

**单次训练（源域→目标域）：**
```bash
python src/main.py --s_dataset movie --t_dataset video --device cuda
```

**快速调试（小步数）：**
```bash
python src/main.py --s_dataset movie --t_dataset video --device cuda --sample_steps 2 --diffusion_steps 3 --epochs 5 --eval_interval 1
```

**批量跑全部域组合：**
```bash
python data_process/batch_test.py
```

**数据预处理：**
```bash
python data_process/process_elec.py
```

**随机分数指标测试：**
```bash
python src/random_scores_metric_test.py --dataset movie --split test --num_samples 5
```

**常用参数：**
- `--s_dataset` / `--t_dataset`：源/目标域（movie, video, beauty, books, cloth, elec, food, phone, music, sports, toys）
- `--hidden_size 128`（默认）或 `256`
- `--batch_size 512`（默认，根据显存调整）
- `--eval_mode negative`（带负采样评估，更快）
- `--num_negatives 100`

## Key Architecture

训练流程（`src/main.py`）：

1. **Phase 1：源域训练** → `model_train(forward_flag=True)`，保存 checkpoint 到 `saved_model/{s_dataset}_{t_dataset}/model.pth`
2. **加载 checkpoint** → 从磁盘 reload 最佳权重
3. **Phase 2：目标域微调** → `model_train(forward_flag=False)`，短 epoch 在目标域继续训练
4. **回源评估** → 在源域测试集上评估微调后的模型

核心组件：

| 文件 | 职责 |
|---|---|
| `main.py` | CLI 入口，两阶段训练流程编排 |
| `model.py` | `AttSBModel` — 嵌入层 + 域共享投影层 + 位置编码，包装 SBRec |
| `sbrec.py` | `SBRec` — Schrödinger Bridge 扩散过程（前向加噪 + 反向 ODE 采样）；`SBXstart` — 带时间嵌入的条件去噪网络；Transformer 编码器 |
| `step_sample.py` | 扩散步采样器（训练时决定每个 batch 从哪个 timestep 采样） |
| `trainer.py` | 训练循环、验证/测试评估、早停、checkpoint 保存、HR/NDCG 指标计算 |

跨域机制：源/目标域各有独立 Embedding，共享 `Linear` 投影层。`forward_flag` 控制当前用哪个域的嵌入和投影。

## Key Data Conventions

- 序列固定长度 `max_len=8`（9 个物品：8 历史 + 1 目标），不足左补 0
- 物品 ID 从 1 开始，0 保留给 padding
- 数据格式：pickle DataFrame，列 `seq`(list[int8]), `len_seq`(int), `next`(int)
- 数据路径：`data/{dataset_name}/train.df` / `val.df` / `test.df`

## Current Design Notes

- 默认 `--schedule_sampler_name=lossaware` 实际等价于均匀采样（`UniformSampler`），loss tracking 未接入训练循环
- 反向采样 `reverse_p_sample` 接收 `item_rep1` 和 `noise_x_t` 参数保持接口完整性，当前内部通过乘以 `std_fwd[0]`（恒为 0）使其不影响结果
- `eval_mode=full` 对所有物品排序做评估，`negative` 模式只对 100 个负样本+1 个正样本排序（更快但指标值偏高）
