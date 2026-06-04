#!/bin/bash
# 6组双向跨域实验：books-elec, books-douban, douban-elec (双向)
PYTHON="E:/code/reproduce/reproduce/Scripts/python.exe"
COMMON="--sample_steps 2 --diffusion_steps 2 --lr 0.005 --epochs 5 --eval_interval 1 --batch_size 256 --random_seed 143 --hidden_size 256 --eval_mode negative --patience 4 --num_negatives 100 --ft_epochs 5 --device cuda"
LOG_DIR="log/batch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Batch experiment: $(date)"
echo "Log dir: $LOG_DIR"
echo "=========================================="

pairs=(
  "books elec"
  "elec books"
  "books douban"
  "douban books"
  "douban elec"
  "elec douban"
)

for pair in "${pairs[@]}"; do
  src=$(echo $pair | cut -d' ' -f1)
  tgt=$(echo $pair | cut -d' ' -f2)
  echo ""
  echo "===== $src -> $tgt ====="
  $PYTHON src/main.py \
    --s_dataset "$src" --t_dataset "$tgt" \
    $COMMON \
    2>&1 | tee "$LOG_DIR/${src}_${tgt}.log"
  echo "===== $src -> $tgt DONE ====="
done

echo ""
echo "=========================================="
echo "All experiments done. Extracting results..."
echo "=========================================="

for pair in "${pairs[@]}"; do
  src=$(echo $pair | cut -d' ' -f1)
  tgt=$(echo $pair | cut -d' ' -f2)
  echo ""
  echo "=== $src -> $tgt ==="
  grep -A1 "Target Val\|Target Test" "$LOG_DIR/${src}_${tgt}.log" | tail -20
done
