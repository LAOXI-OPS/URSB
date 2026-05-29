import subprocess, sys
from pathlib import Path

datasets = ['books', 'cloth', 'elec', 'food', 'movie_new', 'music', 'phone']
python = r'E:\code\reproduce\reproduce\Scripts\python.exe'
base_args = '--sample_steps 2 --diffusion_steps 3 --lr 0.001 --epochs 5 --eval_interval 1 --batch_size 256 --random_seed 143 --lambda_uncertainty 0.001 --p 0.3 --hidden_size 256 --eval_mode negative --patience 4 --num_negatives 100 --ft_epochs 5'

src_dir = Path(__file__).resolve().parent.parent / 'src'
results = {}

for src in datasets:
    for tgt in datasets:
        if src == tgt:
            continue
        key = f'{src} -> {tgt}'
        print(f'Running: {key}', flush=True)
        cmd = f'{python} main.py --s_dataset {src} --t_dataset {tgt} --device cuda {base_args}'
        try:
            out = subprocess.check_output(cmd, shell=True, cwd=str(src_dir),
                                          stderr=subprocess.STDOUT, timeout=900,
                                          text=True, encoding='gbk', errors='replace')
            # 提取所有 HR@10 行，最后一个是 Phase 2 test
            hr10s = []
            for line in out.split('\n'):
                line = line.strip()
                if 'HR@10:' in line and 'Epoch' not in line:
                    try:
                        val = float(line.split('HR@10:')[1].split(',')[0])
                        hr10s.append(val)
                    except:
                        pass
            p1_hr10 = hr10s[0] if len(hr10s) >= 1 else None
            p2_hr10 = hr10s[-1] if len(hr10s) >= 2 else None
            results[key] = {'p1': p1_hr10, 'p2': p2_hr10}
            print(f'  P1={p1_hr10}, P2={p2_hr10}', flush=True)
        except Exception as e:
            print(f'  ERROR: {e}', flush=True)
            results[key] = {'p1': None, 'p2': None}

print('\n========== Phase 2 HR@10 Matrix ==========')
print(f'{"":>12}', end='')
for ds in datasets:
    print(f'{ds:>10}', end='')
print()
for src in datasets:
    print(f'{src:>12}', end='')
    for tgt in datasets:
        if src == tgt:
            print(f'{"-":>10}', end='')
        else:
            r = results.get(f'{src} -> {tgt}', {})
            v = r.get('p2', None)
            print(f'{v:>10.2f}' if v else f'{"N/A":>10}', end='')
    print()

print('\n========== Top 10 Phase 2 pairs ==========')
sorted_results = sorted([(k, v['p2']) for k, v in results.items() if v.get('p2')],
                        key=lambda x: x[1], reverse=True)
for i, (k, v) in enumerate(sorted_results[:10]):
    print(f'  {i+1}. {k}: {v:.2f}')

# 最佳2对均值
print(f'\n最佳2对均值: {(sorted_results[0][1] + sorted_results[1][1]) / 2:.2f}')
