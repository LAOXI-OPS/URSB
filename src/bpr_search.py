"""Search over BPR loss coefficients for target-domain fine-tuning.

Stage 1 (source pre-training) runs once. Stage 2 (target fine-tuning) is
repeated for each --lambda_bpr value.
"""

import subprocess, sys
from pathlib import Path

lambda_bpr_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

python = sys.executable
src_dir = Path(__file__).resolve().parent
base_args = '--s_dataset movie --t_dataset video --device cuda'
base_args += ' --sample_steps 2 --diffusion_steps 3 --epochs 5 --eval_interval 1'
base_args += ' --batch_size 256 --hidden_size 256 --eval_mode negative --ft_epochs 5'
base_args += ' --lr 0.001 --lambda_x1 0.1 --random_seed 143 --patience 4'

results = {}

for lbpr in lambda_bpr_values:
    key = f'lambda_bpr={lbpr}'
    print(f'Running: {key}', flush=True)
    cmd = f'{python} main.py {base_args} --lambda_bpr {lbpr}'
    try:
        out = subprocess.check_output(cmd, shell=True, cwd=str(src_dir),
                                      stderr=subprocess.STDOUT, timeout=600,
                                      text=True, encoding='gbk', errors='replace')
        # Extract the second test results (Post-Finetune source test)
        hr10s = []
        for line in out.split('\n'):
            line = line.strip()
            if 'HR@10:' in line and 'Epoch' not in line and 'Best' not in line:
                try:
                    val = float(line.split('HR@10:')[1].split(',')[0])
                    hr10s.append(val)
                except ValueError:
                    pass
        # Last HR@10 is Phase 2 test (Source Test Post-Finetune)
        hr10 = hr10s[-1] if hr10s else None
        results[key] = hr10
        print(f'  -> HR@10: {hr10}', flush=True)
    except Exception as e:
        print(f'  ERROR: {e}', flush=True)
        results[key] = None

print('\n========== BPR Coefficient Search Results ==========')
print(f'{"lambda_bpr":>15}  {"HR@10":>10}')
print('-' * 28)
for lbpr in lambda_bpr_values:
    key = f'lambda_bpr={lbpr}'
    v = results.get(key, None)
    print(f'{lbpr:>15.3f}  {v:>10.4f}' if v else f'{lbpr:>15.3f}  {"N/A":>10}')
