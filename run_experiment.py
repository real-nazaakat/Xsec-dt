import os
import sys
import argparse
import yaml
import json
import random
import subprocess
from pathlib import Path
from datetime import datetime

def set_global_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def load_config(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as fh:
        return yaml.safe_load(fh)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def call_trainer(trainer_script: Path, features: Path, config: Path, out_dir: Path, seed: int, save_stdout: bool):
    """Call trainer script as subprocess and capture stdout/stderr."""
    cmd = [sys.executable, str(trainer_script), '--features', str(features), '--config', str(config)]
    env = os.environ.copy()
    env['EXPERIMENT_SEED'] = str(seed)
    # create output dir for this run
    ensure_dir(out_dir)
    stdout_path = out_dir / 'stdout.txt'
    stderr_path = out_dir / 'stderr.txt'
    meta = {'cmd': cmd, 'env_seed': seed, 'started_at': datetime.utcnow().isoformat() + 'Z'}
    # run
    with open(stdout_path, 'wb') as out_f, open(stderr_path, 'wb') as err_f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        out, err = proc.communicate()
        out_f.write(out or b'')
        err_f.write(err or b'')
    meta['returncode'] = proc.returncode
    meta['finished_at'] = datetime.utcnow().isoformat() + 'Z'
    (out_dir / 'run_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return proc.returncode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True, help='Path to experiment YAML config')
    parser.add_argument('--repeat', '-r', type=int, default=None, help='Number of repeats (overrides config.training.n_repeats)')
    parser.add_argument('--seed-file', '-s', default=None, help='Optional file with seeds (one per line). If provided, repeat is ignored.')
    parser.add_argument('--dry-run', action='store_true', help='Print planned commands but do not execute trainers')
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print('Config file not found:', cfg_path, file=sys.stderr)
        sys.exit(2)
    cfg = load_config(cfg_path)
    exp = cfg.get('experiment', {})
    training = cfg.get('training', {})
    logging_cfg = cfg.get('logging', {})

    trainer_script = Path(exp.get('trainer_script', ''))
    features_csv = Path(exp.get('features_csv', ''))
    model_config = Path(exp.get('model_config', 'models/model_config.yaml'))
    base_output_dir = Path(exp.get('output_dir', 'experiments/results'))  # per-config override

    save_stdout = bool(logging_cfg.get('save_stdout', True))
    save_meta = bool(logging_cfg.get('save_run_metadata', True))

    # Determine seeds
    if args.seed_file:
        seed_file = Path(args.seed_file)
        seeds = [int(line.strip()) for line in seed_file.read_text().splitlines() if line.strip() and not line.strip().startswith('#')]
    elif args.repeat is not None:
        base_seed = int(training.get('random_seed', 42))
        seeds = [base_seed + i for i in range(args.repeat)]
    else:
        repeats = int(training.get('n_repeats', 1))
        base_seed = int(training.get('random_seed', 42))
        seeds = [base_seed + i for i in range(repeats)]

    print(f'Experiment: {exp.get("name", "unnamed")} — trainer: {trainer_script} — repeats: {len(seeds)}')
    print('Planned seeds:', seeds)
    if args.dry_run:
        print('Dry-run mode: will not execute trainer scripts.')
    # Loop over seeds and run
    for seed in seeds:
        print(f'-- Running seed {seed} --')
        set_global_seed(seed)
        run_dir = Path(base_output_dir) / exp.get('name', 'exp') / str(seed)
        ensure_dir(run_dir)
        # save run config + seed
        if save_meta:
            meta = {
                'config_used': str(cfg_path),
                'experiment': exp,
                'training': training,
                'seed': seed,
                'timestamp_utc': datetime.utcnow().isoformat() + 'Z'
            }
            (run_dir / 'config_snapshot.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        if args.dry_run:
            print('DRY CMD:', sys.executable, trainer_script, '--features', features_csv, '--config', model_config)
            continue
        # check trainer script & features exist
        if not trainer_script.exists():
            print('Trainer script not found:', trainer_script, file=sys.stderr)
            continue
        if not features_csv.exists():
            print('Features CSV not found (expected):', features_csv, file=sys.stderr)
            # still attempt to run trainer; trainer should error if features missing
        rc = call_trainer(trainer_script, features_csv, model_config, run_dir, seed, save_stdout)
        print('Return code:', rc)
    print('All runs finished.')

if __name__ == "__main__":
    main()
