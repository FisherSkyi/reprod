#!/usr/bin/env python
"""Non-interactive evaluation / aggregation for batch (Slurm) runs.

Discovers every trained run under ``experiments/<prefix>*`` , scores each on the
TEST split (AURSAC/AURDAC for ID and, where applicable, OOD experts) by reusing
``evaluate.gogogo()``, aggregates mean +/- std across seeds per (archetype,
model), and prints the result next to the paper's Table 1 (Blood Cells).

Run from the repo root, inside the SAME env that set BLOOD_N_CLASSES, e.g.:
    BLOOD_N_CLASSES=8  python src_try/eval_auto.py --prefix blood_n8
    BLOOD_N_CLASSES=10 python src_try/eval_auto.py --prefix blood_n10
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

from evaluate import gogogo  # per-run scorer (ID + OOD), reused as-is

# Paper Table 1, Blood Cells (SAC, DAC). Stable=realistic_specialist, Variable=variable_specialist.
PAPER_BLOOD = {
    ('realistic_specialist', 'ID'):  {'ifd': (.89, .84), 'l2d-pop-qc': (.88, .83), 'l2d-pop-qi': (.89, .83), 'l2d-multi': (.87, .81)},
    ('realistic_specialist', 'OOD'): {'ifd': (.89, .85), 'l2d-pop-qc': (.87, .81), 'l2d-pop-qi': (.88, .80)},
    ('variable_specialist',  'ID'):  {'ifd': (.81, .73), 'l2d-pop-qc': (.78, .62), 'l2d-pop-qi': (.77, .59), 'l2d-multi': (.75, .56)},
    ('variable_specialist',  'OOD'): {'ifd': (.80, .69), 'l2d-pop-qc': (.74, .54), 'l2d-pop-qi': (.74, .52)},
}
ARCH_LABEL = {'realistic_specialist': 'Stable', 'variable_specialist': 'Variable'}
MODEL_ORDER = ['ifd', 'l2d-pop-qc', 'l2d-pop-qi', 'l2d-multi']


def model_label(cfg):
    """Map a run's config to a paper-aligned variant label."""
    m = cfg['model']
    if m == 'l2d-pop':
        return 'l2d-pop-qc' if cfg.get('with_attn', True) else 'l2d-pop-qi'
    return m


def discover_runs(experiments_dir, prefix):
    """Latest-timestamp run dir per (exp, model, dataset, archetype, seed)."""
    latest = {}
    for root, _, files in os.walk(experiments_dir):
        if 'best_val_acc_sd.pth' not in files or 'config.json' not in files:
            continue
        rel = os.path.relpath(root, experiments_dir).split(os.sep)
        # layout: <exp>/<model>/<dataset>/<archetype>/<seed>/<timestamp>
        if len(rel) < 6 or not rel[0].startswith(prefix):
            continue
        key, ts = tuple(rel[:5]), rel[5]
        if key not in latest or ts > latest[key][0]:
            latest[key] = (ts, root)
    return [path for _, path in latest.values()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', required=True, help="experiment-name prefix, e.g. blood_n8")
    ap.add_argument('--dataset', default='blood_mnist')
    ap.add_argument('--experiments_dir', default='experiments')
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--end', type=float, default=1.0)
    args = ap.parse_args()

    runs = discover_runs(args.experiments_dir, args.prefix)
    print(f"\nFound {len(runs)} run(s) under {args.experiments_dir}/{args.prefix}*  "
          f"| AURSAC/AURDAC over budget [{args.start}, {args.end}] "
          f"| BLOOD_N_CLASSES={os.environ.get('BLOOD_N_CLASSES', '(unset->8)')}")
    if not runs:
        print("  Nothing to evaluate.")
        return

    # group[(archetype, label)][dist] -> list of (sac, dac)
    group = defaultdict(lambda: {'ID': [], 'OOD': []})
    for path in sorted(runs):
        with open(os.path.join(path, 'config.json')) as f:
            cfg = json.load(f)
        if cfg.get('dataset') != args.dataset:
            continue
        label = model_label(cfg)
        arch = cfg['expert_archetypes']
        arch = arch[0] if isinstance(arch, list) else arch
        try:
            res = gogogo(path, args.start, args.end)  # (sac_id, dac_id[, sac_ood, dac_ood])
        except Exception as e:
            print(f"  SKIP {path}: {type(e).__name__}: {e}")
            continue
        group[(arch, label)]['ID'].append((res[0], res[1]))
        if len(res) == 4:
            group[(arch, label)]['OOD'].append((res[2], res[3]))

    # report
    for arch in ('realistic_specialist', 'variable_specialist'):
        print(f"\n=== {ARCH_LABEL.get(arch, arch)} experts ({arch}) — prefix {args.prefix} ===")
        print(f"{'model':12} {'dist':4} {'n':>2}  {'SAC (meas)':>12} {'DAC (meas)':>12}  "
              f"{'paper SAC/DAC':>13} {'dSAC':>6} {'dDAC':>6}")
        for label in MODEL_ORDER:
            for dist in ('ID', 'OOD'):
                vals = group.get((arch, label), {}).get(dist, [])
                if not vals:
                    continue
                a = np.array(vals)
                sac_m, dac_m = a[:, 0].mean(), a[:, 1].mean()
                sac_s = f"{sac_m:.2f}±{a[:, 0].std():.2f}"
                dac_s = f"{dac_m:.2f}±{a[:, 1].std():.2f}"
                ref = PAPER_BLOOD.get((arch, dist), {}).get(label)
                if ref:
                    ref_s = f"{ref[0]:.2f}/{ref[1]:.2f}"
                    dsac, ddac = f"{sac_m - ref[0]:+.2f}", f"{dac_m - ref[1]:+.2f}"
                else:
                    ref_s, dsac, ddac = "-", "-", "-"
                print(f"{label:12} {dist:4} {len(vals):>2}  {sac_s:>12} {dac_s:>12}  "
                      f"{ref_s:>13} {dsac:>6} {ddac:>6}")


if __name__ == '__main__':
    main()
