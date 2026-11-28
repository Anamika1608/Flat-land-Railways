# ============================================================
# Large-scale test: 100 trains on the large map (50 x 50, 8 cities)
# ============================================================
#
# Every approach runs the same test scenarios (seeds from 0 that build the
# large layout) under several disruption conditions:
#
#   one_20      1 breakdown of 20 steps (the training condition)
#   one_60      1 breakdown of 60 steps (longer than in training)
#   five_20     5 breakdowns of 20 steps at random times
#   real        breakdowns at the real rate per train with lengths drawn
#               from the Finnish delay sample (3.77 minutes per step)
#
# Example:
#   python run_large.py --seeds 10 --conditions one_20 five_20 \
#       --baselines fcfs priority --models models/mappo_large_seed1.pt
#
# Writes outputs/large.csv (same columns as robustness.csv).
# ============================================================

import argparse
import csv
import json
import os
from multiprocessing import Pool

from railsim.disruptions import DisruptionSpec
from railsim.env_factory import scenario_config, usable_seeds
from railsim.runner import run_episode
from run_realdata import load_steps
from run_robustness import BASELINES, policy_from_key


def conditions():
    steps = load_steps("results/realdata/minutes_failure_episodes_ge3.csv", 3.77)
    rate = json.load(open("results/realdata/sample_summary.json"))["failure_episodes_ge3_per_train_run"]
    return {
        "one_20": DisruptionSpec(count=1, duration=20),
        "one_60": DisruptionSpec(count=1, duration=60),
        "five_20": DisruptionSpec(count=5, duration=20),
        "real": DisruptionSpec(durations=steps, rate_per_train=rate),
    }


def run_one(job):
    condition, trains, scale, policy_key, seed, reference = job
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)    # MAPPO samples its actions: same seed, same run
    spec = conditions()[condition]
    result = run_episode(scenario_config(trains, seed, scale), policy_from_key(policy_key), spec,
                         reference=reference)
    row = {"shift": condition, "level": trains}
    row.update(result.as_row())
    return row


def main():
    parser = argparse.ArgumentParser(description="Approaches on 100 trains on the large map.")
    parser.add_argument("--trains", type=int, default=100)
    parser.add_argument("--scale", default="large")
    parser.add_argument("--seeds", type=int, default=10, help="test scenarios")
    parser.add_argument("--conditions", nargs="+", default=list(conditions()), choices=list(conditions()))
    parser.add_argument("--baselines", nargs="*", help="baseline names (default: fcfs priority)")
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--no-reference", action="store_true", help="skip the run without breakdowns")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="outputs/large.csv")
    args = parser.parse_args()

    names = ["fcfs", "priority"] if args.baselines is None else args.baselines
    known = {p.name for p in BASELINES}
    for name in names:
        if name not in known:
            parser.error(f"unknown baseline: {name}")
    policies = names + ["model:" + p for p in args.models]
    seeds = usable_seeds(scenario_config(args.trains, 0, args.scale), args.seeds, start=0)
    jobs = [(c, args.trains, args.scale, p, s, not args.no_reference)
            for c in args.conditions for p in policies for s in seeds]
    # Slow policies first, so the pool is not left waiting on them at the end.
    jobs.sort(key=lambda j: 0 if j[3].startswith("model:") or j[3] in ("priority", "greedy_select") else 1)
    print(f"{len(jobs)} runs: conditions {args.conditions}, {len(policies)} policies, {len(seeds)} scenarios "
          f"(seeds {seeds}), {args.trains} trains, map {args.scale}", flush=True)
    rows = []
    with Pool(args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(run_one, jobs), 1):
            rows.append(row)
            print(f"  {i}/{len(jobs)} {row['shift']} {row['policy']} seed {row['seed']}: delay {row['total_delay']}, "
                  f"arrived {row['arrived']}", flush=True)
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w", newline="") as f:     # rewritten after every run (partial results kept)
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(sorted(rows, key=lambda r: (r["shift"], r["policy"], r["seed"])))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
