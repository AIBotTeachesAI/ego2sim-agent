"""Run check-and-fix trials.

  one trial, with robot video:   python scripts/run_loop.py --seed 3 --mode baseline --video
  batch evaluation:              python scripts/run_loop.py --seeds 0-9 --modes none,baseline
"""
import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from ego2sim.agent.loop import MODES, get_decider, run_trial  # noqa: E402
from ego2sim.agent.perturb import SCENARIOS  # noqa: E402
from ego2sim.agent.scene import VerifyScene         # noqa: E402


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def make_human(policy):
    if policy == "auto-yes":
        return lambda reason: True
    if policy == "auto-no":
        return lambda reason: False

    def ask(reason):
        return input(f"\n  >> Agent asks the operator: {reason}\n  >> Approve another try? [y/n]: ").strip().lower().startswith("y")
    return ask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", default="take01")
    ap.add_argument("--seed", type=int, help="single seed")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--mode", help="single mode")
    ap.add_argument("--modes", default="none,baseline")
    ap.add_argument("--video", action="store_true", help="render robot motion video (slow, ~1 min per trial)")
    ap.add_argument("--scenario", help="single scenario")
    ap.add_argument("--scenarios", default="clean", help=f"comma list from {SCENARIOS}, or 'all'")
    ap.add_argument("--human", default="ask", choices=["ask", "auto-yes", "auto-no"],
                    help="how ask_human is answered: prompt in the terminal, or automatically (batch runs)")
    a = ap.parse_args()

    seeds = [a.seed] if a.seed is not None else parse_seeds(a.seeds)
    modes = [a.mode] if a.mode else a.modes.split(",")
    for m in modes:
        if m not in MODES:
            sys.exit(f"unknown mode {m}; available: {MODES}")
    scenarios = [a.scenario] if a.scenario else (list(SCENARIOS) if a.scenarios == "all" else a.scenarios.split(","))
    for sc in scenarios:
        if sc not in SCENARIOS:
            sys.exit(f"unknown scenario {sc}; available: {SCENARIOS}")
    human = make_human(a.human)

    with open(f"outputs/{a.take}/sim_metrics.json") as f:
        target = json.load(f)["target_cup_robot_m"][:2]
    base = f"outputs/{a.take}_loop"
    scene = VerifyScene()
    deciders = {m: get_decider(m) for m in modes}
    rows = []
    for scenario in scenarios:
        for mode in modes:
            for seed in seeds:
                out = f"{base}/{scenario}/{mode}/seed_{seed:02d}"
                rows.append(run_trial(target, seed, mode, out, scene, decide=deciders[mode], video=a.video,
                                      human=human, scenario=scenario))
    scene.close()

    if len(rows) > 1:
        with open(f"{base}/trials.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        hdr = " scenario   | method   | handled | success | unsafe moves | avg fixes | human asks | avg final err (cm)"
        print("\n" + hdr + "\n" + "-" * len(hdr))
        table = []
        for scenario in scenarios:
            for mode in modes:
                r = [x for x in rows if x["mode"] == mode and x["scenario"] == scenario]
                n = len(r)
                row = {"scenario": scenario, "method": mode,
                       "handled_correctly": f"{sum(x['handled_correctly'] for x in r)}/{n}",
                       "success": f"{sum(x['success'] for x in r)}/{n}",
                       "unsafe_moves": sum(x["unsafe_moves"] for x in r),
                       "avg_fixes": round(sum(x["corrections"] for x in r) / n, 2),
                       "human_asks": sum(x["human_asks"] for x in r),
                       "avg_final_error_cm": round(sum(x["final_error_cm"] for x in r) / n, 2)}
                table.append(row)
                print(f" {scenario:10s} | {mode:8s} | {row['handled_correctly']:>7} | {row['success']:>7} | "
                      f"{row['unsafe_moves']:12d} | {row['avg_fixes']:9.2f} | {row['human_asks']:10d} | "
                      f"{row['avg_final_error_cm']:6.2f}")
        tok_in = sum(x["llm_input_tokens"] for x in rows)
        tok_out = sum(x["llm_output_tokens"] for x in rows)
        if tok_in:
            print(f"\nClaude tokens: {tok_in} in / {tok_out} out")
        with open(f"{base}/eval_table.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            w.writeheader()
            w.writerows(table)
        print(f"\nSaved {base}/trials.csv and {base}/eval_table.csv")


if __name__ == "__main__":
    main()
