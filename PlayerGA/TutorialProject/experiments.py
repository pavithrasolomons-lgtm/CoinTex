"""
experiments.py
==============
Runs every experiment quoted in the report and writes the CSV summaries and the
figures.  Four families of experiment:

  E0  headline     -- baseline vs improved over a spread of levels
  E1  ablation     -- turn the improvements on one at a time, so the gain can be
                      attributed to a specific change rather than to "the new
                      version"
  E2  sweeps       -- population size, generations per decision, selection
                      operator, mutation operator
  E3  benchmark    -- the tuned agent over all 24 levels

Each stage runs separately so a long sweep can be resumed:

    python3 experiments.py --stage e0
    python3 experiments.py --stage e1
    python3 experiments.py --stage e2pop
    python3 experiments.py --stage e2gen
    python3 experiments.py --stage e2sel
    python3 experiments.py --stage e2mut
    python3 experiments.py --stage e3
    python3 experiments.py --stage plots      # figures + tables.md from the CSVs

COMP6009 Lab Task 2.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from dataclasses import replace
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ga_agent import AgentConfig, BASELINE_CONFIG, IMPROVED_CONFIG
from ga_sim import LEVELS, run_experiment, summarise

OUT = "results"
FIGS = "results/figures"

C_BASE, C_IMP, C_MID = "#B0623C", "#2F6F6B", "#8A8F98"

HEAD_LEVELS = [1, 5, 10, 16, 20, 24]
SWEEP_LEVELS = [10, 20, 24]


# --------------------------------------------------------------------------- #
#  IO helpers
# --------------------------------------------------------------------------- #
def read_rows(path: str) -> List[Dict]:
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("level", "seed", "coins_total", "coins_collected", "success",
                  "died", "timed_out", "decisions"):
            if k in r:
                r[k] = int(r[k])
        for k in ("collection_rate", "time_s", "life_remaining"):
            if k in r:
                r[k] = float(r[k])
    return rows


def write_summary(rows: List[Dict], by, path: str) -> List[Dict]:
    summ = summarise(rows, by=by)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    return summ


def md_table(summ: List[Dict], label_key: str, label_title: str) -> str:
    lines = [f"| {label_title} | Success rate (%) | Coins collected (%) | "
             f"Mean time (s) | Life remaining (%) | Death rate (%) |",
             "|---|---|---|---|---|---|"]
    for r in summ:
        lines.append(f"| {r[label_key]} | {r['success_rate']} | {r['collection_rate']} | "
                     f"{r['mean_time']} | {r['life_remaining']} | {r['death_rate']} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Configurations under test
# --------------------------------------------------------------------------- #
def ablation_configs() -> List[AgentConfig]:
    base = BASELINE_CONFIG

    a1 = replace(base, name="A1_replanning", move_policy="always",
                 num_generations=5, bounds_x=(0.04, 0.96), bounds_y=(0.06, 0.94))

    a2 = replace(a1, name="A2_plus_operators",
                 sol_per_pop=20, num_parents_mating=8, num_generations=25,
                 parent_selection_type="tournament", crossover_type="uniform",
                 crossover_probability=0.9, mutation_type="adaptive",
                 mutation_probability=(0.6, 0.15),
                 random_mutation_min_val=-0.25, random_mutation_max_val=0.25,
                 mutation_by_replacement=False, keep_elitism=2)

    a3 = replace(IMPROVED_CONFIG, name="A3_no_path_sweep", w_sweep=0.0)
    a4 = replace(IMPROVED_CONFIG, name="A4_no_hazard_term",
                 w_monster=0.0, w_fire=0.0, predict_monsters=False)
    a5 = replace(IMPROVED_CONFIG, name="A5_no_seeding",
                 informed_init=False, warm_start=False)
    a6 = replace(IMPROVED_CONFIG, name="A6_full_improved")
    return [replace(base, name="A0_baseline"), a1, a2, a3, a4, a5, a6]


def sweep_population() -> List[AgentConfig]:
    return [replace(IMPROVED_CONFIG, name=f"pop_{n:02d}", sol_per_pop=n,
                    num_parents_mating=max(2, n // 2),
                    keep_elitism=max(1, n // 10))
            for n in (5, 10, 20, 40, 80)]


def sweep_generations() -> List[AgentConfig]:
    return [replace(IMPROVED_CONFIG, name=f"gen_{g:03d}", num_generations=g)
            for g in (1, 5, 10, 25, 50)]


def sweep_selection() -> List[AgentConfig]:
    return [replace(IMPROVED_CONFIG, name=f"sel_{s}", parent_selection_type=s)
            for s in ("random", "rws", "rank", "tournament", "sss")]


def sweep_mutation() -> List[AgentConfig]:
    return [
        replace(IMPROVED_CONFIG, name="mut_swap", mutation_type="swap",
                mutation_probability=None, mutation_percent_genes=10),
        replace(IMPROVED_CONFIG, name="mut_random_replace", mutation_type="random",
                mutation_probability=0.3, mutation_by_replacement=True),
        replace(IMPROVED_CONFIG, name="mut_random_local", mutation_type="random",
                mutation_probability=0.3, mutation_by_replacement=False),
        replace(IMPROVED_CONFIG, name="mut_adaptive"),
    ]


STAGES = {
    "e0": dict(cfgs=lambda: [replace(BASELINE_CONFIG, name="baseline"),
                             replace(IMPROVED_CONFIG, name="improved")],
               levels=HEAD_LEVELS, csv="e0_headline.csv"),
    "e1": dict(cfgs=ablation_configs, levels=SWEEP_LEVELS, csv="e1_ablation.csv"),
    "e2pop": dict(cfgs=sweep_population, levels=SWEEP_LEVELS, csv="e2_population.csv"),
    "e2gen": dict(cfgs=sweep_generations, levels=SWEEP_LEVELS, csv="e2_generations.csv"),
    "e2sel": dict(cfgs=sweep_selection, levels=SWEEP_LEVELS, csv="e2_selection.csv"),
    "e2mut": dict(cfgs=sweep_mutation, levels=SWEEP_LEVELS, csv="e2_mutation.csv"),
    "e3": dict(cfgs=lambda: [replace(IMPROVED_CONFIG, name="improved")],
               levels=sorted(LEVELS.keys()), csv="e3_all_levels.csv"),
}


# --------------------------------------------------------------------------- #
#  Plotting
# --------------------------------------------------------------------------- #
def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def _bar(ax, labels, values, color, ylabel, title, rotate=25, fmt="{:.0f}"):
    bars = ax.bar(labels, values, color=color, edgecolor="white", linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
    _style(ax)
    plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right", fontsize=8)
    top = max(values) if max(values) else 1.0
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + top * 0.02, fmt.format(v),
                ha="center", fontsize=8)
    ax.set_ylim(0, top * 1.18)


def sweep_figure(name: str, title: str, fname: str, label_title: str) -> str:
    rows = read_rows(f"{OUT}/{name}")
    ss = write_summary(rows, ("config",), f"{OUT}/{name.replace('.csv', '')}_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    labs = [r["config"] for r in ss]
    _bar(axes[0], labs, [r["success_rate"] for r in ss], C_IMP,
         "Success rate (%)", f"Success vs {title}")
    _bar(axes[1], labs, [r["mean_time"] for r in ss], C_MID,
         "Mean time (s)", f"Time vs {title}", fmt="{:.1f}")
    _bar(axes[2], labs, [r["life_remaining"] for r in ss], "#6B7F9E",
         "Life remaining (%)", f"Survival vs {title}")
    fig.tight_layout()
    fig.savefig(f"{FIGS}/{fname}", dpi=160)
    plt.close(fig)
    return md_table(ss, "config", label_title)


def make_plots() -> None:
    os.makedirs(FIGS, exist_ok=True)
    tables = []

    # --- E0 ---------------------------------------------------------------
    rows0 = read_rows(f"{OUT}/e0_headline.csv")
    s0 = write_summary(rows0, ("config",), f"{OUT}/e0_summary.csv")
    write_summary(rows0, ("config", "level"), f"{OUT}/e0_summary_by_level.csv")
    tables.append(("E0 — Baseline vs improved (levels "
                   + ", ".join(map(str, HEAD_LEVELS)) + ")",
                   md_table(s0, "config", "Configuration")))

    levels = sorted({r["level"] for r in rows0})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, metric, lab in ((axes[0], "success", "Success rate (%)"),
                            (axes[1], "collection_rate", "Coins collected (%)"),
                            (axes[2], "life_remaining", "Life remaining (%)")):
        width = 0.38
        for i, c in enumerate(["baseline", "improved"]):
            vals = [100 * statistics.mean(
                r[metric] for r in rows0 if r["config"] == c and r["level"] == lv)
                for lv in levels]
            xs = [j + (i - 0.5) * width for j in range(len(levels))]
            ax.bar(xs, vals, width=width * 0.92,
                   color=C_BASE if c == "baseline" else C_IMP, label=c,
                   edgecolor="white", linewidth=0.6)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([f"L{l}" for l in levels], fontsize=8)
        ax.set_ylabel(lab)
        ax.set_ylim(0, 108)
        ax.set_title(lab, loc="left", fontweight="bold", fontsize=10)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig1_baseline_vs_improved.png", dpi=160)
    plt.close(fig)

    # --- E1 ---------------------------------------------------------------
    rows1 = read_rows(f"{OUT}/e1_ablation.csv")
    s1 = write_summary(rows1, ("config",), f"{OUT}/e1_summary.csv")
    tables.append(("E1 — Ablation (levels " + ", ".join(map(str, SWEEP_LEVELS)) + ")",
                   md_table(s1, "config", "Variant")))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    labs = [r["config"] for r in s1]
    cols = [C_BASE] + [C_MID] * (len(labs) - 2) + [C_IMP]
    _bar(axes[0], labs, [r["success_rate"] for r in s1], cols,
         "Success rate (%)", "Success by variant")
    _bar(axes[1], labs, [r["life_remaining"] for r in s1], cols,
         "Life remaining (%)", "Survival by variant")
    _bar(axes[2], labs, [r["mean_time"] for r in s1], cols,
         "Mean time (s)", "Time by variant", fmt="{:.1f}")
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig2_ablation.png", dpi=160)
    plt.close(fig)

    # --- E2 ---------------------------------------------------------------
    for fname, title, fig_name, lab in (
            ("e2_population.csv", "population size", "fig3_population.png", "Population"),
            ("e2_generations.csv", "generations", "fig4_generations.png", "Generations"),
            ("e2_selection.csv", "selection", "fig5_selection.png", "Selection"),
            ("e2_mutation.csv", "mutation", "fig6_mutation.png", "Mutation")):
        if os.path.exists(f"{OUT}/{fname}"):
            tables.append((f"E2 — Sweep: {title} (levels "
                           + ", ".join(map(str, SWEEP_LEVELS)) + ")",
                           sweep_figure(fname, title, fig_name, lab)))

    # --- E3 ---------------------------------------------------------------
    if os.path.exists(f"{OUT}/e3_all_levels.csv"):
        rows3 = read_rows(f"{OUT}/e3_all_levels.csv")
        s3 = write_summary(rows3, ("level",), f"{OUT}/e3_summary_by_level.csv")
        tables.append(("E3 — Tuned agent across all 24 levels",
                       md_table(s3, "level", "Level")))
        fig, ax = plt.subplots(figsize=(9, 3.6))
        lv = [r["level"] for r in s3]
        ax.plot(lv, [r["success_rate"] for r in s3], "-o", color=C_IMP, ms=4,
                label="Success rate (%)")
        ax.plot(lv, [r["life_remaining"] for r in s3], "-s", color=C_MID, ms=4,
                label="Life remaining (%)")
        ax.plot(lv, [r["collection_rate"] for r in s3], "-^", color=C_BASE, ms=4,
                label="Coins collected (%)")
        ax.set_xlabel("Level")
        ax.set_ylabel("%")
        ax.set_ylim(0, 108)
        ax.set_xticks(lv)
        ax.tick_params(labelsize=8)
        ax.set_title("Tuned agent across all 24 levels", loc="left",
                     fontweight="bold", fontsize=10)
        ax.legend(frameon=False, fontsize=8, ncol=3)
        _style(ax)
        ax.grid(alpha=0.25, linewidth=0.6)
        fig.tight_layout()
        fig.savefig(f"{FIGS}/fig7_all_levels.png", dpi=160)
        plt.close(fig)

    with open(f"{OUT}/tables.md", "w") as fh:
        for title, tbl in tables:
            fh.write(f"\n### {title}\n\n{tbl}\n")
    print(f"figures -> {FIGS}/ ; tables -> {OUT}/tables.md")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES.keys()) + ["plots"])
    ap.add_argument("--runs", type=int, default=20)
    args = ap.parse_args()

    if args.stage == "plots":
        make_plots()
        return

    spec = STAGES[args.stage]
    print(f"=== {args.stage} ===", flush=True)
    run_experiment(spec["cfgs"](), spec["levels"], args.runs, f"{OUT}/{spec['csv']}")


if __name__ == "__main__":
    main()
