"""
analyse_partA.py
================
Author:           Pavithra Solomons
Last modified by: Pavithra Solomons
Last modified:    29 October 2026
Group:            Colombo-group-08, COMP6009 Final Assignment Part A

Summarise partA_results.csv for the report.

Outputs (next to the csv):
    partA_summary.csv      one row per agent
    partA_per_level.csv    one row per agent and level
    chart_success_by_level.png, chart_time_by_level.png, chart_overall.png
                           (only if matplotlib is installed:
                            pip install matplotlib)

Metrics
    success rate        % of episodes where every coin was collected
    level success rate  % of the 24 levels completed in at least 3 of 5 runs
    completion time     mean seconds to collect all coins (successful runs)
    life remaining      mean % of life left at the end of an episode
    RACR (proposed)     risk-adjusted collection rate =
                        coins collected per second x fraction of life left.
                        It rewards being fast AND unhurt in one number, and
                        still credits partial progress on failed runs.

Usage:  python analyse_partA.py [partA_results.csv]
"""

import csv
import os
import statistics as st
import sys
from collections import defaultdict

AGENTS = ["greedy", "ga", "astar"]
LABELS = {"greedy": "Greedy (original)", "ga": "Genetic algorithm",
          "astar": "A* danger map (proposed)"}
# validated categorical palette, fixed order (identity, not rank)
COLOURS = {"greedy": "#2a78d6", "ga": "#eb6834", "astar": "#1baf7a"}


def load(path):
    rows = list(csv.DictReader(open(path, newline="")))
    for r in rows:
        for k in ("level", "run", "coins_total", "coins_collected", "success",
                  "died", "timed_out", "hits", "moves"):
            r[k] = int(r[k])
        for k in ("time_s", "life_remaining", "collection_rate"):
            r[k] = float(r[k])
        r["racr"] = (r["coins_collected"] / r["time_s"] * r["life_remaining"]
                     if r["time_s"] > 0 else 0.0)
    return rows


def mean(xs):
    xs = list(xs)
    return st.mean(xs) if xs else float("nan")


def summarise(rows):
    agents = [a for a in AGENTS if any(r["agent"] == a for r in rows)]
    levels = sorted({r["level"] for r in rows})
    summary, per_level = [], []
    for a in agents:
        ra = [r for r in rows if r["agent"] == a]
        ok = [r for r in ra if r["success"]]
        lvl_done = 0
        for lv in levels:
            rl = [r for r in ra if r["level"] == lv]
            if not rl:
                continue
            okl = [r for r in rl if r["success"]]
            rate = len(okl) / len(rl)
            lvl_done += rate >= 0.6
            per_level.append(dict(
                agent=a, level=lv, runs=len(rl),
                success_rate=round(100 * rate, 1),
                mean_time_success=round(mean(r["time_s"] for r in okl), 2) if okl else "",
                mean_life_pct=round(100 * mean(r["life_remaining"] for r in rl), 1),
                mean_coins_pct=round(100 * mean(r["collection_rate"] for r in rl), 1),
                died=sum(r["died"] for r in rl), timeouts=sum(r["timed_out"] for r in rl),
                mean_racr=round(mean(r["racr"] for r in rl), 3)))
        n_levels = len({r["level"] for r in ra})
        summary.append(dict(
            agent=a, episodes=len(ra),
            success_rate=round(100 * len(ok) / len(ra), 1),
            levels_completed=lvl_done, levels_tested=n_levels,
            level_success_rate=round(100 * lvl_done / n_levels, 1),
            mean_time_success=round(mean(r["time_s"] for r in ok), 2) if ok else "",
            median_time_success=round(st.median([r["time_s"] for r in ok]), 2) if ok else "",
            mean_coins_pct=round(100 * mean(r["collection_rate"] for r in ra), 1),
            mean_life_pct=round(100 * mean(r["life_remaining"] for r in ra), 1),
            deaths=sum(r["died"] for r in ra), timeouts=sum(r["timed_out"] for r in ra),
            mean_hits=round(mean(r["hits"] for r in ra), 1),
            mean_racr=round(mean(r["racr"] for r in ra), 3)))
    return summary, per_level, agents, levels


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def charts(per_level, summary, agents, levels, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping charts (pip install matplotlib)")
        return
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7",
                         "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.spines.top": False,
                         "axes.spines.right": False})
    pl = {(r["agent"], r["level"]): r for r in per_level}
    width = 0.8 / len(agents)

    # 1. success rate by level (grouped bars)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for i, a in enumerate(agents):
        xs = [lv + (i - (len(agents) - 1) / 2) * width for lv in levels]
        ys = [pl[(a, lv)]["success_rate"] if (a, lv) in pl else 0 for lv in levels]
        ax.bar(xs, ys, width=width * 0.9, color=COLOURS[a], label=LABELS[a])
    ax.set_xticks(levels)
    ax.set_xlabel("Level")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=len(agents), loc="upper center",
              bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "chart_success_by_level.png"), dpi=200)
    plt.close(fig)

    # 2. completion time by level (successful runs only)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for a in agents:
        pts = [(lv, pl[(a, lv)]["mean_time_success"]) for lv in levels
               if (a, lv) in pl and pl[(a, lv)]["mean_time_success"] != ""]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLOURS[a],
                    linewidth=2, marker="o", markersize=5, label=LABELS[a])
            ax.annotate(LABELS[a].split(" (")[0], xy=pts[-1], xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9,
                        color="#52514e")
    ax.set_xticks(levels)
    ax.set_xlabel("Level")
    ax.set_ylabel("Mean time to collect all coins (s)")
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=len(agents), loc="upper center",
              bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "chart_time_by_level.png"), dpi=200)
    plt.close(fig)

    # 3. overall: three separate panels (no shared/dual axis)
    metrics = [("success_rate", "Success rate (%)"),
               ("mean_time_success", "Mean completion time (s)"),
               ("mean_racr", "RACR (coins/s x life)")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (key, title) in zip(axes, metrics):
        vals = [s[key] if s[key] != "" else 0 for s in summary]
        bars = ax.bar(range(len(summary)), vals, width=0.6,
                      color=[COLOURS[s["agent"]] for s in summary])
        for b, v in zip(bars, vals):
            ax.annotate(("%.1f" % v) if key != "mean_racr" else ("%.2f" % v),
                        xy=(b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=9,
                        color="#0b0b0b")
        ax.set_xticks(range(len(summary)))
        ax.set_xticklabels([LABELS[s["agent"]].split(" (")[0].replace(" danger map", "")
                            for s in summary], fontsize=9)
        ax.set_title(title, fontsize=10, color="#0b0b0b")
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "chart_overall.png"), dpi=200)
    plt.close(fig)
    print("charts written to", outdir)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "partA_results.csv"
    rows = load(path)
    summary, per_level, agents, levels = summarise(rows)
    outdir = os.path.dirname(os.path.abspath(path))
    write_csv(os.path.join(outdir, "partA_summary.csv"), summary)
    write_csv(os.path.join(outdir, "partA_per_level.csv"), per_level)

    print("\n%-8s %8s %9s %10s %9s %8s %7s %6s" % (
        "agent", "episodes", "success%", "levels", "time(s)", "life%", "deaths", "RACR"))
    for s in summary:
        print("%-8s %8d %9.1f %6d/%-3d %9s %8.1f %7d %6.2f" % (
            s["agent"], s["episodes"], s["success_rate"], s["levels_completed"],
            s["levels_tested"], s["mean_time_success"], s["mean_life_pct"],
            s["deaths"], s["mean_racr"]))
    charts(per_level, summary, agents, levels, outdir)


if __name__ == "__main__":
    main()
