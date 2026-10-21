"""
ga_sim.py
=========
A headless, lightweight re-implementation of the CoinTex dynamics that matter to
the agent, so that GA settings can be compared over many repeated runs instead of
a handful of slow GUI sessions.

What is reproduced exactly (read off main_updated.py):
  * coin layout          -- the screen is split into num_coins vertical strips and
                            one coin is dropped uniformly at random in each
  * coin pickup rule     -- bounding boxes must overlap AND the centre gap must be
                            within width/3 and height/3
  * monster damage rule  -- overlap AND centre gap within width/2 and height/2
  * fire damage rule     -- overlap AND centre gap within width/3 and height/3
  * life                 -- one hit per frame of contact; num_collisions_level
                            hits ends the episode
  * character motion     -- a linear animation to the requested point lasting
                            char_anim_duration seconds *regardless of distance*
  * monster motion       -- linear animation to a uniformly random point, then a
                            new random point, forever

Because pos_hint is a fraction of the layout, and both the centre gap and the
"width/3" threshold scale with the same layout width, the pixel-space collision
tests reduce *exactly* to per-axis tests in normalised units.  The simulator is
therefore geometrically faithful up to one set of unknowns: the widget
size_hints and animation durations, which live in the .kv file rather than in
main_updated.py.  Those are collected in SimParams below, with the defaults
noted as assumptions.  Absolute KPI values therefore should be confirmed in the
real game (main_improved.py logs the same KPIs); the *relative* ranking of GA
configurations, which is what the tuning experiments need, is robust to them.

COMP6009 Lab Task 2.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

from ga_agent import AgentConfig, BASELINE_CONFIG, IMPROVED_CONFIG, World, decide

# --------------------------------------------------------------------------- #
#  Level table, copied from the Screen subclasses in main_updated.py
# --------------------------------------------------------------------------- #
LEVELS: Dict[int, Dict[str, int]] = {
    1:  dict(coins=5,  monsters=1, fires=0, life=20),
    2:  dict(coins=8,  monsters=1, fires=0, life=30),
    3:  dict(coins=12, monsters=1, fires=0, life=30),
    4:  dict(coins=10, monsters=1, fires=1, life=20),
    5:  dict(coins=15, monsters=1, fires=2, life=20),
    6:  dict(coins=12, monsters=1, fires=3, life=20),
    7:  dict(coins=10, monsters=3, fires=0, life=25),
    8:  dict(coins=15, monsters=2, fires=0, life=25),
    9:  dict(coins=12, monsters=2, fires=0, life=25),
    10: dict(coins=14, monsters=3, fires=0, life=30),
    11: dict(coins=15, monsters=2, fires=1, life=30),
    12: dict(coins=12, monsters=2, fires=1, life=30),
    13: dict(coins=10, monsters=2, fires=2, life=20),
    14: dict(coins=15, monsters=0, fires=6, life=30),
    15: dict(coins=16, monsters=2, fires=3, life=30),
    16: dict(coins=15, monsters=3, fires=2, life=35),
    17: dict(coins=10, monsters=0, fires=4, life=30),
    18: dict(coins=15, monsters=3, fires=4, life=30),
    19: dict(coins=12, monsters=0, fires=6, life=30),
    20: dict(coins=15, monsters=0, fires=8, life=30),
    21: dict(coins=18, monsters=2, fires=4, life=30),
    22: dict(coins=20, monsters=2, fires=4, life=30),
    23: dict(coins=25, monsters=2, fires=2, life=30),
    24: dict(coins=20, monsters=3, fires=2, life=30),
}


# --------------------------------------------------------------------------- #
#  Simulation parameters (the ".kv unknowns")
# --------------------------------------------------------------------------- #
@dataclass
class SimParams:
    fps: int = 60
    char_anim_duration: float = 0.25          # from the Level classes
    char_size: Tuple[float, float] = (0.15, 0.15)    # MEASURED: <Character@Image> in cointex.kv
    coin_size: Tuple[float, float] = (0.05, 0.05)    # from screen_on_pre_enter
    monster_size: Tuple[float, float] = (0.15, 0.15)  # MEASURED: <Monster@Image> in cointex.kv
    fire_size: Tuple[float, float] = (0.03, 0.03)     # MEASURED: <Fire@Label> in cointex.kv
    monst_anim_low: float = 1.5               # assumption
    monst_anim_high: float = 3.5              # assumption
    fire_anim_duration: float = 2.0           # assumption
    max_episode_seconds: float = 60.0


PARAMS = SimParams()


# --------------------------------------------------------------------------- #
#  Little animation helper: linear interpolation with a fixed duration
# --------------------------------------------------------------------------- #
class LinearMover:
    def __init__(self, pos: Tuple[float, float]):
        self.pos = list(pos)
        self._from = list(pos)
        self._to = list(pos)
        self._t = 0.0
        self._dur = 1.0
        self.done = True

    def start(self, target: Tuple[float, float], duration: float) -> None:
        self._from = list(self.pos)
        self._to = [float(target[0]), float(target[1])]
        self._t = 0.0
        self._dur = max(duration, 1e-6)
        self.done = False

    def step(self, dt: float) -> None:
        if self.done:
            return
        self._t += dt
        u = min(1.0, self._t / self._dur)
        self.pos[0] = self._from[0] + u * (self._to[0] - self._from[0])
        self.pos[1] = self._from[1] + u * (self._to[1] - self._from[1])
        if u >= 1.0:
            self.done = True


def _overlap_and_gap(a: Tuple[float, float], sa: Tuple[float, float],
                     b: Tuple[float, float], sb: Tuple[float, float],
                     divisor: float) -> bool:
    """Reproduce `collide_widget(...) and abs(dx) <= w/div and abs(dy) <= h/div`,
    where w and h are the *character's* size and div is 2 or 3."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    if dx >= (sa[0] + sb[0]) / 2.0 or dy >= (sa[1] + sb[1]) / 2.0:
        return False                       # bounding boxes do not overlap
    return dx <= sa[0] / divisor and dy <= sa[1] / divisor


# --------------------------------------------------------------------------- #
#  One episode
# --------------------------------------------------------------------------- #
def run_episode(level: int, cfg: AgentConfig, seed: int = 0,
                params: SimParams = PARAMS) -> Dict[str, float]:
    spec = LEVELS[level]
    # The reachable set for the character's centre follows from its size_hint.
    # The baseline keeps its original (0.3, 1.3) init range on purpose, so the
    # comparison is against the algorithm as shipped.
    if cfg.fitness != "baseline":
        _w, _h = params.char_size
        cfg = replace(cfg, bounds_x=(_w / 2.0, 1.0 - _w / 2.0),
                           bounds_y=(_h / 2.0, 1.0 - _h / 2.0))
    rng = random.Random(seed)
    ga_rng = random.Random(seed + 991)

    dt = 1.0 / params.fps
    frames_per_decision = max(1, int(round(params.char_anim_duration * params.fps)))

    # --- coins: exactly the placement loop from screen_on_pre_enter --------
    n_coins = spec["coins"]
    section = 1.0 / n_coins
    cw, ch = params.coin_size
    coins: List[List[float]] = []
    for k in range(n_coins):
        x = rng.uniform(section * k, section * (k + 1) - cw)
        y = rng.uniform(0.0, 1.0 - ch)
        coins.append([x + cw / 2.0, y + ch / 2.0])      # store centres

    # --- character --------------------------------------------------------
    chw, chh = params.char_size
    char = LinearMover((chw / 2.0, chh / 2.0))           # pos_hint (0,0)

    # --- monsters ---------------------------------------------------------
    mw, mh = params.monster_size
    monsters: List[LinearMover] = []
    monster_durations: List[float] = []
    for _ in range(spec["monsters"]):
        m = LinearMover((0.8 + mw / 2.0, 0.8 + mh / 2.0))
        d = rng.uniform(params.monst_anim_low, params.monst_anim_high)
        m.start((rng.uniform(mw / 2, 1 - mw / 2), rng.uniform(mh / 2, 1 - mh / 2)), d)
        monsters.append(m)
        monster_durations.append(d)

    # --- fires: a repeating sweep between two points ----------------------
    fw, fh = params.fire_size
    fires = []
    for _ in range(spec["fires"]):
        a = (rng.uniform(fw / 2, 1 - fw / 2), rng.uniform(fh / 2, 1 - fh / 2))
        b = (rng.uniform(fw / 2, 1 - fw / 2), rng.uniform(fh / 2, 1 - fh / 2))
        mover = LinearMover(a)
        mover.start(b, params.fire_anim_duration)
        fires.append([mover, a, b, 1])                  # 1 == heading towards b

    hits = 0
    life_total = spec["life"]
    t = 0.0
    decisions = 0
    previous_best: Optional[Tuple[float, float]] = None
    last_fitness: Optional[float] = None
    prev_monster_pos = [tuple(m.pos) for m in monsters]

    collected = 0
    max_frames = int(params.max_episode_seconds * params.fps)

    for frame in range(max_frames):
        # ---------------- decision tick -------------------------------- #
        if frame % frames_per_decision == 0 and coins:
            vel = []
            for i, m in enumerate(monsters):
                vx = (m.pos[0] - prev_monster_pos[i][0])
                vy = (m.pos[1] - prev_monster_pos[i][1])
                vel.append((vx, vy))
            prev_monster_pos = [tuple(m.pos) for m in monsters]

            world = World(
                char=(char.pos[0], char.pos[1]),
                coins=[(c[0], c[1]) for c in coins],
                monsters=[(m.pos[0], m.pos[1]) for m in monsters],
                fires=[(f[0].pos[0], f[0].pos[1]) for f in fires],
                monster_vel=vel,
            )
            d = decide(world, cfg, previous_best, ga_rng)
            decisions += 1
            previous_best = d.target

            move = True
            if cfg.move_policy == "on_change":
                # the original only issues a move when the best fitness changed
                move = (last_fitness is None) or (abs(d.fitness - last_fitness) > 1e-12)
            last_fitness = d.fitness

            if move:
                char.start(d.target, params.char_anim_duration)

        # ---------------- advance animations --------------------------- #
        char.step(dt)
        for i, m in enumerate(monsters):
            m.step(dt)
            if m.done:
                dur = rng.uniform(params.monst_anim_low, params.monst_anim_high)
                m.start((rng.uniform(mw / 2, 1 - mw / 2),
                         rng.uniform(mh / 2, 1 - mh / 2)), dur)
        for f in fires:
            mover, a, b, heading = f
            mover.step(dt)
            if mover.done:
                f[3] = -heading
                mover.start(a if heading == 1 else b, params.fire_anim_duration)

        t += dt

        # ---------------- coin pickup ---------------------------------- #
        remaining = []
        for c in coins:
            if _overlap_and_gap(tuple(char.pos), params.char_size,
                                (c[0], c[1]), params.coin_size, 3.0):
                collected += 1
            else:
                remaining.append(c)
        coins = remaining
        if not coins:
            break

        # ---------------- damage --------------------------------------- #
        for m in monsters:
            if _overlap_and_gap(tuple(char.pos), params.char_size,
                                (m.pos[0], m.pos[1]), params.monster_size, 2.0):
                hits += 1
        for f in fires:
            if _overlap_and_gap(tuple(char.pos), params.char_size,
                                (f[0].pos[0], f[0].pos[1]), params.fire_size, 3.0):
                hits += 1
        if hits >= life_total:
            break

    success = (len(coins) == 0) and hits < life_total
    died = hits >= life_total
    return dict(
        config=cfg.name,
        level=level,
        seed=seed,
        coins_total=n_coins,
        coins_collected=collected,
        collection_rate=collected / n_coins,
        success=int(success),
        died=int(died),
        timed_out=int(not success and not died),
        time_s=round(t, 3),
        life_remaining=round(max(0.0, 1.0 - hits / life_total), 4),
        decisions=decisions,
    )


# --------------------------------------------------------------------------- #
#  Experiment driver
# --------------------------------------------------------------------------- #
def run_experiment(configs: List[AgentConfig], levels: List[int], runs: int,
                   out_csv: str, params: SimParams = PARAMS) -> List[Dict]:
    rows: List[Dict] = []
    for cfg in configs:
        for lvl in levels:
            for r in range(runs):
                rows.append(run_episode(lvl, cfg, seed=1000 * lvl + r, params=params))
            done = [x for x in rows if x["config"] == cfg.name and x["level"] == lvl]
            print(f"  {cfg.name:<22} level {lvl:>2}  "
                  f"success {100*statistics.mean(x['success'] for x in done):5.1f}%  "
                  f"coins {100*statistics.mean(x['collection_rate'] for x in done):5.1f}%  "
                  f"time {statistics.mean(x['time_s'] for x in done):5.1f}s  "
                  f"life {100*statistics.mean(x['life_remaining'] for x in done):5.1f}%",
                  flush=True)

    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {out_csv}")
    return rows


def summarise(rows: List[Dict], by=("config",)) -> List[Dict]:
    keys = sorted({tuple(r[k] for k in by) for r in rows})
    out = []
    for key in keys:
        sel = [r for r in rows if tuple(r[k] for k in by) == key]
        out.append(dict(
            **{k: v for k, v in zip(by, key)},
            n=len(sel),
            success_rate=round(100 * statistics.mean(r["success"] for r in sel), 1),
            collection_rate=round(100 * statistics.mean(r["collection_rate"] for r in sel), 1),
            mean_time=round(statistics.mean(r["time_s"] for r in sel), 2),
            life_remaining=round(100 * statistics.mean(r["life_remaining"] for r in sel), 1),
            death_rate=round(100 * statistics.mean(r["died"] for r in sel), 1),
        ))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CoinTex GA experiment runner")
    p.add_argument("--levels", default="1,5,10,20", help="comma separated")
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--out", default="results/raw_results.csv")
    args = p.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    rows = run_experiment([BASELINE_CONFIG, IMPROVED_CONFIG], levels, args.runs, args.out)
    for r in summarise(rows, by=("config", "level")):
        print(r)
