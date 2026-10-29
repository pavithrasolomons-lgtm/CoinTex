"""
sim_partA.py
============
Group:            Colombo-group-08, COMP6009 Final Assignment Part A

Headless re-implementation of the CoinTex rules that matter to an agent, used
to develop and tune the agents quickly. Final results for the report come from
the real game (main_partA.py); this simulator is a development tool.

Read directly from the game files, so nothing is guessed:
  * level table (coins, monsters, fires, lives, animation time) from the
    Level classes in main_work_greedy.py
  * monster speeds (monst_anim_duration_low/high) and fire paths
    (fire_start/end_pos_hint, fire_anim_duration) from cointex.kv
  * sizes from cointex.kv: character 0.15, monster 0.15, fire 0.03, coin 0.05

Game rules reproduced (from main_work_greedy.py):
  * coins: one per vertical strip, uniform position inside the strip
  * coin pickup: on every frame the CHARACTER moves, centre gap <= w/3, h/3
  * monster hit: on every frame a MONSTER moves, centre gap <= w/2, h/2
  * fire hit: on every frame a FIRE moves, centre gap <= w/3, h/3
  * one life point lost per hit frame; num_collisions_level points = death
  * character move: linear, char_anim_duration seconds whatever the distance
  * monsters: linear moves to random points, random durations, forever
  * fires: bounce start <-> end, fire_anim_duration per leg, forever

Usage:
    python sim_partA.py --agents greedy astar --levels 1-24 --runs 5
"""

import argparse
import csv
import math
import os
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from partA_common import EpisodeOver, FireInfo, GameIO, World

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 60
CHAR_SIZE = (0.15, 0.15)
MONSTER_SIZE = (0.15, 0.15)
FIRE_SIZE = (0.03, 0.03)
COIN_SIZE = (0.05, 0.05)
TIMEOUT_S = 60.0


# --------------------------------------------------------------------------- #
#  Level table
# --------------------------------------------------------------------------- #
@dataclass
class LevelSpec:
    num: int
    num_coins: int
    num_monsters: int
    num_fires: int
    lives: int
    anim: float
    monster_durations: List[Tuple[float, float]] = field(default_factory=list)
    fire_paths: List[Tuple[Tuple[float, float], Tuple[float, float], float]] = field(default_factory=list)


def _dict_xy(s):
    x = float(re.search(r"'x'\s*:\s*(-?[\d.]+)", s).group(1))
    y = float(re.search(r"'y'\s*:\s*(-?[\d.]+)", s).group(1))
    return (x, y)


def load_levels(py_file="main_work_greedy.py", kv_file="cointex.kv") -> Dict[int, LevelSpec]:
    py = open(os.path.join(HERE, py_file), encoding="utf-8").read()
    kv = open(os.path.join(HERE, kv_file), encoding="utf-8").read()
    levels = {}
    for m in re.finditer(r"class Level(\d+)\(.*?\):(.*?)(?=\nclass |\napp = |\Z)", py, re.S):
        n, body = int(m.group(1)), m.group(2)
        g = lambda k: float(re.search(k + r"\s*=\s*([\d.]+)", body).group(1))
        levels[n] = LevelSpec(n, int(g("num_coins")), int(g("num_monsters")),
                              int(g("num_fires")), int(g("num_collisions_level")),
                              g("char_anim_duration"))
    blocks = re.split(r"\n<Level(\d+)>:", kv)
    for i in range(1, len(blocks), 2):
        n, body = int(blocks[i]), blocks[i + 1].split("\n<")[0]
        spec = levels[n]
        for k in range(1, spec.num_monsters + 1):
            mm = re.search(r"id: monster%d_image_lvl%d\s*\n\s*monst_anim_duration_low:\s*([\d.]+)\s*\n\s*monst_anim_duration_high:\s*([\d.]+)" % (k, n), body)
            spec.monster_durations.append((float(mm.group(1)), float(mm.group(2))))
        for k in range(1, spec.num_fires + 1):
            fm = re.search(r"id: fire%d_lvl%d\s*\n\s*fire_start_pos_hint:\s*(\{[^}]*\})\s*\n\s*fire_end_pos_hint:\s*(\{[^}]*\})\s*\n\s*fire_anim_duration:\s*([\d.]+)" % (k, n), body)
            spec.fire_paths.append((_dict_xy(fm.group(1)), _dict_xy(fm.group(2)), float(fm.group(3))))
    return levels


# --------------------------------------------------------------------------- #
#  Movers
# --------------------------------------------------------------------------- #
class Tween:
    """Linear animation of a pos_hint, like kivy.animation.Animation."""

    def __init__(self, pos):
        self.pos = tuple(pos)
        self.start = self.pos
        self.end = self.pos
        self.t = 0.0
        self.dur = 0.0

    def go(self, end, dur):
        self.start, self.end, self.t, self.dur = self.pos, tuple(end), 0.0, max(1e-6, dur)

    @property
    def active(self):
        return self.t < self.dur

    def step(self, dt):
        if not self.active:
            return False
        self.t = min(self.dur, self.t + dt)
        a = self.t / self.dur
        old = self.pos
        self.pos = (self.start[0] + (self.end[0] - self.start[0]) * a,
                    self.start[1] + (self.end[1] - self.start[1]) * a)
        return self.pos != old


class Game:
    def __init__(self, spec: LevelSpec, rng: random.Random):
        self.spec, self.rng = spec, rng
        self.t = 0.0
        self.frame = 0
        self.char = Tween((0.0, 0.0))
        self.hits = 0
        self.killed = False
        self.coins: Dict[str, Tuple[float, float]] = {}
        section = 1.0 / spec.num_coins
        for k in range(spec.num_coins):
            x = rng.uniform(section * k, section * (k + 1) - COIN_SIZE[0])
            y = rng.uniform(0, 1 - COIN_SIZE[1])
            self.coins["coin%d" % k] = (x, y)
        self.collected = 0
        self.monsters = []
        for lo, hi in spec.monster_durations:
            tw = Tween((0.8, 0.8))
            tw.lo, tw.hi = lo, hi
            self._new_monster_target(tw)
            self.monsters.append(tw)
        self.fires = []
        for start, end, dur in spec.fire_paths:
            tw = Tween((1.1, 1.1))
            tw.path = (start, end, dur)
            tw.leg = 0
            tw.go(start, dur)
            self.fires.append(tw)

    def _new_monster_target(self, tw):
        lim = 1 - MONSTER_SIZE[0] / 4
        tw.go((self.rng.uniform(0.0, lim), self.rng.uniform(0.0, lim)),
              self.rng.uniform(tw.lo, tw.hi))

    @property
    def done(self):
        return self.killed or not self.coins

    def step(self):
        dt = 1.0 / FPS
        self.t += dt
        self.frame += 1
        # monsters
        for tw in self.monsters:
            moved = tw.step(dt)
            if not tw.active:
                self._new_monster_target(tw)
            if moved and not self.killed:
                self._hazard_check(tw.pos, MONSTER_SIZE, CHAR_SIZE[0] / 2, CHAR_SIZE[1] / 2)
        # fires
        for tw in self.fires:
            moved = tw.step(dt)
            if not tw.active:
                start, end, dur = tw.path
                tw.leg = 1 - tw.leg
                tw.go(end if tw.leg == 1 else start, dur)
            if moved and not self.killed:
                self._hazard_check(tw.pos, FIRE_SIZE, CHAR_SIZE[0] / 3, CHAR_SIZE[1] / 3)
        # character
        if not self.killed and self.char.step(dt):
            self._coin_check()

    def _overlap(self, p1, s1, p2, s2):
        return (p1[0] < p2[0] + s2[0] and p1[0] + s1[0] > p2[0] and
                p1[1] < p2[1] + s2[1] and p1[1] + s1[1] > p2[1])

    def _hazard_check(self, hpos, hsize, gx, gy):
        cc = (self.char.pos[0] + CHAR_SIZE[0] / 2, self.char.pos[1] + CHAR_SIZE[1] / 2)
        hc = (hpos[0] + hsize[0] / 2, hpos[1] + hsize[1] / 2)
        if self._overlap(self.char.pos, CHAR_SIZE, hpos, hsize) and \
                abs(cc[0] - hc[0]) <= gx and abs(cc[1] - hc[1]) <= gy:
            self.hits += 1
            if self.hits >= self.spec.lives:
                self.killed = True

    def _coin_check(self):
        cc = (self.char.pos[0] + CHAR_SIZE[0] / 2, self.char.pos[1] + CHAR_SIZE[1] / 2)
        for key in list(self.coins):
            pos = self.coins[key]
            kc = (pos[0] + COIN_SIZE[0] / 2, pos[1] + COIN_SIZE[1] / 2)
            if self._overlap(self.char.pos, CHAR_SIZE, pos, COIN_SIZE) and \
                    abs(cc[0] - kc[0]) <= CHAR_SIZE[0] / 3 and abs(cc[1] - kc[1]) <= CHAR_SIZE[1] / 3:
                del self.coins[key]
                self.collected += 1

    def world(self) -> World:
        fires = []
        for tw in self.fires:
            start, end, dur = tw.path
            fires.append(FireInfo(tw.pos, FIRE_SIZE, start, end, dur))
        return World(t=self.t, char_pos=self.char.pos, char_size=CHAR_SIZE,
                     coins=list(self.coins.values()), coin_size=COIN_SIZE,
                     monsters=[tw.pos for tw in self.monsters],
                     monster_sizes=[MONSTER_SIZE] * len(self.monsters),
                     fires=fires, char_anim_duration=self.spec.anim,
                     killed=self.killed)


# --------------------------------------------------------------------------- #
#  GameIO for the simulator: the agent runs in this thread, and every io call
#  advances the simulated clock (including the agent's own compute time).
# --------------------------------------------------------------------------- #
class SimIO(GameIO):
    def __init__(self, game: Game, compute_scale=1.0):
        self.game = game
        self.compute_scale = compute_scale
        self._acc = 0.0
        self._wall = time.perf_counter()
        self.moves = 0

    def _charge_compute(self):
        now = time.perf_counter()
        self._advance(max(1e-4, (now - self._wall) * self.compute_scale))

    def _advance(self, dt):
        self._acc += dt
        frame = 1.0 / FPS
        while self._acc >= frame:
            self._acc -= frame
            self.game.step()
            if self.game.done or self.game.t >= TIMEOUT_S:
                self._wall = time.perf_counter()
                raise EpisodeOver()
        self._wall = time.perf_counter()

    def snapshot(self) -> World:
        self._charge_compute()
        return self.game.world()

    def move(self, target):
        self._charge_compute()
        if self.game.killed:
            return
        self.moves += 1
        tp = (target[0] - CHAR_SIZE[0] / 2, target[1] - CHAR_SIZE[1] / 2)
        self.game.char.go(tp, self.game.spec.anim)

    def sleep(self, seconds):
        self._charge_compute()
        self._advance(seconds)


def make_agent(name):
    if name == "greedy":
        from agent_greedy import GreedyAgent
        return GreedyAgent()
    if name == "astar":
        from agent_astar import AStarAgent
        return AStarAgent()
    if name == "ga":
        from agent_ga import GAAgent
        return GAAgent()
    raise ValueError(name)


def run_episode(agent_name, spec, seed, compute_scale=1.0):
    rng = random.Random(seed)
    game = Game(spec, rng)
    io = SimIO(game, compute_scale)
    agent = make_agent(agent_name)
    random.seed(seed)
    try:
        agent.run(io)
        # agent returned on its own: let the clock run until the level ends
        while not game.done and game.t < TIMEOUT_S:
            io.sleep(0.5)
    except EpisodeOver:
        pass
    except Exception as e:                       # pragma: no cover
        if not isinstance(getattr(e, "__cause__", None), EpisodeOver) and "EpisodeOver" not in repr(e):
            raise
    success = (not game.coins) and not game.killed
    return dict(agent=agent_name, level=spec.num, seed=seed,
                coins_total=spec.num_coins, coins_collected=game.collected,
                success=int(success), died=int(game.killed),
                timed_out=int(not success and not game.killed),
                time_s=round(game.t, 2),
                life_remaining=round(max(0.0, 1 - game.hits / spec.lives), 3),
                moves=io.moves)


def parse_levels(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", default=["greedy", "astar"])
    ap.add_argument("--levels", default="1-24")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", default="")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    levels = load_levels()
    rows = []
    for agent in args.agents:
        for lv in parse_levels(args.levels):
            for r in range(args.runs):
                row = run_episode(agent, levels[lv], seed=1000 * lv + r)
                rows.append(row)
                if not args.quiet:
                    print(row, flush=True)
    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    # summary
    print("\nagent   success%  mean_time(success)  mean_coins%  died%  timeout%")
    for agent in args.agents:
        rs = [r for r in rows if r["agent"] == agent]
        succ = [r for r in rs if r["success"]]
        print("%-7s %7.1f  %18s  %11.1f  %5.1f  %8.1f" % (
            agent, 100 * len(succ) / len(rs),
            ("%.2f" % statistics.mean(r["time_s"] for r in succ)) if succ else "-",
            100 * statistics.mean(r["coins_collected"] / r["coins_total"] for r in rs),
            100 * statistics.mean(r["died"] for r in rs),
            100 * statistics.mean(r["timed_out"] for r in rs)))


if __name__ == "__main__":
    main()
