"""
agent_astar.py
==============
Author:           Pavithra Solomons
Last modified by: Pavithra Solomons
Last modified:    29 October 2026
Group:            Colombo-group-08, COMP6009 Final Assignment Part A

Danger-aware A* agent for CoinTex (COMP6009 Final Assignment, Part A).

Why not greedy: main_work_greedy.py orders coins by Manhattan distance once,
then jumps through the whole route with its eyes closed. Monsters and fires
only remove coins from the list at planning time; the jumps themselves can go
straight through a hazard, and the plan is never refreshed while hazards move.

This agent re-plans every character move (one decision per 0.25 s animation):

 1. SENSE    positions of character, coins, monsters and fires; the velocity
             of every hazard from the previous snapshot.
 2. PREDICT  where each hazard will be over the next PREDICT_TAUS seconds.
             Monsters: constant velocity, with a safety margin that grows
             with look-ahead time (they turn at random).
             Fires: they bounce between two known end points, so their
             position is extrapolated along that segment with reflection.
 3. DANGER   a GRID x GRID map of the play area. A cell inside a predicted
             hit box (plus margin) is "hard" danger; outside it the danger
             decays exponentially with clearance.
 4. SEARCH   multi-goal A* from the character's cell to the cheapest coin.
             Edge cost = length * (1 + ALPHA * danger). Heuristic = straight-
             line distance to the closest coin: admissible and consistent,
             because no edge costs less than its length.
 5. ACT      shortcut along the A* path: jump to the farthest point of the
             path whose straight-line move is safe in SPACE-TIME (each sample
             point is checked against where the hazards will be when the
             character passes it). If the coin itself is reachable safely,
             jump straight to it. A jump takes 0.25 s whatever its length, so
             fewer, longer safe jumps also mean faster levels. If nothing on
             the path is safe, take the escape move with the largest
             clearance (staying put is one of the options).
"""

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from partA_common import GameIO, World, centre

# --------------------------------------------------------------------------- #
#  Parameters (tuned in the headless simulator, sim_partA.py)
# --------------------------------------------------------------------------- #
GRID = 40                      # cells per axis (cell = 0.025 of the screen)
ALPHA = 8.0                    # weight of danger in the edge cost
HARD_DANGER = 40.0             # danger value inside a predicted hit box
PREDICT_TAUS = (0.0, 0.15, 0.3, 0.45, 0.6)   # look-ahead times for the map
TAU_WEIGHT_DECAY = 0.8         # later predictions count less (per second)
MONSTER_MARGIN = 0.035         # extra clearance around a monster hit box
MONSTER_UNCERTAINTY = 0.30     # margin growth per second of look-ahead
FIRE_MARGIN = 0.03             # extra clearance around a fire hit box
FIRE_UNCERTAINTY = 0.05
SOFT_SIGMA_MONSTER = 0.06      # decay length of soft danger
SOFT_SIGMA_FIRE = 0.04
DECISION_LATENCY = 0.04        # time from snapshot to the move starting
LOS_MARGIN = 0.012             # clearance required on a straight-line jump
DWELL_AFTER_MOVE = 0.12        # the end point must stay safe this long
TARGET_MIN, TARGET_MAX = 0.04, 0.96   # keep the character centre on screen

# neighbour offsets for an 8-connected grid
_NEIGHBOURS = [(-1, 0), (1, 0), (0, -1), (0, 1),
               (-1, -1), (-1, 1), (1, -1), (1, 1)]


@dataclass
class Hazard:
    kind: str                       # "monster" | "fire"
    pos: Tuple[float, float]        # centre now
    vel: Tuple[float, float]        # estimated velocity (units / s)
    half: float                     # half-size of the hit box (char centre)
    seg_a: Optional[Tuple[float, float]] = None   # fire path end points
    seg_b: Optional[Tuple[float, float]] = None


@dataclass
class Decision:
    target: Optional[Tuple[float, float]]   # None = stay
    mode: str                                # direct | path | escape | stay
    coin: Optional[int] = None
    expanded: int = 0
    plan_ms: float = 0.0


# --------------------------------------------------------------------------- #
#  Hazard prediction
# --------------------------------------------------------------------------- #
def predict(h: Hazard, taus: np.ndarray) -> np.ndarray:
    """Predicted centres of hazard h at look-ahead times taus -> (n, 2)."""
    taus = np.asarray(taus, dtype=float)
    p = np.array(h.pos, dtype=float)
    v = np.array(h.vel, dtype=float)

    if h.kind == "fire" and h.seg_a is not None:
        a = np.array(h.seg_a)
        b = np.array(h.seg_b)
        d = b - a
        length = float(np.hypot(d[0], d[1]))
        if length > 1e-6:
            u = d / length
            rel = p - a
            s = float(rel @ u) / length                      # 0..1 along path
            off = float(abs(rel[0] * u[1] - rel[1] * u[0]))  # distance off path
            if off < 0.03 and -0.02 <= s <= 1.02:
                s_dot = float(v @ u) / length
                x = np.mod(s + s_dot * taus, 2.0)
                x = np.where(x > 1.0, 2.0 - x, x)            # bounce
                return a[None, :] + (x * length)[:, None] * u[None, :]

    q = p[None, :] + taus[:, None] * v[None, :]
    if h.kind == "monster":
        q = np.clip(q, 0.075, 1.04)
    return q


def margin(h: Hazard, tau):
    if h.kind == "monster":
        return MONSTER_MARGIN + MONSTER_UNCERTAINTY * tau
    return FIRE_MARGIN + FIRE_UNCERTAINTY * tau


# --------------------------------------------------------------------------- #
#  The agent
# --------------------------------------------------------------------------- #
class AStarAgent:
    name = "astar"

    def __init__(self, grid: int = GRID):
        self.grid = grid
        self.cell = 1.0 / grid
        g = (np.arange(grid) + 0.5) / grid
        self.gx, self.gy = np.meshgrid(g, g)       # [iy, ix]
        self.prev_world: Optional[World] = None
        self.last_decision: Optional[Decision] = None
        self.decisions = 0
        self.plan_ms_total = 0.0
        self.mode_counts: Dict[str, int] = {}
        # precomputed neighbour table on the flattened grid
        self._nbrs = []
        for iy in range(grid):
            for ix in range(grid):
                lst = []
                for dx, dy in _NEIGHBOURS:
                    jx, jy = ix + dx, iy + dy
                    if 0 <= jx < grid and 0 <= jy < grid:
                        lst.append((jy * grid + jx, math.hypot(dx, dy) * self.cell))
                self._nbrs.append(lst)

    # ---------------- sensing ---------------- #
    def hazards_from(self, world: World) -> List[Hazard]:
        prev = self.prev_world
        dt = (world.t - prev.t) if prev is not None else 0.0
        char_w, char_h = world.char_size
        hz: List[Hazard] = []

        for i, (mpos, msize) in enumerate(zip(world.monsters, world.monster_sizes)):
            c = centre(mpos, msize)
            vel = (0.0, 0.0)
            if prev is not None and dt > 1e-3 and i < len(prev.monsters):
                pc = centre(prev.monsters[i], prev.monster_sizes[i])
                vel = ((c[0] - pc[0]) / dt, (c[1] - pc[1]) / dt)
            hz.append(Hazard("monster", c, vel, half=char_w / 2.0))

        for i, f in enumerate(world.fires):
            c = centre(f.pos, f.size)
            vel = (0.0, 0.0)
            if prev is not None and dt > 1e-3 and i < len(prev.fires):
                pf = prev.fires[i]
                pc = centre(pf.pos, pf.size)
                vel = ((c[0] - pc[0]) / dt, (c[1] - pc[1]) / dt)
            a = centre(f.seg_start, f.size)
            b = centre(f.seg_end, f.size)
            hz.append(Hazard("fire", c, vel, half=char_w / 3.0, seg_a=a, seg_b=b))
        return hz

    # ---------------- danger map ---------------- #
    def danger_map(self, hazards: List[Hazard]) -> np.ndarray:
        danger = np.zeros_like(self.gx)
        taus = np.array(PREDICT_TAUS)
        for h in hazards:
            preds = predict(h, taus + DECISION_LATENCY)
            sigma = SOFT_SIGMA_MONSTER if h.kind == "monster" else SOFT_SIGMA_FIRE
            for tau, (px, py) in zip(taus, preds):
                w = math.exp(-TAU_WEIGHT_DECAY * tau)
                r = h.half + margin(h, tau)
                d = np.maximum(np.abs(self.gx - px), np.abs(self.gy - py))
                val = np.where(d <= r, HARD_DANGER * w,
                               w * np.exp(-(d - r) / sigma))
                np.maximum(danger, val, out=danger)
        return danger

    # ---------------- A* search ---------------- #
    def cell_of(self, p):
        ix = min(self.grid - 1, max(0, int(p[0] * self.grid)))
        iy = min(self.grid - 1, max(0, int(p[1] * self.grid)))
        return iy * self.grid + ix

    def cell_centre(self, idx):
        iy, ix = divmod(idx, self.grid)
        return ((ix + 0.5) * self.cell, (iy + 0.5) * self.cell)

    def astar(self, start: int, goals: Dict[int, int], cost_mult: np.ndarray):
        """Multi-goal A*. Returns (path cells, coin index, expanded) or
        (None, None, expanded) if no goal is reachable."""
        flat = cost_mult.ravel().tolist()
        goal_pts = [self.cell_centre(gc) for gc in goals]

        def h(idx):
            x, y = self.cell_centre(idx)
            return min(math.hypot(x - gx, y - gy) for gx, gy in goal_pts)

        g_cost = {start: 0.0}
        parent = {start: None}
        open_heap = [(h(start), 0.0, start)]
        closed = set()
        expanded = 0
        while open_heap:
            f, g, u = heapq.heappop(open_heap)
            if u in closed:
                continue
            closed.add(u)
            expanded += 1
            if u in goals:
                path = []
                while u is not None:
                    path.append(u)
                    u = parent[u]
                path.reverse()
                return path, goals[path[-1]], expanded
            cu = flat[u]
            for v, step in self._nbrs[u]:
                if v in closed:
                    continue
                ng = g + step * (cu + flat[v]) * 0.5
                if ng < g_cost.get(v, 1e18):
                    g_cost[v] = ng
                    parent[v] = u
                    heapq.heappush(open_heap, (ng + h(v), ng, v))
        return None, None, expanded

    # ---------------- space-time safety ---------------- #
    def clearance(self, p0, p1, hazards, move_time):
        """Smallest clearance (beyond hit box + margin) along a straight move
        p0 -> p1 that starts after DECISION_LATENCY and lasts move_time, plus
        a short dwell at p1. Negative = predicted hit."""
        dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        n = max(6, int(math.ceil(dist / 0.02)))
        s = np.linspace(0.0, 1.0, n + 1)
        pts = np.array(p0)[None, :] + s[:, None] * (np.array(p1) - np.array(p0))[None, :]
        taus = DECISION_LATENCY + s * move_time
        # dwell at the end point
        dwell_t = DECISION_LATENCY + move_time + np.array([0.05, DWELL_AFTER_MOVE])
        pts = np.vstack([pts, np.repeat(pts[-1:], len(dwell_t), axis=0)])
        taus = np.concatenate([taus, dwell_t])
        best = 1e9
        for h in hazards:
            q = predict(h, taus)
            d = np.maximum(np.abs(pts[:, 0] - q[:, 0]), np.abs(pts[:, 1] - q[:, 1]))
            m = h.half + margin(h, taus) * 0.5 + LOS_MARGIN
            best = min(best, float(np.min(d - m)))
        return best

    # ---------------- decision ---------------- #
    def decide(self, world: World) -> Decision:
        t0 = time.perf_counter()
        hazards = self.hazards_from(world)
        char_c = centre(world.char_pos, world.char_size)
        coins_c = [centre(c, world.coin_size) for c in world.coins]
        move_time = world.char_anim_duration
        if not coins_c:
            return Decision(None, "stay")

        danger = self.danger_map(hazards)
        cost_mult = 1.0 + ALPHA * danger

        goals: Dict[int, int] = {}
        for k, c in enumerate(coins_c):
            goals.setdefault(self.cell_of(c), k)
        start = self.cell_of(char_c)
        path, coin_idx, expanded = self.astar(start, goals, cost_mult)

        decision = None
        if path is not None:
            coin_c = coins_c[coin_idx]
            # 1) the coin itself, if the straight jump is safe
            if self.clearance(char_c, coin_c, hazards, move_time) > 0:
                decision = Decision(self._clamp(coin_c), "direct", coin_idx)
            else:
                # 2) farthest safe point along the A* path
                for idx in reversed(path[1:]):
                    p = self.cell_centre(idx)
                    if self.clearance(char_c, p, hazards, move_time) > 0:
                        decision = Decision(self._clamp(p), "path", coin_idx)
                        break

        if decision is None:
            decision = self.escape(char_c, coins_c, hazards, move_time,
                                   path, coin_idx)

        decision.expanded = expanded
        decision.plan_ms = (time.perf_counter() - t0) * 1000.0
        self.prev_world = world
        self.last_decision = decision
        self.decisions += 1
        self.plan_ms_total += decision.plan_ms
        self.mode_counts[decision.mode] = self.mode_counts.get(decision.mode, 0) + 1
        return decision

    def escape(self, char_c, coins_c, hazards, move_time, path, coin_idx):
        """No safe progress: pick the candidate with the best clearance;
        among safe candidates, the one closest to the next path step."""
        goal = self.cell_centre(path[1]) if path and len(path) > 1 else \
            min(coins_c, key=lambda c: math.hypot(c[0] - char_c[0], c[1] - char_c[1]))
        cands = [None]   # None = stay where we are
        for r in (0.08, 0.16, 0.28, 0.42):
            for k in range(16):
                a = 2 * math.pi * k / 16
                p = (char_c[0] + r * math.cos(a), char_c[1] + r * math.sin(a))
                if TARGET_MIN <= p[0] <= TARGET_MAX and TARGET_MIN <= p[1] <= TARGET_MAX:
                    cands.append(p)
        best, best_key = None, None
        for p in cands:
            q = char_c if p is None else p
            clr = self.clearance(char_c, q, hazards, move_time)
            progress = math.hypot(q[0] - goal[0], q[1] - goal[1])
            key = (1, -progress) if clr > 0 else (0, clr)
            if best_key is None or key > best_key:
                best, best_key = p, key
        if best is None:
            return Decision(None, "stay", coin_idx)
        return Decision(self._clamp(best), "escape", coin_idx)

    @staticmethod
    def _clamp(p):
        return (min(TARGET_MAX, max(TARGET_MIN, p[0])),
                min(TARGET_MAX, max(TARGET_MIN, p[1])))

    # ---------------- control loop ---------------- #
    def run(self, io: GameIO):
        while True:
            world = io.snapshot()
            if world.killed or not world.coins:
                return
            d = self.decide(world)
            if d.target is not None:
                io.move(d.target)
            io.sleep(world.char_anim_duration)
