"""
ga_agent.py
===========
The genetic-algorithm "brain" of the CoinTex agent, kept deliberately free of any
Kivy imports so that exactly the same code can be driven by

  * main_improved.py  -- the real game (Kivy GUI), and
  * ga_sim.py         -- a headless approximation of the game used for
                         parameter sweeps and KPI collection.

Everything works in *normalised screen coordinates*: the playing field is the
unit square, x to the right, y upwards.  A chromosome is a 2-gene real vector

        solution = [x, y]

which is interpreted as the **centre** of the square the agent should walk to
next.  (The game's start_char_animation() subtracts half the character size from
the value it is given, so the value it expects really is a centre, not a
bottom-left corner.  The original code compared it against raw pos_hint values,
which are bottom-left corners -- a systematic half-widget offset that this
module fixes.)

COMP6009 Lab Task 2.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pygad


# --------------------------------------------------------------------------- #
#  World snapshot
# --------------------------------------------------------------------------- #
@dataclass
class World:
    """An immutable picture of the game at one instant.

    All positions are the *centres* of the corresponding widgets, in normalised
    screen coordinates.
    """

    char: Tuple[float, float]                       # agent centre
    coins: List[Tuple[float, float]] = field(default_factory=list)
    monsters: List[Tuple[float, float]] = field(default_factory=list)
    fires: List[Tuple[float, float]] = field(default_factory=list)
    # optional: estimated velocity of each monster (units per decision step),
    # used by the predictive danger term when available.
    monster_vel: List[Tuple[float, float]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
@dataclass
class AgentConfig:
    """Every knob the report experiments with lives here."""

    name: str = "improved"

    # ---- which fitness function to use -----------------------------------
    fitness: str = "improved"          # "baseline" | "improved"

    # ---- fitness weights (improved only) ---------------------------------
    w_near: float = 3.0                # pull towards the nearest coin
    w_sweep: float = 2.5               # reward coins swept by the *path*
    sigma_sweep: float = 0.07          # kernel width of the sweep term
    w_monster: float = 6.0             # monster avoidance strength
    r_monster: float = 0.22            # monster danger radius
    w_fire: float = 6.0                # fire avoidance strength
    r_fire: float = 0.18               # fire danger radius
    w_len: float = 2.0                 # penalty on over-long jumps
    max_jump: float = 0.45             # jump length above which w_len bites
    w_oob: float = 20.0                # out-of-bounds penalty
    predict_monsters: bool = True      # aim at where monsters will be

    # ---- GA operators ----------------------------------------------------
    sol_per_pop: int = 20
    num_parents_mating: int = 8
    num_generations: int = 25          # generations *per decision*
    parent_selection_type: str = "tournament"
    K_tournament: int = 3
    crossover_type: str = "uniform"
    crossover_probability: Optional[float] = 0.9
    mutation_type: str = "adaptive"    # "adaptive" | "random" | "swap" | None
    mutation_probability: Optional[Sequence[float]] = (0.6, 0.15)
    mutation_percent_genes: float = 50
    random_mutation_min_val: float = -0.25
    random_mutation_max_val: float = 0.25
    mutation_by_replacement: bool = False
    keep_elitism: int = 2

    # ---- search space / seeding -----------------------------------------
    # The GA must search the set of points the character can physically stand
    # on, which is the unit square shrunk by half the character size -- not the
    # unit square itself.  main_improved.py overwrites these at runtime from the
    # real character size_hint; the defaults match ga_sim's assumed 0.08 x 0.12
    # character.  Getting this wrong makes coins near the screen edge literally
    # untargetable (CoinTex places coins partly off-screen once num_coins > 20,
    # because section_width drops below the coin width).
    bounds_x: Tuple[float, float] = (0.04, 0.96)
    bounds_y: Tuple[float, float] = (0.06, 0.94)
    warm_start: bool = True            # re-seed with last generation's best
    informed_init: bool = True         # seed a few individuals on coins

    # ---- control policy --------------------------------------------------
    # "always": issue a move after every GA burst (improved behaviour)
    # "on_change": only move when the best fitness changed (original behaviour)
    move_policy: str = "always"


# The original main_updated.py, expressed in the same language, so that the
# report's "baseline" column is the real baseline and not a strawman.
BASELINE_CONFIG = AgentConfig(
    name="baseline",
    fitness="baseline",
    sol_per_pop=10,
    num_parents_mating=2,
    num_generations=1,                 # the original runs one long GA; the
                                       # controller re-enters it continuously
    parent_selection_type="random",
    K_tournament=3,
    crossover_type="single_point",
    crossover_probability=None,
    mutation_type="swap",
    mutation_probability=None,
    mutation_percent_genes=10,
    random_mutation_min_val=1.0,
    random_mutation_max_val=10.0,
    mutation_by_replacement=True,
    keep_elitism=0,
    bounds_x=(0.3, 1.3),
    bounds_y=(0.3, 1.3),
    warm_start=False,
    informed_init=False,
    predict_monsters=False,
    move_policy="on_change",
)

IMPROVED_CONFIG = AgentConfig(name="improved")


# --------------------------------------------------------------------------- #
#  Geometry helpers
# --------------------------------------------------------------------------- #
def point_segment_distance(px: float, py: float,
                           ax: float, ay: float,
                           bx: float, by: float) -> float:
    """Shortest distance from point P to the line segment AB.

    The agent travels from A (current position) to B (the chromosome) in a
    straight line, so the segment -- not the end point -- is what actually
    sweeps up coins and what actually runs into monsters.
    """
    vx, vy = bx - ax, by - ay
    len_sq = vx * vx + vy * vy
    if len_sq < 1e-12:
        t = 0.0
    else:
        t = ((px - ax) * vx + (py - ay) * vy) / len_sq
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    dx = ax + t * vx - px
    dy = ay + t * vy - py
    return math.hypot(dx, dy)


# --------------------------------------------------------------------------- #
#  Fitness functions
# --------------------------------------------------------------------------- #
def baseline_fitness(solution, world: World, cfg: AgentConfig) -> float:
    """Faithful re-implementation of the fitness shipped with main_updated.py.

    Reciprocal of the Manhattan distance to the nearest coin.  The monster and
    fire loops in the original add or subtract *zero*, so hazards have no
    influence whatsoever -- that is reproduced here by simply omitting them.
    """
    if not world.coins:
        return 0.0
    best = 1000.0
    for (cx, cy) in world.coins:
        d = abs(solution[0] - cx) + abs(solution[1] - cy)
        if d < best:
            best = d
    if best <= 1e-9:                    # guard the division the original lacks
        return 1e9
    return 1.0 / best


def improved_fitness(solution, world: World, cfg: AgentConfig) -> float:
    """Smooth, path-aware, hazard-aware fitness.

        f(s) = w_near * (1 - d_nearest)                      # get to a coin
             + w_sweep * SUM_i exp(-(seg_dist_i / sigma)^2)  # sweep coins en route
             - SUM_h w_h * (1 - d_h / r_h)^2   for d_h < r_h # avoid hazards
             - w_len * max(0, |s - p| - max_jump)^2          # avoid tunnelling
             - w_oob * out_of_bounds^2                       # stay on screen

    Every term is continuous and bounded, so the landscape has a usable
    gradient everywhere instead of the single near-singular spike produced by
    1/distance.
    """
    if not world.coins:
        return 0.0

    sx, sy = float(solution[0]), float(solution[1])
    px, py = world.char

    # --- out of bounds ----------------------------------------------------
    oob = 0.0
    for v, (lo, hi) in ((sx, cfg.bounds_x), (sy, cfg.bounds_y)):
        if v < lo:
            oob += (lo - v) ** 2
        elif v > hi:
            oob += (v - hi) ** 2

    # --- attraction to the nearest coin -----------------------------------
    d_near = min(math.hypot(sx - cx, sy - cy) for (cx, cy) in world.coins)
    score = cfg.w_near * (1.0 - d_near)

    # --- reward every coin the straight-line path sweeps ------------------
    if cfg.w_sweep > 0.0:
        inv_sigma = 1.0 / cfg.sigma_sweep
        sweep = 0.0
        for (cx, cy) in world.coins:
            d = point_segment_distance(cx, cy, px, py, sx, sy) * inv_sigma
            if d < 3.0:                          # exp(-9) is negligible
                sweep += math.exp(-d * d)
        score += cfg.w_sweep * sweep

    # --- hazards: penalise proximity to the whole path --------------------
    if cfg.predict_monsters and world.monster_vel:
        monsters = [(mx + vx, my + vy)
                    for (mx, my), (vx, vy) in zip(world.monsters, world.monster_vel)]
    else:
        monsters = world.monsters

    for (mx, my) in monsters:
        d = point_segment_distance(mx, my, px, py, sx, sy)
        if d < cfg.r_monster:
            t = 1.0 - d / cfg.r_monster
            score -= cfg.w_monster * t * t

    for (fx, fy) in world.fires:
        d = point_segment_distance(fx, fy, px, py, sx, sy)
        if d < cfg.r_fire:
            t = 1.0 - d / cfg.r_fire
            score -= cfg.w_fire * t * t

    # --- discourage jumps so long the animation tunnels past coins --------
    jump = math.hypot(sx - px, sy - py)
    if jump > cfg.max_jump:
        score -= cfg.w_len * (jump - cfg.max_jump) ** 2

    return score - cfg.w_oob * oob


def evaluate(solution, world: World, cfg: AgentConfig) -> float:
    if cfg.fitness == "baseline":
        return baseline_fitness(solution, world, cfg)
    return improved_fitness(solution, world, cfg)


# --------------------------------------------------------------------------- #
#  Population seeding
# --------------------------------------------------------------------------- #
def _initial_population(world: World, cfg: AgentConfig,
                        previous_best: Optional[Sequence[float]],
                        rng: random.Random) -> np.ndarray:
    xlo, xhi = cfg.bounds_x
    ylo, yhi = cfg.bounds_y
    pop = np.empty((cfg.sol_per_pop, 2), dtype=float)

    idx = 0
    if cfg.warm_start and previous_best is not None and idx < cfg.sol_per_pop:
        pop[idx] = [float(previous_best[0]), float(previous_best[1])]
        idx += 1

    if cfg.informed_init and world.coins:
        # Seed a handful of individuals on the coins nearest the agent.  This is
        # domain knowledge injected as *initialisation only* -- the GA is still
        # free to move away from those points, and the hazard terms routinely
        # make it do so.
        px, py = world.char
        nearest = sorted(world.coins,
                         key=lambda c: (c[0] - px) ** 2 + (c[1] - py) ** 2)
        for (cx, cy) in nearest[:3]:
            if idx >= cfg.sol_per_pop:
                break
            pop[idx] = [min(max(cx, xlo), xhi), min(max(cy, ylo), yhi)]
            idx += 1

    while idx < cfg.sol_per_pop:
        pop[idx] = [rng.uniform(xlo, xhi), rng.uniform(ylo, yhi)]
        idx += 1

    return pop


# --------------------------------------------------------------------------- #
#  One GA decision
# --------------------------------------------------------------------------- #
class Decision:
    """Result of a single GA burst."""

    __slots__ = ("target", "fitness", "generations")

    def __init__(self, target, fitness, generations):
        self.target = target
        self.fitness = fitness
        self.generations = generations


def decide(world: World, cfg: AgentConfig,
           previous_best: Optional[Sequence[float]] = None,
           rng: Optional[random.Random] = None) -> Decision:
    """Run a short GA against a *frozen* snapshot of the world and return the
    best target found.

    Running many short GAs rather than one 10 000-generation GA is the single
    biggest architectural change relative to main_updated.py: the game is
    non-stationary (coins disappear, monsters move), so a solution optimised
    against the state of the world 200 generations ago is optimising a problem
    that no longer exists.
    """
    rng = rng or random

    if not world.coins:
        return Decision(world.char, 0.0, 0)

    init_pop = _initial_population(world, cfg, previous_best, rng)

    def _fitness(ga_instance, solution, solution_idx):
        return evaluate(solution, world, cfg)

    kwargs = dict(
        num_generations=max(1, cfg.num_generations),
        num_parents_mating=max(2, min(cfg.num_parents_mating, cfg.sol_per_pop)),
        fitness_func=_fitness,
        initial_population=init_pop,
        parent_selection_type=cfg.parent_selection_type,
        crossover_type=cfg.crossover_type,
        mutation_type=cfg.mutation_type,
        keep_elitism=cfg.keep_elitism,
        suppress_warnings=True,
        save_solutions=False,
        save_best_solutions=False,
    )
    if cfg.parent_selection_type == "tournament":
        kwargs["K_tournament"] = cfg.K_tournament
    if cfg.crossover_probability is not None:
        kwargs["crossover_probability"] = cfg.crossover_probability

    if cfg.mutation_type in ("random", "adaptive"):
        kwargs["random_mutation_min_val"] = cfg.random_mutation_min_val
        kwargs["random_mutation_max_val"] = cfg.random_mutation_max_val
        kwargs["mutation_by_replacement"] = cfg.mutation_by_replacement
        if cfg.mutation_probability is not None:
            kwargs["mutation_probability"] = list(cfg.mutation_probability) \
                if cfg.mutation_type == "adaptive" else cfg.mutation_probability
        else:
            kwargs["mutation_percent_genes"] = cfg.mutation_percent_genes
        # keep the search inside the playing field
        if cfg.fitness != "baseline":
            kwargs["gene_space"] = [
                {"low": cfg.bounds_x[0], "high": cfg.bounds_x[1]},
                {"low": cfg.bounds_y[0], "high": cfg.bounds_y[1]},
            ]
    elif cfg.mutation_type == "swap":
        kwargs["mutation_percent_genes"] = cfg.mutation_percent_genes

    ga = pygad.GA(**kwargs)
    ga.run()
    solution, fitness, _ = ga.best_solution(ga.last_generation_fitness)

    x = min(max(float(solution[0]), cfg.bounds_x[0]), cfg.bounds_x[1])
    y = min(max(float(solution[1]), cfg.bounds_y[0]), cfg.bounds_y[1])
    return Decision((x, y), float(fitness), cfg.num_generations)
