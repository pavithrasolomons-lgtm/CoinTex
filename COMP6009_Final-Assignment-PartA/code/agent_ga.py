"""
agent_ga.py
===========
Author:           Pavithra Solomons
Last modified by: Pavithra Solomons
Last modified:    29 October 2026
Group:            Colombo-group-08, COMP6009 Final Assignment Part A
Based on:         PlayerGA/TutorialProject/main.py from CoinTex by Ahmed Fawzy Gad

The genetic-algorithm agent that ships with CoinTex
(PlayerGA/TutorialProject/main.py), used as the GA benchmark.

The search settings and the fitness function are the tutorial's own. Only the
PyGAD calls were ported, because the tutorial was written for PyGAD 2.x and
does not start on PyGAD 3.x (the version requirements.txt installs):

  * fitness_func(solution, idx)      -> fitness_func(ga_instance, solution, idx)
    (PyGAD >= 2.20 requires three arguments)
  * callback_generation=...          -> on_generation=...
    (callback_generation was removed in PyGAD 3.0)
  * delay_after_gen=char_anim_dur    -> io.sleep(char_anim_dur) at the end of
    on_generation (delay_after_gen was removed in PyGAD 3.0)
  * keep_elitism=0, keep_parents=0   PyGAD 3 reuses the cached fitness of
    elites and parents; PyGAD 2.x re-scored every solution each generation.
    In a moving game the cached scores go stale, the best fitness stops
    changing and the tutorial's "move only when fitness changes" rule freezes
    the character. Turning the cache off restores the tutorial's behaviour.

The tutorial read the widgets inside every fitness call; here the positions are
read once per generation (one generation lasts ~0.25 s, so this changes nothing
of substance and keeps 1000 fitness calls from touching Kivy widgets).
"""

import logging

import pygad

from partA_common import EpisodeOver, GameIO

# PyGAD logs any exception raised inside a callback before re-raising it. The
# harness ends an episode by raising EpisodeOver, so keep that out of the log.
_QUIET = logging.getLogger("partA.pygad")
_QUIET.addHandler(logging.NullHandler())
_QUIET.setLevel(logging.CRITICAL + 1)
_QUIET.propagate = False


class GAAgent:
    name = "ga"

    def __init__(self):
        self.io = None
        self.world = None
        self.last_fitness = 0
        self.decisions = 0

    # ---- tutorial fitness, unchanged except for the data source ---------- #
    def fitness_func(self, ga_instance, solution, solution_idx):
        world = self.world

        coins = world.coins
        if len(coins) == 0:
            return 0

        curr_coin_center = [coins[0][0], coins[0][1]]   # first coin in coins_ids

        output = abs(solution[0] - curr_coin_center[0]) + abs(solution[1] - curr_coin_center[1])
        output = 1.0 / output

        monsters_pos = [[m[0], m[1]] for m in world.monsters]

        for monst_pos in monsters_pos:
            char_monst_h_distance = abs(solution[0] - monst_pos[0])
            char_monst_v_distance = abs(solution[1] - monst_pos[1])
            if char_monst_h_distance <= 0.3 and char_monst_v_distance <= 0.3:
                output -= 300
            else:
                output += 100

        fires_pos = [[f.pos[0], f.pos[1]] for f in world.fires]

        for fire_pos in fires_pos:
            char_fire_h_distance = abs(solution[0] - fire_pos[0])
            char_fire_v_distance = abs(solution[1] - fire_pos[1])
            if char_fire_h_distance <= 0.3 and char_fire_v_distance <= 0.3:
                output -= 300
            else:
                output += 100

        fitness = output
        return fitness

    # ---- tutorial callback_generation, ported to on_generation ------------ #
    def on_generation(self, ga_instance):
        self.decisions += 1            # harness counter: generations
        best_sol_fitness = ga_instance.best_solution()[1]
        fitness_change = best_sol_fitness - self.last_fitness

        self.last_fitness = best_sol_fitness

        coins = self.world.coins

        if len(coins) == 0 or self.world.killed:
            return "stop"
        elif len(coins) != 0 and fitness_change != 0:
            best_sol = ga_instance.best_solution()[0]
            self.io.move([float(best_sol[0]), float(best_sol[1])])

        # was delay_after_gen=self.screen.char_anim_duration
        self.io.sleep(self.world.char_anim_duration)
        self.world = self.io.snapshot()

    def run(self, io: GameIO):
        self.io = io
        self.world = io.snapshot()
        ga_instance = pygad.GA(num_generations=9999,
                               num_parents_mating=300,
                               fitness_func=self.fitness_func,
                               sol_per_pop=1000,
                               num_genes=2,
                               init_range_low=0.0,
                               init_range_high=1.0,
                               random_mutation_min_val=0.0,
                               random_mutation_max_val=1.0,
                               mutation_by_replacement=True,
                               # port: PyGAD 3 reuses the cached fitness of
                               # elites/parents; the tutorial's PyGAD 2.x
                               # re-scored every solution each generation.
                               # Cached scores go stale in a moving game and
                               # freeze the agent, so caching is switched off.
                               keep_elitism=0,
                               keep_parents=0,
                               on_generation=self.on_generation,
                               logger=_QUIET,
                               suppress_warnings=True)
        try:
            ga_instance.run()
        except EpisodeOver:
            raise
