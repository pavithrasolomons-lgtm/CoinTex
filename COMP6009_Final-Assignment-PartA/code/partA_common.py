"""
partA_common.py
===============
Group:            Colombo-group-08, COMP6009 Final Assignment Part A

Shared data types for the COMP6009 Final Assignment (Part A) agents.

Every agent (greedy, GA, A*) talks to the game through the same small
interface, GameIO, so the three algorithms are compared under identical
conditions:

    io.snapshot() -> World     read the current positions
    io.move(target)            same as app.start_char_animation(lvl, target)
    io.sleep(seconds)          wait (the real game sleeps; the simulator
                               advances its clock)

All positions are normalised layout coordinates (0..1), the unit Kivy's
pos_hint uses. pos_hint is a widget's BOTTOM-LEFT corner, so a centre is
corner + size / 2. Like start_char_animation(), io.move() treats its target
as the point where the character's CENTRE should end up.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

Vec = Tuple[float, float]


class EpisodeOver(Exception):
    """Raised by GameIO when the level has ended (success, death, timeout or
    stop request), so the agent's loop unwinds cleanly."""


@dataclass
class FireInfo:
    pos: Vec               # current pos_hint (bottom-left corner)
    size: Vec              # size_hint
    seg_start: Vec         # fire_start_pos_hint (corner)
    seg_end: Vec           # fire_end_pos_hint (corner)
    leg_duration: float    # fire_anim_duration (seconds per leg)


@dataclass
class World:
    t: float                       # seconds since the level started
    char_pos: Vec                  # pos_hint corner of the character
    char_size: Vec
    coins: List[Vec]               # pos_hint corners, in coins_ids order
    coin_size: Vec
    monsters: List[Vec]            # pos_hint corners
    monster_sizes: List[Vec]
    fires: List[FireInfo] = field(default_factory=list)
    char_anim_duration: float = 0.25
    killed: bool = False


def centre(pos: Vec, size: Vec) -> Vec:
    return (pos[0] + size[0] / 2.0, pos[1] + size[1] / 2.0)


class GameIO:
    """Interface implemented by the real Kivy runner and by the simulator."""

    def snapshot(self) -> World:
        raise NotImplementedError

    def move(self, target: Vec) -> None:
        raise NotImplementedError

    def sleep(self, seconds: float) -> None:
        raise NotImplementedError
