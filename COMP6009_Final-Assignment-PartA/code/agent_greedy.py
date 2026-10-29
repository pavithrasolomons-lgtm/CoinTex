"""
agent_greedy.py
===============
Group:            Colombo-group-08, COMP6009 Final Assignment Part A
Based on:         main_work_greedy.py (greedy program from the COMP6009 unit materials)

The ORIGINAL greedy agent from main_work_greedy.py, unchanged in its logic,
so it can run inside the same experiment harness as the other agents.

create_matrix(), check_coin() and findMinRoute() are copied verbatim (their
debug print() calls are switched off by default because console output slows
the agent thread and is not part of the algorithm). run() is the original
CollectCoinThread.run() with three harness-only changes:

  * positions come from io.snapshot() instead of reading widgets directly
    (same values: pos_hint bottom-left corners, as position_func_v3 used);
  * time.sleep(0.4) became io.sleep(0.4);
  * when no coins are left the loop ends cleanly (the original crashed with
    "TypeError: cannot unpack non-iterable int" at this point).
"""

from typing import DefaultDict

from partA_common import GameIO

INT_MAX = 2147483647
DEBUG_PRINTS = False
MOVE_PAUSE = 0.4          # seconds between moves, as in the original


def _debug(*args):
    if DEBUG_PRINTS:
        print(*args)


class GreedyAgent:
    name = "greedy"
    decisions = 0          # harness counter: number of route plans

    @staticmethod
    def create_matrix(start_pos, coin_pos, monsters_pos, fires_pos):  #

        coin_pos = GreedyAgent.check_coin(coin_pos, monsters_pos, fires_pos)

        tem_pos = coin_pos
        tem_pos.insert(0, start_pos)
        tem_dif_y = []
        for i in range(len(coin_pos)):
            tem_dif_x = []
            for j in range(len(coin_pos)):
                if (i == j):
                    tem_dif_x.append(-1)
                else:
                    tem_dif_x.append(abs(coin_pos[i][0] - coin_pos[j][0]) + abs(coin_pos[i][1] - coin_pos[j][1]))
            tem_dif_y.append(tem_dif_x)

        _debug("coin_pos=", tem_dif_y)
        return tem_dif_y

    @staticmethod
    def check_coin(coin_pos, monsters_pos, fires_pos):  #

        monsters_pos.append([-10, -10])
        fires_pos.append([-10, -10])

        tem_coin_pos = coin_pos
        _debug("1 tem_coin_pos", tem_coin_pos)
        p = 0.3  # parameter to adjust the closeness between coin to monster (or fire)
        for i in range(len(coin_pos)):
            for j in range(len(monsters_pos)):
                for k in range(len(fires_pos)):
                    try:
                        if ((abs(coin_pos[i][0] - monsters_pos[j][0]) + abs(coin_pos[i][1] - monsters_pos[j][1])) < p) \
                                or ((abs(coin_pos[i][0] - fires_pos[k][0]) + abs(coin_pos[i][1] - fires_pos[k][1])) < p):
                            tem_coin_pos.pop(i)
                    except:
                        None
        _debug("2 tem_coin_pos", tem_coin_pos)
        return tem_coin_pos

    @staticmethod
    def findMinRoute(tsp):
        sum = 0
        counter = 0
        j = 0
        i = 0
        min = INT_MAX
        visitedRouteList = DefaultDict(int)

        # Starting from the 0th indexed
        # city i.e., the first city
        visitedRouteList[0] = 1
        route = [0] * len(tsp)

        # Traverse the adjacency
        # matrix tsp[][]
        while i < len(tsp) and j < len(tsp[i]):

            # Corner of the Matrix
            if counter >= len(tsp[i]) - 1:
                break

            # If this path is unvisited then
            # and if the cost is less then
            # update the cost
            if j != i and (visitedRouteList[j] == 0):
                if tsp[i][j] < min:
                    min = tsp[i][j]
                    route[counter] = j + 1

            j += 1

            # Check all paths from the
            # ith indexed city
            if j == len(tsp[i]):
                sum += min
                min = INT_MAX
                visitedRouteList[route[counter] - 1] = 1
                j = 0
                i = route[counter] - 1
                counter += 1

        # Update the ending city in array
        # from city which was last visited
        i = route[counter - 1] - 1

        for j in range(len(tsp)):

            if (i != j) and tsp[i][j] < min:
                min = tsp[i][j]
                route[counter] = j + 1

        sum += min

        return route

    # ------------------------------------------------------------------ #
    @staticmethod
    def position_func_v3(world):
        """Same values as the original position_func_v3(): pos_hint corners."""
        if len(world.coins) == 0:
            return 0
        coin_pos = [[x, y] for (x, y) in world.coins]
        monsters_pos = [[x, y] for (x, y) in world.monsters]
        fires_pos = [[f.pos[0], f.pos[1]] for f in world.fires]
        return coin_pos, monsters_pos, fires_pos

    def run(self, io: GameIO):
        start_pos = [0.1, 0.1]
        io.move(start_pos)

        state = self.position_func_v3(io.snapshot())
        if state == 0:
            return
        coin_pos, monsters_pos, fires_pos = state
        while True:
            tsp = GreedyAgent.create_matrix(start_pos, coin_pos, monsters_pos, fires_pos)
            route = GreedyAgent.findMinRoute(tsp)
            self.decisions += 1

            for i in range(0, len(route) - 1):
                io.move(coin_pos[route[i] - 1])
                io.sleep(MOVE_PAUSE)

            state = self.position_func_v3(io.snapshot())
            if state == 0:          # harness fix: original crashed here
                break
            coin_pos, monsters_pos, fires_pos = state
            if len(coin_pos) <= 0:
                break
