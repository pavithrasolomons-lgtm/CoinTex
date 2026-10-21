import kivy.app
import kivy.uix.screenmanager
import kivy.uix.image
import random
import kivy.core.audio
import os
import functools
import kivy.uix.behaviors
import pickle
import pygad
import threading
import kivy.base
import time
import csv
import sys
import argparse
import datetime

# ===================================================================
# COMP6009 Lab 2 - Experiment configuration
# ===================================================================
# Run the BASELINE (original settings, broken fitness):
#     python main_improved.py --mode baseline --level 1 --runs 5
# Run the IMPROVED algorithm:
#     python main_improved.py --mode improved --level 1 --runs 5
# Play manually (no auto-restart):
#     python main_improved.py --mode improved
# Results are appended to results.csv
# ===================================================================

parser = argparse.ArgumentParser(description="CoinTex GA agent")
parser.add_argument("--mode", choices=["baseline", "improved"], default="improved",
                    help="which GA configuration to use")
parser.add_argument("--level", type=int, default=0,
                    help="level to auto-play (0 = manual level select)")
parser.add_argument("--runs", type=int, default=1,
                    help="how many times to repeat the level automatically")
parser.add_argument("--log", default="results.csv", help="CSV file for KPI logging")
ARGS, _ = parser.parse_known_args()

MODE = ARGS.mode
AUTO_LEVEL = ARGS.level
RUNS_REMAINING = ARGS.runs
LOG_FILE = ARGS.log
RUN_INDEX = 0

# --- Improved fitness weights (tune these for your experiments) -----
COIN_WEIGHT = 1.0        # strength of attraction to the nearest coin
EPSILON = 0.02           # avoids division by zero when sitting on a coin
MONSTER_PENALTY = 6.0    # how strongly to avoid monsters
MONSTER_RADIUS = 0.30    # Manhattan distance considered "dangerous"
FIRE_PENALTY = 8.0       # fires are stationary and deadly -> heavier
FIRE_RADIUS = 0.25
BOUNDS_PENALTY = 50.0    # discourage targets outside the visible board
MOVE_INTERVAL = 0.25     # seconds between issued moves (= char_anim_duration)


class CollectCoinThread(threading.Thread):

    def __init__(self, screen):
        super().__init__()
        self.screen = screen

    def run(self):
        if MODE == "baseline":
            # ---- ORIGINAL settings, kept verbatim for comparison ----
            ga_instance = pygad.GA(num_generations=10000,
                                   num_parents_mating=2,
                                   fitness_func=fitness_func,
                                   parent_selection_type="random",
                                   sol_per_pop=10,
                                   num_genes=2,
                                   init_range_low=0.3,
                                   init_range_high=1.3,
                                   random_mutation_min_val=1.0,
                                   random_mutation_max_val=10.0,
                                   mutation_by_replacement=True,
                                   mutation_type="swap",
                                   mutation_percent_genes=10,
                                   on_generation=callback_generation)
        else:
            # ---- IMPROVED settings ----
            # tournament selection    -> real selection pressure (was: random)
            # random mutation         -> can actually explore (was: swap, which
            #                            only swapped x and y on a 2-gene chromosome)
            # gene_space 0..1         -> targets always on-screen (was: 0.3..1.3)
            # uniform crossover       -> better gene mixing for 2 genes
            # keep_parents=0          -> every solution re-scored each generation,
            #                            essential because coins/monsters move
            ga_instance = pygad.GA(num_generations=100000,
                                   num_parents_mating=8,
                                   fitness_func=fitness_func,
                                   parent_selection_type="tournament",
                                   K_tournament=3,
                                   sol_per_pop=20,
                                   num_genes=2,
                                   gene_space={"low": 0.0, "high": 1.0},
                                   crossover_type="uniform",
                                   mutation_type="random",
                                   mutation_probability=0.3,
                                   random_mutation_min_val=-0.15,
                                   random_mutation_max_val=0.15,
                                   mutation_by_replacement=False,
                                   keep_parents=0,
                                   keep_elitism=0,
                                   on_generation=callback_generation)

        ga_instance.run()



def fitness_func(ga_instance, solution, solution_idx):
    """Score a candidate target position (solution[0], solution[1]).

    Improved fitness = coin attraction - hazard penalties - out-of-bounds penalty

    In BASELINE mode the hazard terms are switched off, reproducing the
    original behaviour where the monster/fire penalties were written as
    "output -= 0" and therefore had no effect at all.
    """
    curr_screen = app.root.screens[lvl_num]

    coins = curr_screen.coins_ids
    if len(coins) == 0:
        return 0

    # ---------- 1. Attraction to the nearest remaining coin ----------
    min_dist = 1000.0
    for curr_coin in list(coins.values()):
        try:
            coin_x = curr_coin.pos_hint['x']
            coin_y = curr_coin.pos_hint['y']
            dist = abs(solution[0] - coin_x) + abs(solution[1] - coin_y)
            if dist < min_dist:
                min_dist = dist
        except (KeyError, AttributeError):
            continue

    if MODE == "baseline":
        # original formulation: 1/distance, with no epsilon guard
        try:
            fitness = 1.0 / min_dist
        except ZeroDivisionError:
            fitness = 1000.0
    else:
        # epsilon guard prevents a divide-by-zero spike when a chromosome
        # lands exactly on a coin
        fitness = COIN_WEIGHT / (min_dist + EPSILON)

    if MODE == "baseline":
        # The original code computed monster/fire distances and then applied
        # "output -= 0" / "output += 0", i.e. no penalty whatsoever.
        # Reproduced here as a no-op so the baseline is a fair comparison.
        return fitness

    # ---------- 2. Penalty for being near a monster ----------
    for i in range(curr_screen.num_monsters):
        try:
            monster_image = curr_screen.ids['monster' + str(i + 1) + '_image_lvl' + str(lvl_num)]
            mx = monster_image.pos_hint['x']
            my = monster_image.pos_hint['y']
        except (KeyError, AttributeError):
            continue

        d = abs(solution[0] - mx) + abs(solution[1] - my)
        if d < MONSTER_RADIUS:
            # graded penalty: strongest when the target sits on the monster,
            # fading to zero at the edge of the danger radius. A graded term
            # gives the GA a usable gradient; a hard cut-off would not.
            fitness -= MONSTER_PENALTY * (MONSTER_RADIUS - d) / MONSTER_RADIUS

    # ---------- 3. Penalty for being near a fire ----------
    for i in range(curr_screen.num_fires):
        try:
            fire_image = curr_screen.ids['fire' + str(i + 1) + '_lvl' + str(lvl_num)]
            fx = fire_image.pos_hint['x']
            fy = fire_image.pos_hint['y']
        except (KeyError, AttributeError):
            continue

        d = abs(solution[0] - fx) + abs(solution[1] - fy)
        if d < FIRE_RADIUS:
            fitness -= FIRE_PENALTY * (FIRE_RADIUS - d) / FIRE_RADIUS

    # ---------- 4. Penalty for leaving the visible board ----------
    for gene in (solution[0], solution[1]):
        if gene < 0.0:
            fitness -= BOUNDS_PENALTY * abs(gene)
        elif gene > 1.0:
            fitness -= BOUNDS_PENALTY * (gene - 1.0)

    return fitness




last_fitness = 0
last_move_time = 0.0

def callback_generation(ga_instance):
    global last_fitness, last_move_time

    curr_screen = app.root.screens[lvl_num]
    coins = curr_screen.coins_ids

    # Stop conditions: level cleared, or the character died.
    if len(coins) == 0 or curr_screen.character_killed:
        return "stop"

    fitness_values = ga_instance.last_generation_fitness

    if MODE == "baseline":
        # ---- ORIGINAL logic, bugs included, for comparison ----
        # Note the [1:] slice: individual 0 can never be selected (off-by-one).
        max_index = max(range(len(fitness_values[1:])),
                        key=fitness_values[1:].__getitem__)
        best_sol_fitness = fitness_values[max_index + 1]
        fitness_change = best_sol_fitness - last_fitness
        last_fitness = best_sol_fitness
        if fitness_change != 0:
            best_sol = ga_instance.population[max_index + 1]
            app.start_char_animation(lvl_num, [float(best_sol[0]), float(best_sol[1])])
        return

    # ---- IMPROVED logic ----
    # Fix 1: search the whole population, not population[1:].
    max_index = max(range(len(fitness_values)), key=fitness_values.__getitem__)
    best_sol = ga_instance.population[max_index]
    last_fitness = fitness_values[max_index]

    # Fix 2: throttle. The GA completes thousands of generations per second,
    # but a move takes char_anim_duration (0.25 s). Without this gate every
    # generation cancels the previous animation and the character barely moves.
    # Issuing one move per animation period lets each move actually finish.
    now = time.time()
    if now - last_move_time < MOVE_INTERVAL:
        time.sleep(0.005)   # yield to the Kivy UI thread
        return
    last_move_time = now

    # Fix 3: clamp to the visible board before moving.
    target_x = min(max(float(best_sol[0]), 0.0), 1.0)
    target_y = min(max(float(best_sol[1]), 0.0), 1.0)

    app.start_char_animation(lvl_num, [target_x, target_y])


class CointexApp(kivy.app.App):

    def on_start(self):
        music_dir = os.getcwd()+"/music/"
        self.main_bg_music = kivy.core.audio.SoundLoader.load(music_dir+"bg_music_piano_flute.wav")
        self.main_bg_music.loop = True
        self.main_bg_music.play()

        next_level_num, congrats_displayed_once = self.read_game_info()
        self.activate_levels(next_level_num, congrats_displayed_once)

    def read_game_info(self):
        return 24, True

    def activate_levels(self, next_level_num, congrats_displayed_once):
        num_levels = len(self.root.screens[0].ids['lvls_imagebuttons'].children)

        levels_imagebuttons = self.root.screens[0].ids['lvls_imagebuttons'].children
        for i in range(num_levels-next_level_num, num_levels):
            levels_imagebuttons[i].disabled = False
            levels_imagebuttons[i].color = [1,1,1,1]

        for i in range(0, num_levels-next_level_num):
            levels_imagebuttons[i].disabled = True
            levels_imagebuttons[i].color = [1,1,1,0.5]

        if next_level_num == (num_levels+1) and congrats_displayed_once == False:
            self.root.current = "alllevelscompleted"

    def screen_on_pre_leave(self, screen_num):
        curr_screen = self.root.screens[screen_num]
        for i in range(curr_screen.num_monsters): curr_screen.ids['monster'+str(i+1)+'_image_lvl'+str(screen_num)].pos_hint = {'x': 0.8, 'y': 0.8}
        curr_screen.ids['character_image_lvl'+str(screen_num)].pos_hint = {'x': 0.0, 'y': 0.0}

        next_level_num, congrats_displayed_once = self.read_game_info()
        self.activate_levels(next_level_num, congrats_displayed_once)

    def screen_on_pre_enter(self, screen_num):
        curr_screen = self.root.screens[screen_num]
        curr_screen.character_killed = False
        curr_screen.num_coins_collected = 0
        curr_screen.ids['character_image_lvl'+str(screen_num)].im_num = curr_screen.ids['character_image_lvl'+str(screen_num)].start_im_num
        for i in range(curr_screen.num_monsters): curr_screen.ids['monster'+str(i+1)+'_image_lvl'+str(screen_num)].im_num = curr_screen.ids['monster'+str(i+1)+'_image_lvl'+str(screen_num)].start_im_num
        curr_screen.ids['num_coins_collected_lvl'+str(screen_num)].text = "Coins 0/"+str(curr_screen.num_coins)
        curr_screen.ids['level_number_lvl'+str(screen_num)].text = "Level "+str(screen_num)

        curr_screen.num_collisions_hit = 0
        remaining_life_percent_lvl_widget = curr_screen.ids['remaining_life_percent_lvl'+str(screen_num)]
        remaining_life_percent_lvl_widget.size_hint = (remaining_life_percent_lvl_widget.remaining_life_size_hint_x, remaining_life_percent_lvl_widget.size_hint[1])

        for i in range(curr_screen.num_fires): curr_screen.ids['fire'+str(i+1)+'_lvl'+str(screen_num)].pos_hint = {'x': 1.1, 'y': 1.1}

        for key, coin in curr_screen.coins_ids.items():
            curr_screen.ids['layout_lvl'+str(screen_num)].remove_widget(coin)
        curr_screen.coins_ids = {}

        coin_width = 0.05
        coin_height = 0.05

        curr_screen = self.root.screens[screen_num]

        section_width = 1.0/curr_screen.num_coins
        for k in range(curr_screen.num_coins):
            x = random.uniform(section_width*k, section_width*(k+1)-coin_width)
            y = random.uniform(0, 1-coin_height)
            coin = kivy.uix.image.Image(source="other-images/coin.png", size_hint=(coin_width, coin_height), pos_hint={'x': x, 'y': y}, allow_stretch=True)
            curr_screen.ids['layout_lvl'+str(screen_num)].add_widget(coin, index=-1)
            curr_screen.coins_ids['coin'+str(k)] = coin

    def screen_on_enter(self, screen_num):
        music_dir = os.getcwd()+"/music/"
        self.bg_music = kivy.core.audio.SoundLoader.load(music_dir+"bg_music_piano.wav")
        self.bg_music.loop = True

        self.coin_sound = kivy.core.audio.SoundLoader.load(music_dir+"coin.wav")
        self.level_completed_sound = kivy.core.audio.SoundLoader.load(music_dir+"level_completed_flaute.wav")
        self.char_death_sound = kivy.core.audio.SoundLoader.load(music_dir+"char_death_flaute.wav")

        self.bg_music.play()

        curr_screen = self.root.screens[screen_num]
        for i in range(curr_screen.num_monsters):
            monster_image = curr_screen.ids['monster'+str(i+1)+'_image_lvl'+str(screen_num)]
            new_pos = (random.uniform(0.0, 1 - monster_image.size_hint[0]/4), random.uniform(0.0, 1 - monster_image.size_hint[1]/4))
            self.start_monst_animation(monster_image=monster_image, new_pos=new_pos, anim_duration=random.uniform(monster_image.monst_anim_duration_low, monster_image.monst_anim_duration_high))

        for i in range(curr_screen.num_fires):
            fire_widget = curr_screen.ids['fire'+str(i+1)+'_lvl'+str(screen_num)]
            self.start_fire_animation(fire_widget=fire_widget, pos=(0.0, 0.5), anim_duration=5.0)

        global lvl_num, last_fitness, last_move_time, RUN_INDEX

        lvl_num = screen_num
        last_fitness = 0
        last_move_time = 0.0
        RUN_INDEX += 1

        # KPI: start the clock for this run
        curr_screen.run_start_time = time.time()
        curr_screen.result_logged = False

        collectCoinThread = CollectCoinThread(screen=curr_screen)
        collectCoinThread.start()

    def log_result(self, screen_num, outcome):
        """Append one row of KPI data to the results CSV."""
        curr_screen = self.root.screens[screen_num]

        if getattr(curr_screen, "result_logged", False):
            return
        curr_screen.result_logged = True

        elapsed = round(time.time() - getattr(curr_screen, "run_start_time", time.time()), 2)
        collisions = curr_screen.num_collisions_hit
        max_collisions = curr_screen.num_collisions_level
        life_remaining = round(100.0 * (1 - collisions / float(max_collisions)), 1)

        row = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": MODE,
            "level": screen_num,
            "run": RUN_INDEX,
            "outcome": outcome,
            "coins_collected": curr_screen.num_coins_collected,
            "coins_total": curr_screen.num_coins,
            "time_seconds": elapsed,
            "collisions": collisions,
            "life_remaining_pct": life_remaining,
        }

        write_header = not os.path.exists(LOG_FILE)
        try:
            with open(LOG_FILE, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            print("[KPI] could not write log:", e)

        print("[KPI] {mode} | level {level} | run {run} | {outcome} | "
              "coins {coins_collected}/{coins_total} | {time_seconds}s | "
              "life {life_remaining_pct}%".format(**row))

    def start_monst_animation(self, monster_image, new_pos, anim_duration):
        monst_anim = kivy.animation.Animation(pos_hint={'x': new_pos[0], 'y': new_pos[1]}, im_num=monster_image.end_im_num, duration=anim_duration)
        monst_anim.bind(on_complete=self.monst_animation_completed)
        monst_anim.start(monster_image)

    def monst_animation_completed(self, *args):
        monster_image = args[1]
        monster_image.im_num = monster_image.start_im_num

        new_pos = (random.uniform(0.0, 1 - monster_image.size_hint[0]/4), random.uniform(0.0, 1 - monster_image.size_hint[1]/4))
        self.start_monst_animation(monster_image=monster_image, new_pos= new_pos,anim_duration=random.uniform(monster_image.monst_anim_duration_low, monster_image.monst_anim_duration_high))

    def monst_pos_hint(self, monster_image):
        screen_num = int(monster_image.parent.parent.name[5:])
        curr_screen = self.root.screens[screen_num]
        character_image = curr_screen.ids['character_image_lvl'+str(screen_num)]

        character_center = character_image.center
        monster_center = monster_image.center

        gab_x = character_image.width / 2
        gab_y = character_image.height / 2
        if character_image.collide_widget(monster_image) and abs(character_center[0] - monster_center[0]) <= gab_x and abs(character_center[1] - monster_center[1]) <= gab_y:
            curr_screen.num_collisions_hit = curr_screen.num_collisions_hit + 1
            life_percent = float(curr_screen.num_collisions_hit)/float(curr_screen.num_collisions_level)

#            life_remaining_percent = 100-round(life_percent, 2)*100
#            remaining_life_percent_lvl_widget.text = str(int(life_remaining_percent))+"%"
            remaining_life_percent_lvl_widget=curr_screen.ids['remaining_life_percent_lvl'+str(screen_num)]
            remaining_life_size_hint_x = remaining_life_percent_lvl_widget.remaining_life_size_hint_x
            remaining_life_percent_lvl_widget.size_hint =  (remaining_life_size_hint_x-remaining_life_size_hint_x*life_percent, remaining_life_percent_lvl_widget.size_hint[1])

            if curr_screen.num_collisions_hit == curr_screen.num_collisions_level:
                self.log_result(screen_num, "LOSS")
                self.bg_music.stop()
                self.char_death_sound.play()
                curr_screen.character_killed = True

                kivy.animation.Animation.cancel_all(character_image)
                for i in range(curr_screen.num_monsters): kivy.animation.Animation.cancel_all(curr_screen.ids['monster'+str(i+1)+'_image_lvl'+str(screen_num)])
                for i in range(curr_screen.num_fires): kivy.animation.Animation.cancel_all(curr_screen.ids['fire'+str(i+1)+'_lvl'+str(screen_num)])

                character_image.im_num = character_image.dead_start_im_num
                char_anim = kivy.animation.Animation(im_num=character_image.dead_end_im_num, duration=1.0)
                char_anim.start(character_image)
                kivy.clock.Clock.schedule_once(functools.partial(self.back_to_main_screen, curr_screen.parent), 3)

    def change_monst_im(self, monster_image):
        monster_image.source = "monsters-images/" + str(int(monster_image.im_num)) + ".png"

    def touch_down_handler(self, screen_num, args):
        curr_screen = self.root.screens[screen_num]
        if curr_screen.character_killed == False:
            self.start_char_animation(screen_num, args[1].spos)

    def start_char_animation(self, screen_num, touch_pos):
        curr_screen = self.root.screens[screen_num]
        if curr_screen.character_killed:
            return

        character_image = curr_screen.ids['character_image_lvl'+str(screen_num)]
        character_image.im_num = character_image.start_im_num
        char_anim = kivy.animation.Animation(pos_hint={'x': touch_pos[0] - character_image.size_hint[0] / 2,'y': touch_pos[1] - character_image.size_hint[1] / 2}, im_num=character_image.end_im_num, duration=curr_screen.char_anim_duration)
        char_anim.bind(on_complete=self.char_animation_completed)
        char_anim.start(character_image)

    def char_animation_completed(self, *args):
        character_image = args[1]
        character_image.im_num = character_image.start_im_num

    def char_pos_hint(self, character_image):
        screen_num = int(character_image.parent.parent.name[5:])
        character_center = character_image.center

        gab_x = character_image.width / 3
        gab_y = character_image.height / 3
        coins_to_delete = []
        curr_screen = self.root.screens[screen_num]

        for coin_key, curr_coin in curr_screen.coins_ids.items():
            curr_coin_center = curr_coin.center
            if character_image.collide_widget(curr_coin) and abs(character_center[0] - curr_coin_center[0]) <= gab_x and abs(character_center[1] - curr_coin_center[1]) <= gab_y:
                self.coin_sound.play()
                coins_to_delete.append(coin_key)
                curr_screen.ids['layout_lvl'+str(screen_num)].remove_widget(curr_coin)
                curr_screen.num_coins_collected = curr_screen.num_coins_collected + 1
                curr_screen.ids['num_coins_collected_lvl'+str(screen_num)].text = "Coins "+str(curr_screen.num_coins_collected)+"/"+str(curr_screen.num_coins)
                if curr_screen.num_coins_collected == curr_screen.num_coins:
                    self.log_result(screen_num, "WIN")
                    self.bg_music.stop()
                    self.level_completed_sound.play()
                    kivy.clock.Clock.schedule_once(functools.partial(self.back_to_main_screen, curr_screen.parent), 3)
                    for i in range(curr_screen.num_monsters): kivy.animation.Animation.cancel_all(curr_screen.ids['monster'+str(i+1)+'_image_lvl'+str(screen_num)])
                    for i in range(curr_screen.num_fires): kivy.animation.Animation.cancel_all(curr_screen.ids['fire'+str(i+1)+'_lvl'+str(screen_num)])

                    next_level_num, congrats_displayed_once = self.read_game_info()
                    if (screen_num+1) > next_level_num:
                        game_info_file = open("game_info",'wb')
                        pickle.dump([{'lastlvl':screen_num+1, "congrats_displayed_once": False}], game_info_file)
                        game_info_file.close()
                    else:
                        game_info_file = open("game_info",'wb')
                        pickle.dump([{'lastlvl':next_level_num, "congrats_displayed_once": True}], game_info_file)
                        game_info_file.close()

        if len(coins_to_delete) > 0:
            for coin_key in coins_to_delete:
                del curr_screen.coins_ids[coin_key]

    def change_char_im(self, character_image):
        character_image.source = "character-images/" + str(int(character_image.im_num)) + ".png"

    def start_fire_animation(self, fire_widget, pos, anim_duration):
        fire_anim = kivy.animation.Animation(pos_hint=fire_widget.fire_start_pos_hint, duration=fire_widget.fire_anim_duration)+kivy.animation.Animation(pos_hint=fire_widget.fire_end_pos_hint, duration=fire_widget.fire_anim_duration)
        fire_anim.repeat = True
        fire_anim.start(fire_widget)

    def fire_pos_hint(self, fire_widget):
        screen_num = int(fire_widget.parent.parent.name[5:])
        curr_screen = self.root.screens[screen_num]
        character_image = curr_screen.ids['character_image_lvl'+str(screen_num)]

        character_center = character_image.center
        fire_center = fire_widget.center

        gab_x = character_image.width / 3
        gab_y = character_image.height / 3
        if character_image.collide_widget(fire_widget) and abs(character_center[0] - fire_center[0]) <= gab_x and abs(character_center[1] - fire_center[1]) <= gab_y:
            curr_screen.num_collisions_hit = curr_screen.num_collisions_hit + 1
            life_percent = float(curr_screen.num_collisions_hit)/float(curr_screen.num_collisions_level)

            remaining_life_percent_lvl_widget = curr_screen.ids['remaining_life_percent_lvl'+str(screen_num)]
#            life_remaining_percent = 100-round(life_percent, 2)*100
#            remaining_life_percent_lvl_widget.text = str(int(life_remaining_percent))+"%"

            remaining_life_size_hint_x = remaining_life_percent_lvl_widget.remaining_life_size_hint_x
            remaining_life_percent_lvl_widget.size_hint =  (remaining_life_size_hint_x-remaining_life_size_hint_x*life_percent, remaining_life_percent_lvl_widget.size_hint[1])

            if curr_screen.num_collisions_hit == curr_screen.num_collisions_level:
                self.log_result(screen_num, "LOSS")
                self.bg_music.stop()
                self.char_death_sound.play()
                curr_screen.character_killed = True

                kivy.animation.Animation.cancel_all(character_image)
                for i in range(curr_screen.num_monsters): kivy.animation.Animation.cancel_all(curr_screen.ids['monster'+str(i+1)+'_image_lvl'+str(screen_num)])
                for i in range(curr_screen.num_fires): kivy.animation.Animation.cancel_all(curr_screen.ids['fire'+str(i+1)+'_lvl'+str(screen_num)])

                character_image.im_num = character_image.dead_start_im_num
                char_anim = kivy.animation.Animation(im_num=character_image.dead_end_im_num, duration=1.0)
                char_anim.start(character_image)
                kivy.clock.Clock.schedule_once(functools.partial(self.back_to_main_screen, curr_screen.parent), 3)

    def back_to_main_screen(self, screenManager, *args):
        screenManager.current = "main"

    def main_screen_on_enter(self):
        self.main_bg_music.play()

        # Batch experiment mode: automatically replay the configured level
        # until the requested number of runs has been completed.
        global RUNS_REMAINING
        if AUTO_LEVEL > 0 and RUNS_REMAINING > 0:
            RUNS_REMAINING -= 1
            kivy.clock.Clock.schedule_once(
                lambda dt: setattr(self.root, "current", "level" + str(AUTO_LEVEL)), 1.5)
        elif AUTO_LEVEL > 0 and RUNS_REMAINING <= 0:
            print("\n[KPI] All runs complete. Results saved to: " + LOG_FILE + "\n")
            kivy.clock.Clock.schedule_once(lambda dt: self.stop(), 1.0)

    def main_screen_on_leave(self):
        self.main_bg_music.stop()

class ImageButton(kivy.uix.behaviors.ButtonBehavior, kivy.uix.image.Image):
    pass

class MainScreen(kivy.uix.screenmanager.Screen):
    pass

class AboutUs(kivy.uix.screenmanager.Screen):
    pass

class AllLevelsCompleted(kivy.uix.screenmanager.Screen):
    pass

class Level1(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 5
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 1
    num_fires = 0
    num_collisions_hit = 0
    num_collisions_level = 20

class Level2(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 8
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 1
    num_fires = 0
    num_collisions_hit = 0
    num_collisions_level = 30

class Level3(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 12
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 1
    num_fires = 0
    num_collisions_hit = 0
    num_collisions_level = 30

class Level4(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 10
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 1
    num_fires = 1
    num_collisions_hit = 0
    num_collisions_level = 20

class Level5(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 15
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 1
    num_fires = 2
    num_collisions_hit = 0
    num_collisions_level = 20

class Level6(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 12
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 1
    num_fires = 3
    num_collisions_hit = 0
    num_collisions_level = 20

class Level7(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 10
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 3
    num_fires = 0
    num_collisions_hit = 0
    num_collisions_level = 25

class Level8(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 15
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 2
    num_fires = 0
    num_collisions_hit = 0
    num_collisions_level = 25

class Level9(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 12
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 2
    num_fires = 0
    num_collisions_hit = 0
    num_collisions_level = 25

class Level10(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 14
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 3
    num_fires = 0
    num_collisions_hit = 0
    num_collisions_level = 30

class Level11(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 15
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 2
    num_fires = 1
    num_collisions_hit = 0
    num_collisions_level = 30

class Level12(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 12
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 2
    num_fires = 1
    num_collisions_hit = 0
    num_collisions_level = 30

class Level13(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 10
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 2
    num_fires = 2
    num_collisions_hit = 0
    num_collisions_level = 20

class Level14(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 15
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 0
    num_fires = 6
    num_collisions_hit = 0
    num_collisions_level = 30

class Level15(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 16
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 2
    num_fires = 3
    num_collisions_hit = 0
    num_collisions_level = 30

class Level16(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 15
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 3
    num_fires = 2
    num_collisions_hit = 0
    num_collisions_level = 35

class Level17(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 10
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 0
    num_fires = 4
    num_collisions_hit = 0
    num_collisions_level = 30

class Level18(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 15
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 3
    num_fires = 4
    num_collisions_hit = 0
    num_collisions_level = 30

class Level19(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 12
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 0
    num_fires = 6
    num_collisions_hit = 0
    num_collisions_level = 30

class Level20(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 15
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 0
    num_fires = 8
    num_collisions_hit = 0
    num_collisions_level = 30

class Level21(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 18
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 2
    num_fires = 4
    num_collisions_hit = 0
    num_collisions_level = 30

class Level22(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 20
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 2
    num_fires = 4
    num_collisions_hit = 0
    num_collisions_level = 30

class Level23(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 25
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 2
    num_fires = 2
    num_collisions_hit = 0
    num_collisions_level = 30

class Level24(kivy.uix.screenmanager.Screen):
    character_killed = False
    num_coins = 20
    num_coins_collected = 0
    coins_ids = {}
    char_anim_duration = 0.25
    num_monsters = 3
    num_fires = 2
    num_collisions_hit = 0
    num_collisions_level = 30

app = CointexApp()
app.title = "CoinTex"
app.icon = 'cointex_logo.png'
app.run()
