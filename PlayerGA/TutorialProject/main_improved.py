"""
main_improved.py -- CoinTex driven by an improved genetic-algorithm agent.

COMP6009 Lab Task 2.  Derived from main_updated.py; every change is confined to
the GA controller, the fitness function (now in ga_agent.py), thread handling
and KPI logging.  The game logic itself is untouched.

Run normally:
    python main_improved.py

Run a scripted batch of experiments (no clicking required):
    GA_MODE=improved GA_LEVEL=20 GA_RUNS=10 python main_improved.py
    GA_MODE=baseline GA_LEVEL=20 GA_RUNS=10 python main_improved.py

KPIs for every episode are appended to ga_kpi_log.csv.
"""

import kivy.app
import kivy.uix.screenmanager
import kivy.uix.image
import random
import kivy.core.audio
import os
import functools
import kivy.uix.behaviors
import pickle
import threading
import kivy.base
import kivy.clock
import kivy.animation
import csv
import time
import dataclasses

import ga_agent

# =========================================================================== #
#  Agent configuration
#  --------------------------------------------------------------------------
#  Chosen from the environment so a batch of experiments can be scripted
#  without editing source:
#     GA_MODE   improved | baseline     (which agent to run)
#     GA_LEVEL  1..24                   (auto-play this level; 0 = click one)
#     GA_RUNS   n                       (how many episodes before quitting)
#     GA_LOG    path                    (KPI csv, default ga_kpi_log.csv)
# =========================================================================== #
GA_MODE = os.environ.get("GA_MODE", "improved").strip().lower()
GA_LEVEL = int(os.environ.get("GA_LEVEL", "0"))
GA_RUNS = int(os.environ.get("GA_RUNS", "1"))
GA_LOG = os.environ.get("GA_LOG", "ga_kpi_log.csv")

AGENT_CFG = (ga_agent.BASELINE_CONFIG if GA_MODE == "baseline"
             else ga_agent.IMPROVED_CONFIG)

lvl_num = 0
runs_completed = 0


# --------------------------------------------------------------------------- #
#  Reading the world
# --------------------------------------------------------------------------- #
def widget_centre(widget):
    """Centre of a widget in normalised screen coordinates.

    pos_hint is the *bottom-left* corner as a fraction of the layout, while
    start_char_animation() expects a centre (it subtracts half the character
    size from whatever it is given).  main_updated.py compared chromosomes
    against raw pos_hint values, i.e. against corners, which biased every
    target by half a widget.  Converting to centres here removes that bias.
    """
    pos = widget.pos_hint
    return (pos.get('x', 0.0) + widget.size_hint[0] / 2.0,
            pos.get('y', 0.0) + widget.size_hint[1] / 2.0)


def snapshot(app, screen_num, previous_monsters=None):
    """Freeze the current game state into a plain-data World object.

    Taking a snapshot matters: the GA then optimises a *stationary* problem for
    one short burst, instead of chasing a target that moves underneath it.
    """
    screen = app.root.screens[screen_num]

    char = widget_centre(screen.ids['character_image_lvl' + str(screen_num)])

    coins = []
    for key in list(screen.coins_ids.keys()):
        coin = screen.coins_ids.get(key)
        if coin is not None:
            coins.append(widget_centre(coin))

    monsters = []
    for i in range(screen.num_monsters):
        monsters.append(widget_centre(
            screen.ids['monster' + str(i + 1) + '_image_lvl' + str(screen_num)]))

    fires = []
    for i in range(screen.num_fires):
        fires.append(widget_centre(
            screen.ids['fire' + str(i + 1) + '_lvl' + str(screen_num)]))

    # one-step velocity estimate, used to aim at where a monster will be
    velocities = []
    if previous_monsters and len(previous_monsters) == len(monsters):
        for (mx, my), (px, py) in zip(monsters, previous_monsters):
            velocities.append((mx - px, my - py))

    return ga_agent.World(char=char, coins=coins, monsters=monsters,
                          fires=fires, monster_vel=velocities), monsters


def config_for_screen(cfg, app, screen_num):
    """Restrict the search space to the points the character can actually
    occupy: the unit square shrunk by half the character size."""
    if cfg.fitness == "baseline":
        return cfg                       # keep the original (0.3, 1.3) range
    char = app.root.screens[screen_num].ids['character_image_lvl' + str(screen_num)]
    w, h = char.size_hint
    return dataclasses.replace(cfg,
                               bounds_x=(w / 2.0, 1.0 - w / 2.0),
                               bounds_y=(h / 2.0, 1.0 - h / 2.0))


# --------------------------------------------------------------------------- #
#  The controller
# --------------------------------------------------------------------------- #
class GAController(threading.Thread):
    """Sense -> optimise -> act, once per character animation.

    main_updated.py instead ran a single 10 000-generation GA and only moved the
    character on the generations where the best fitness happened to change, so
    the agent froze as soon as the population converged, and every movement
    command was issued from a worker thread straight into Kivy widgets.  Here
    the loop is explicit, paced to the animation, and every widget touch is
    marshalled back onto the main thread.
    """

    def __init__(self, app, screen_num, cfg):
        super().__init__(daemon=True)
        self.app = app
        self.screen_num = screen_num
        self.cfg = cfg
        self.stop_flag = threading.Event()
        self.decisions = 0

    def run(self):
        screen = self.app.root.screens[self.screen_num]
        cfg = config_for_screen(self.cfg, self.app, self.screen_num)
        period = screen.char_anim_duration
        previous_best = None
        last_fitness = None
        previous_monsters = None

        while not self.stop_flag.is_set():
            if screen.character_killed or len(screen.coins_ids) == 0:
                break

            try:
                world, previous_monsters = snapshot(self.app, self.screen_num,
                                                    previous_monsters)
            except (KeyError, AttributeError):
                break                    # screen torn down underneath us
            if not world.coins:
                break

            decision = ga_agent.decide(world, cfg, previous_best)
            previous_best = decision.target
            self.decisions += 1

            move = True
            if cfg.move_policy == "on_change":
                move = (last_fitness is None
                        or abs(decision.fitness - last_fitness) > 1e-12)
            last_fitness = decision.fitness

            if move:
                self.app.request_move(self.screen_num, decision.target)

            time.sleep(period)

        self.app.finish_episode(self.screen_num, self.decisions)


# --------------------------------------------------------------------------- #
#  KPI logging
# --------------------------------------------------------------------------- #
KPI_FIELDS = ["config", "mode", "level", "run", "coins_total", "coins_collected",
              "collection_rate", "success", "died", "time_s", "life_remaining",
              "decisions"]


_kpi_lock = threading.Lock()


def append_kpi(row):
    new_file = not os.path.exists(GA_LOG)
    with open(GA_LOG, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=KPI_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    print("[KPI] " + "  ".join("%s=%s" % (k, row[k]) for k in KPI_FIELDS))


class CointexApp(kivy.app.App):

    controller = None

    def on_start(self):
        music_dir = os.getcwd()+"/music/"
        self.main_bg_music = kivy.core.audio.SoundLoader.load(music_dir+"bg_music_piano_flute.wav")
        if self.main_bg_music is not None:
            self.main_bg_music.loop = True
            self.main_bg_music.play()

        next_level_num, congrats_displayed_once = self.read_game_info()
        self.activate_levels(next_level_num, congrats_displayed_once)

        print("[agent] mode=%s config=%s level=%s runs=%s log=%s"
              % (GA_MODE, AGENT_CFG.name, GA_LEVEL or "manual", GA_RUNS, GA_LOG))
        if GA_LEVEL > 0:
            kivy.clock.Clock.schedule_once(
                lambda dt: setattr(self.root, "current", "level" + str(GA_LEVEL)), 1.0)

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

        global lvl_num
        lvl_num = screen_num

        curr_screen.episode_start = time.time()
        curr_screen.episode_logged = False

        if getattr(self, "controller", None) is not None:
            self.controller.stop_flag.set()
        self.controller = GAController(app=self, screen_num=screen_num,
                                       cfg=AGENT_CFG)
        self.controller.start()

    # ------------------------------------------------------------------ #
    #  Agent plumbing
    # ------------------------------------------------------------------ #
    def request_move(self, screen_num, target):
        """Hand a target to the main thread.

        Kivy widgets and animations are not thread-safe; the original code
        called start_char_animation() directly from the GA thread.  Clock
        callbacks are the supported way to cross that boundary."""
        kivy.clock.Clock.schedule_once(
            lambda dt: self.start_char_animation(screen_num, target), 0)

    def finish_episode(self, screen_num, decisions=0):
        """Write one row of KPIs for the episode that just ended."""
        global runs_completed
        curr_screen = self.root.screens[screen_num]
        with _kpi_lock:
            if getattr(curr_screen, "episode_logged", False):
                return
            curr_screen.episode_logged = True

        collected = curr_screen.num_coins_collected
        total = curr_screen.num_coins
        died = bool(curr_screen.character_killed)
        elapsed = time.time() - getattr(curr_screen, "episode_start", time.time())
        life = max(0.0, 1.0 - float(curr_screen.num_collisions_hit)
                   / float(curr_screen.num_collisions_level))

        runs_completed += 1
        append_kpi(dict(config=AGENT_CFG.name, mode=GA_MODE, level=screen_num,
                        run=runs_completed, coins_total=total,
                        coins_collected=collected,
                        collection_rate=round(collected / float(total), 4),
                        success=int(collected >= total and not died),
                        died=int(died), time_s=round(elapsed, 2),
                        life_remaining=round(life, 4), decisions=decisions))

    def maybe_start_next_run(self):
        """Auto-replay the configured level so experiments need no clicking."""
        if GA_LEVEL <= 0:
            return
        if runs_completed >= GA_RUNS:
            kivy.clock.Clock.schedule_once(lambda dt: kivy.base.stopTouchApp(), 1.0)
            return
        kivy.clock.Clock.schedule_once(
            lambda dt: setattr(self.root, "current", "level" + str(GA_LEVEL)), 1.5)

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
                self.bg_music.stop()
                self.char_death_sound.play()
                curr_screen.character_killed = True
                self.finish_episode(screen_num,
                                    getattr(getattr(self, "controller", None),
                                            "decisions", 0))

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
                    self.bg_music.stop()
                    self.level_completed_sound.play()
                    self.finish_episode(screen_num,
                                        getattr(getattr(self, "controller", None),
                                                "decisions", 0))
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
                self.bg_music.stop()
                self.char_death_sound.play()
                curr_screen.character_killed = True
                self.finish_episode(screen_num,
                                    getattr(getattr(self, "controller", None),
                                            "decisions", 0))

                kivy.animation.Animation.cancel_all(character_image)
                for i in range(curr_screen.num_monsters): kivy.animation.Animation.cancel_all(curr_screen.ids['monster'+str(i+1)+'_image_lvl'+str(screen_num)])
                for i in range(curr_screen.num_fires): kivy.animation.Animation.cancel_all(curr_screen.ids['fire'+str(i+1)+'_lvl'+str(screen_num)])

                character_image.im_num = character_image.dead_start_im_num
                char_anim = kivy.animation.Animation(im_num=character_image.dead_end_im_num, duration=1.0)
                char_anim.start(character_image)
                kivy.clock.Clock.schedule_once(functools.partial(self.back_to_main_screen, curr_screen.parent), 3)

    def back_to_main_screen(self, screenManager, *args):
        if getattr(self, "controller", None) is not None:
            self.controller.stop_flag.set()
        screenManager.current = "main"
        self.maybe_start_next_run()

    def main_screen_on_enter(self):
        if self.main_bg_music is not None:
            self.main_bg_music.play()

    def main_screen_on_leave(self):
        if self.main_bg_music is not None:
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
