import json
import os
import random
import uuid
from collections import deque
from pathlib import Path
from typing import Optional

import mediapy as media
import numpy as np
from skimage.transform import resize
from gymnasium import Env, spaces
from pyboy import PyBoy
from pyboy.utils import WindowEvent

from pokemonred_puffer.global_map import GLOBAL_MAP_SHAPE, local_to_global
import cv2
from functools import partial
from collections import defaultdict, deque

EVENT_FLAGS_START = 0xD747
EVENTS_FLAGS_LENGTH = 320
MUSEUM_TICKET = (0xD754, 0)
PARTY_SIZE = 0xD163
PARTY_LEVEL_ADDRS = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]

CUT_SEQ = [
    ((0x3D, 1, 1, 0, 4, 1), (0x3D, 1, 1, 0, 1, 1)),
    ((0x50, 1, 1, 0, 4, 1), (0x50, 1, 1, 0, 1, 1)),
]

X_POS_ADDR = 0xD362
Y_POS_ADDR = 0xD361
MAP_N_ADDR = 0xD35E
CUT_GRASS_SEQ = deque([(0x52, 255, 1, 0, 1, 1), (0x52, 255, 1, 0, 1, 1), (0x52, 1, 1, 0, 1, 1)])
CUT_FAIL_SEQ = deque([(-1, 255, 0, 0, 4, 1), (-1, 255, 0, 0, 1, 1), (-1, 255, 0, 0, 1, 1)])
bg = cv2.imread('/puffertank/tg_networks/pokemonred_puffer/pokemonred_puffer/k_map_1_16th_scale.png')

MAP_PATH = '/puffertank/tg_networks/pokemonred_puffer/pokemonred_puffer/map_data.json' # __file__.rstrip('game_map.py') + 'map_data.json'
MAP_DATA = json.load(open(MAP_PATH, 'r'))['regions']
MAP_DATA = {int(e['id']): e for e in MAP_DATA}

# Handle KeyErrors
def local_to_global(r, c, map_n):
    try:
        map_x, map_y,= MAP_DATA[map_n]['coordinates']
        return r + map_y, c + map_x
    except KeyError:
        print(f'Map id {map_n} not found in map_data.json.')
        return r + 0, c + 0

# VISITED_MASK_SHAPE = (144 // 16, 160 // 16, 1)


# TODO: Make global map usage a configuration parameter
class RedGymEnv(Env):
    def __init__(self, config=None):
        self.s_path = config["session_path"]
        self.save_final_state = config["save_final_state"]
        self.print_rewards = config["print_rewards"]
        self.headless = config["headless"]
        self.init_state = config["init_state"]
        self.act_freq = config["action_freq"]
        self.max_steps = config["max_steps"]
        self.save_video = config["save_video"]
        self.fast_video = config["fast_video"]
        self.frame_stacks = config["frame_stacks"]
        self.explore_weight = 1 if "explore_weight" not in config else config["explore_weight"]
        self.explore_npc_weight = (
            1 if "explore_npc_weight" not in config else config["explore_npc_weight"]
        )
        self.explore_hidden_obj_weight = (
            1 if "explore_hidden_obj_weight" not in config else config["explore_hidden_obj_weight"]
        )
        self.instance_id = (
            str(uuid.uuid4())[:8] if "instance_id" not in config else config["instance_id"]
        )
        self.step_forgetting_factor = config["step_forgetting_factor"]
        self.forgetting_frequency = config["forgetting_frequency"]
        self.perfect_ivs = config["perfect_ivs"]
        self.s_path.mkdir(exist_ok=True)
        self.full_frame_writer = None
        self.model_frame_writer = None
        self.map_frame_writer = None
        self.reset_count = 0
        self.all_runs = []
        self.counts_map = np.zeros((444, 436), dtype=np.uint8)

        self.essential_map_locations = {
            v: i for i, v in enumerate([40, 0, 12, 1, 13, 51, 2, 54, 14, 59, 60, 61, 15, 3, 65])
        }

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]

        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START,
        ]

        # load event names (parsed from https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm)
        with open(os.path.join(os.path.dirname(__file__), "events.json")) as f:
            event_names = json.load(f)
        self.event_names = event_names

        # self.screen_output_shape = (144, 160, self.frame_stacks)
        self.coords_pad = 12

        self.counts_map_overlay_obs = np.zeros((444, 436))
        self.counts_map_overlay_output_shape = self.counts_map_overlay_obs.shape
        self.screen_window_combined = np.zeros((144, 160, 4))
        self.screen_output_shape = self.screen_window_combined.shape

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))

        self.enc_freqs = 8
        
        self.observation_space = spaces.Dict({
    "screen": spaces.Box(low=0, high=255, shape=(144, 160, 3), dtype=np.uint8),
    "global_map": spaces.Box(low=0, high=255, shape=(444, 436, 3), dtype=np.uint8),
})

        # self.observation_space = spaces.Dict(
        #     {
        #         "screen": spaces.Box(
        #             low=0, high=255, shape=self.screen_output_shape, dtype=np.uint8
        #         ),
        #         # "masks": spaces.Box(
        #         #     low=0, high=1, shape=VISITED_MASK_SHAPE, dtype=np.float32
        #         # ),
        #         "global_map": spaces.Box(low=0, high=1, shape=(*GLOBAL_MAP_SHAPE, 1), dtype=np.float32),
        #     }
        # )

        head = "headless" if config["headless"] else "SDL2"

        self.pyboy = PyBoy(
            config["gb_path"],
            debugging=False,
            disable_input=False,
            window_type=head,
        )

        self.screen = self.pyboy.botsupport_manager().screen()
        R, C = self.screen.raw_screen_buffer_dims()
        self.obs_size = (R, C) 
        self.screen_memory = defaultdict(
        lambda: np.zeros((255, 255, 1), dtype=np.uint8))
        self.obs_size += (4,)

        if not config["headless"]:
            self.pyboy.set_emulation_speed(6)
        
        self.first = True
        
    def position(self):
        r_pos = self.pyboy.get_memory_value(Y_POS_ADDR)
        c_pos = self.pyboy.get_memory_value(X_POS_ADDR)
        map_n = self.pyboy.get_memory_value(MAP_N_ADDR)
        if r_pos >= 443:
            r_pos = 444
        if r_pos <= 0:
            r_pos = 0
        if c_pos >= 443:
            c_pos = 444
        if c_pos <= 0:
            c_pos = 0
        if map_n > 247:
            map_n = 247
        if map_n < -1:
            map_n = -1
        return r_pos, c_pos, map_n

    def make_pokemon_red_overlay_fast(self, bg, counts):
        # Normalize counts to the range [0, 255] for grayscale
        if counts.max() > 0:
            counts_normalized = np.clip((counts / counts.max()) * 255, 0, 255).astype(np.uint8)
        else:
            counts_normalized = np.zeros_like(counts, dtype=np.uint8)

        # Expand the overlay to have three channels
        counts_resized_expanded = np.repeat(counts_normalized[:, :, np.newaxis], 3, axis=2)

        # Apply the overlay with adjusted alpha
        alpha = 0.8  # Adjust alpha to control the visibility of the overlay
        overlay = ((1 - alpha) * bg.astype(float) + alpha * counts_resized_expanded.astype(float)).astype(np.uint8)

        return overlay


    # def make_pokemon_red_overlay_fast(self, bg, counts):
    #     if counts.max() > 0:
    #         # Normalize counts to the range [0, 255] for grayscale
    #         counts_normalized = np.clip((counts / counts.max()) * 255, 0, 255).astype(np.uint8)
    #     else:
    #         # Handle the case where counts.max() <= 0, e.g., by setting to zero
    #         counts_normalized = np.zeros_like(counts, dtype=np.uint8)
    #     # Resize counts map to the size of the background
    #     counts_resized = cv2.resize(counts_normalized, (bg.shape[1], bg.shape[0]), interpolation=cv2.INTER_NEAREST)
    #     # Create a mask where counts are non-zero
    #     mask = counts_resized > 0
    #     # Prepare the overlay by combining background and scaled counts map
    #     # Adjust the alpha value to control the visibility of the overlay
    #     alpha = 0.8
    #     overlay = np.where(mask, ((1-alpha) * bg + alpha * counts_resized), bg).astype(np.uint8)
    #     return overlay
    
    def map_updater(self):
        # bg = cv2.imread('kanto_map_dsv.png', cv2.IMREAD_GRAYSCALE)  # Load in grayscale
        return partial(self.make_pokemon_red_overlay_fast, bg)
    
    def get_fixed_window_2(self, arr, y, x, window_size):
        height, width, _ = arr.shape
        h_w, w_w = window_size[0], window_size[1]
        h_w, w_w = window_size[0] // 2, window_size[1] // 2
        y_min = max(0, y - h_w)
        y_max = min(height, y + h_w + (window_size[0] % 2))
        x_min = max(0, x - w_w)
        x_max = min(width, x + w_w + (window_size[1] % 2))
        window = arr[y_min:y_max, x_min:x_max]
        pad_top = h_w - (y - y_min)
        pad_bottom = h_w + (window_size[0] % 2) - 1 - (y_max - y - 1)
        pad_left = w_w - (x - x_min)
        pad_right = w_w + (window_size[1] % 2) - 1 - (x_max - x - 1)
        return np.pad(
            window,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
        )


    def update_heat_map(self, r, c, current_map):
        '''
        Updates the heat map based on the agent's current position.
        Args:
            r (int): global y coordinate of the agent's position.
            c (int): global x coordinate of the agent's position.
            current_map (int): ID of the current map (map_n)
        Updates the counts_map to track the frequency of visits to each position on the map.
        '''
        # Convert local position to global position
        try:
            glob_r, glob_c = local_to_global(r, c, current_map)
        except IndexError:
            print(f'IndexError: index {glob_r} or {glob_c} for {current_map} is out of bounds for axis 0 with size 444.')
            glob_r = 0
            glob_c = 0
        # Update heat map based on current map
        if self.last_map == current_map or self.last_map == -1:
            # Increment count for current global position
                try:
                    self.counts_map[glob_r, glob_c] += 1
                except:
                    pass
        else:
            # Reset count for current global position if it's a new map for warp artifacts
            self.counts_map[(glob_r, glob_c)] = -1
        # Update last_map for the next iteration
        self.last_map = current_map

    def reset(self, seed: Optional[int] = None):
        # restart game, skipping credits
        self.explore_map_dim = 384
        if self.first:
            self.recent_screens = deque()  # np.zeros(self.output_shape, dtype=np.uint8)
            self.recent_actions = deque()  # np.zeros((self.frame_stacks,), dtype=np.uint8)
            self.seen_pokemon = np.zeros(152, dtype=np.uint8)
            self.caught_pokemon = np.zeros(152, dtype=np.uint8)
            self.moves_obtained = np.zeros(0xA5, dtype=np.uint8)
            self.explore_map = np.zeros(GLOBAL_MAP_SHAPE, dtype=np.float32)
            self.init_map_mem()
            self.init_npc_mem()
            self.init_hidden_obj_mem()
            self.init_cut_mem()

        if self.first:  # or random.uniform(0, 1) < 0.5:
            with open(self.init_state, "rb") as f:
                self.pyboy.load_state(f)
            self.recent_screens.clear()
            self.recent_actions.clear()
            self.seen_pokemon.fill(0)
            self.caught_pokemon.fill(0)
            self.moves_obtained.fill(0)

            # lazy random seed setting
            if not seed:
                seed = random.randint(0, 4096)
            for _ in range(seed):
                self.pyboy.tick()

            self.explore_map *= 0
            self.init_map_mem()
            self.init_npc_mem()
            self.init_hidden_obj_mem()
            self.init_cut_mem()

        self.taught_cut = self.check_if_party_has_cut()
        self.base_event_flags = sum(
            self.bit_count(self.read_m(i))
            for i in range(EVENT_FLAGS_START, EVENT_FLAGS_START + EVENTS_FLAGS_LENGTH)
        )

        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.max_level_sum = 0
        self.last_health = 1
        self.total_healing_rew = 0
        self.died_count = 0
        self.party_size = 0
        self.step_count = 0

        self.current_event_flags_set = {}

        self.action_hist = np.zeros(len(self.valid_actions))
        self.max_map_progress = 0
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])

        self.reset_count += 1
        self.first = False
        return self._get_obs(), {}

    def init_map_mem(self):
        self.seen_coords = {}
        # self.seen_global_coords = np.zeros(GLOBAL_MAP_SHAPE)
        self.seen_map_ids = np.zeros(256)

    def init_npc_mem(self):
        self.seen_npcs = {}

    def init_hidden_obj_mem(self):
        self.seen_hidden_objs = {}

    def init_cut_mem(self):
        self.cut_coords = {}
        self.cut_state = deque(maxlen=3)

    def step_forget_explore(self):
        self.seen_coords.update(
            (k, max(0.15, v * self.step_forgetting_factor["coords"]))
            for k, v in self.seen_coords.items()
        )
        # self.seen_global_coords *= self.step_forgetting_factor["coords"]
        self.seen_map_ids *= self.step_forgetting_factor["map_ids"]
        self.seen_npcs.update(
            (k, max(0.15, v * self.step_forgetting_factor["npc"]))
            for k, v in self.seen_npcs.items()
        )
        self.explore_map *= self.step_forgetting_factor["explore"]
        self.explore_map[self.explore_map > 0] = np.clip(
            self.explore_map[self.explore_map > 0], 0.15, 1
        )

    def render(self, reduce_res=False):
        # (144, 160, 3)
        game_pixels_render = self.screen.screen_ndarray()[:, :, 0:1]
        # # place an overlay on top of the screen greying out places we haven't visited
        # # first get our location
        # player_x, player_y, map_n = self.get_game_coords()
        # # game_pixels_render = np.concatenate([game_pixels_render, visited_mask, cut_mask], axis=-1)
        # # game_pixels_render = np.concatenate([game_pixels_render, visited_mask], axis=-1)

        r, c, map_n = self.get_game_coords()
        mmap = self.screen_memory[map_n]
        if 0 <= r <= 254 and 0 <= c <= 254:
            mmap[r, c] = 255
        # screen_data_2 = (self.screen.screen_ndarray()[:, :]).astype(np.uint8) # astype(np.uint8)
        screen_data_2 = self.screen.screen_ndarray()[:, :, 0:1]
        window_data = (self.get_fixed_window_2(mmap, r, c, (144, 160, 1))).astype(np.uint8) # astype(np.uint8)

        # If window_data is (144, 160, 1), repeat its channels to match screen_data_2's shape (144, 160, 3)
        # window_data_expanded = np.repeat(window_data, 3, axis=-1)
        # Concatenate window_data and screen_data_2 along the channel axis
        print(f'LINE317 \nwindow_data shape={np.shape(window_data)},\nscreen_data_2 shape={np.shape(screen_data_2)}\n')

        screen_window_combined = np.concatenate((screen_data_2, window_data, window_data * 0), axis=2)
        # screen_window_combined = np.concatenate((screen_data_2, window_data), axis=2)
        
        # obs_size=3, window_data size=23040, counts_map_overlay_obs size=193584, screen_data_2 size=69120
        # obs_shape=(3,), window_data shape=(144, 160, 1), counts_map_overlay_obs shape=(444, 436), screen_data_2 shape=(144, 160, 3)
        print(f'screen_window_combined size&shape: {np.size(screen_window_combined)}\n{np.shape(screen_window_combined)}')
        #         LINE317 
        # window_data shape=(144, 160, 1),
        # screen_data_2 shape=(144, 160, 1)

        # screen_window_combined size&shape: 46080
        # (144, 160, 2)
        # Error generating overlay: operands could not be broadcast together with shapes (444,436,3) (444,436) 
        if reduce_res:
            # game_pixels_render = (
            #     downscale_local_mean(game_pixels_render, (2, 2, 1))
            # ).astype(np.uint8)
            game_pixels_render = game_pixels_render[::2, ::2, :]
        return {
            "screen": screen_window_combined,
            # "masks": visited_mask,
        }

    def _get_obs(self):
        screen = self.render()
        try:
            # Use the adjusted overlay function to generate the overlay
            overlay = self.make_pokemon_red_overlay_fast(bg, self.counts_map)     
            # Assuming `overlay` now correctly matches the shape and type expected by your environment
            # If `screen['screen']` is where the overlay should be applied, ensure compatibility
            # For simplicity, let's say we're directly using `overlay` as part of the observation
            self.counts_map_overlay_obs = overlay
            print(f'LINE430 self.counts_map_overlay_obs size&shape: {np.size(self.counts_map_overlay_obs)}\n{np.shape(self.counts_map_overlay_obs)}')
        except Exception as e:
            print(f"Error generating overlay: {e}")
            # Fallback or error handling
        return {**screen, "global_map": np.expand_dims(self.counts_map_overlay_obs, axis=-1)}

    # def _get_obs(self):
    #     screen = self.render()
    #     """
    #     screen = np.concatenate(
    #         [
    #             screen,
    #             np.expand_dims(
    #                 255 * resize(self.explore_map, screen.shape[:-1], anti_aliasing=False),
    #                 axis=-1,
    #             ).astype(np.uint8),
    #         ],
    #         axis=-1,
    #     )
    #     """
    #     try:
    #         # Assuming counts_resized is your grayscale overlay with shape (444, 436)
    #         # And bg is your background image with shape (444, 436, 3)
    #         # Expand the overlay to have three channels
    #         counts_resized_expanded = np.repeat(counts_resized[:, :, np.newaxis], 3, axis=2)
    #         alpha = 0.8  # Adjust alpha to control the visibility of the overlay
    #         overlay = ((1 - alpha) * bg + alpha * counts_resized_expanded).astype(np.uint8)
    #         # Verify the shape
    #         print(counts_resized_expanded.shape)  # Should be (444, 436, 3)
    #         overlay_function = self.map_updater()
    #         self.counts_map_overlay_obs = np.expand_dims(
    #                 255 * resize(overlay_function(self.counts_map), screen.shape[:-1], anti_aliasing=False),
    #                 axis=-1,).astype(np.uint8)
    #         print(f'counts_map_overlay_obs shape&size: {np.size(self.counts_map_overlay_obs)}\n{np.shape(self.counts_map_overlay_obs)}')
    #     except Exception as e:
    #         print(f"Error generating overlay: {e}")
    #         print(f'counts_map_overlay_obs shape&size: {np.size(self.counts_map_overlay_obs)}\n{np.shape(self.counts_map_overlay_obs)}')
    #     return {**screen, "global_map": np.expand_dims(self.counts_map_overlay_obs, axis=-1)}

    def set_perfect_iv_dvs(self):
        party_size = self.read_m(PARTY_SIZE)
        for i in [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247][:party_size]:
            for m in range(12):  # Number of offsets for IV/DV
                self.pyboy.set_memory_value(i + 17 + m, 0xFF)

    def check_if_party_has_cut(self) -> bool:
        party_size = self.read_m(PARTY_SIZE)
        for i in [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247][:party_size]:
            for m in range(4):  # Number of offsets for IV/DV
                if self.pyboy.get_memory_value(i + 8 + m) == 15:
                    return True
        return False

    def step(self, action):
        if self.save_video and self.step_count == 0:
            self.start_video()

        if self.step_count % self.forgetting_frequency == 0:
            self.step_forget_explore()

        self.run_action_on_emulator(action)
        # self.update_recent_actions(action)
        r, c, map_n = self.position()
        self.update_heat_map(r, c, map_n)
        self.update_seen_coords()
        self.update_heal_reward()
        self.update_pokedex()
        self.update_moves_obtained()
        self.party_size = self.read_m(0xD163)
        self.update_max_op_level()
        new_reward = self.update_reward()
        self.last_health = self.read_hp_fraction()
        self.update_map_progress()
        if self.perfect_ivs:
            self.set_perfect_iv_dvs()
        self.taught_cut = self.check_if_party_has_cut()

        info = {}
        # TODO: Make log frequency a configuration parameter
        if self.step_count % 2000 == 0: # 20000
            info = self.agent_stats(action)
        obs = self._get_obs()
        self.step_count += 1

        # return obs, new_reward, self.step_count > self.max_steps, False, info
        return obs, new_reward, False, False, info

    def find_neighboring_sign(self, sign_id, player_direction, player_x, player_y) -> bool:
        sign_y = self.pyboy.get_memory_value(0xD4B1 + (2 * sign_id))
        sign_x = self.pyboy.get_memory_value(0xD4B1 + (2 * sign_id + 1))

        # Check if player is facing the sign (skip sign direction)
        # 0 - down, 4 - up, 8 - left, 0xC - right
        # We are making the assumption that a player will only ever be 1 space away
        # from a sign
        return (
            (player_direction == 0 and sign_x == player_x and sign_y == player_y + 1)
            or (player_direction == 4 and sign_x == player_x and sign_y == player_y - 1)
            or (player_direction == 8 and sign_y == player_y and sign_x == player_x - 1)
            or (player_direction == 0xC and sign_y == player_y and sign_x == player_x + 1)
        )

    def find_neighboring_npc(self, npc_id, player_direction, player_x, player_y) -> int:
        npc_y = self.pyboy.get_memory_value(0xC104 + (npc_id * 0x10))
        npc_x = self.pyboy.get_memory_value(0xC106 + (npc_id * 0x10))

        # Check if player is facing the NPC (skip NPC direction)
        # 0 - down, 4 - up, 8 - left, 0xC - right
        if (
            (player_direction == 0 and npc_x == player_x and npc_y > player_y)
            or (player_direction == 4 and npc_x == player_x and npc_y < player_y)
            or (player_direction == 8 and npc_y == player_y and npc_x < player_x)
            or (player_direction == 0xC and npc_y == player_y and npc_x > player_x)
        ):
            # Manhattan distance
            return abs(npc_y - player_y) + abs(npc_x - player_x)

        return False

    def run_action_on_emulator(self, action):
        self.action_hist[action] += 1

        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        if not self.save_video and self.headless:
            self.pyboy._rendering(False)
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 8 and action < len(self.release_actions):
                # release button
                self.pyboy.send_input(self.release_actions[action])

            if self.save_video and not self.fast_video:
                self.add_video_frame()
            if i == self.act_freq - 1:
                # rendering must be enabled on the tick before frame is needed
                self.pyboy._rendering(True)
            self.pyboy.tick()

        # Cut check
        # 0xCFC6 - wTileInFrontOfPlayer
        # 0xCFCB - wUpdateSpritesEnabled
        if self.taught_cut:
            player_direction = self.pyboy.get_memory_value(0xC109)
            x, y, map_id = self.get_game_coords()  # x, y, map_id
            if player_direction == 0:  # down
                coords = (x, y + 1, map_id)
            if player_direction == 4:
                coords = (x, y - 1, map_id)
            if player_direction == 8:
                coords = (x - 1, y, map_id)
            if player_direction == 0xC:
                coords = (x + 1, y, map_id)
            self.cut_state.append(
                (
                    self.pyboy.get_memory_value(0xCFC6),
                    self.pyboy.get_memory_value(0xCFCB),
                    self.pyboy.get_memory_value(0xCD6A),
                    self.pyboy.get_memory_value(0xD367),
                    self.pyboy.get_memory_value(0xD125),
                    self.pyboy.get_memory_value(0xCD3D),
                )
            )
            if tuple(list(self.cut_state)[1:]) in CUT_SEQ:
                self.cut_coords[coords] = 1
            elif self.cut_state == CUT_GRASS_SEQ:
                self.cut_coords[coords] = 0.3
            elif deque([(-1, *elem[1:]) for elem in self.cut_state]) == CUT_FAIL_SEQ:
                self.cut_coords[coords] = 0.005

        # check if the font is loaded
        if self.pyboy.get_memory_value(0xCFC4):
            # check if we are talking to a hidden object:
            player_direction = self.pyboy.get_memory_value(0xC109)
            player_y_tiles = self.pyboy.get_memory_value(0xD361)
            player_x_tiles = self.pyboy.get_memory_value(0xD362)
            if (
                self.pyboy.get_memory_value(0xCD3D) != 0x0
                and self.pyboy.get_memory_value(0xCD3E) != 0x0
            ):
                # add hidden object to seen hidden objects
                self.seen_hidden_objs[
                    (
                        self.pyboy.get_memory_value(0xD35E),
                        self.pyboy.get_memory_value(0xCD3F),
                    )
                ] = 1
            elif any(
                self.find_neighboring_sign(
                    sign_id, player_direction, player_x_tiles, player_y_tiles
                )
                for sign_id in range(self.pyboy.get_memory_value(0xD4B0))
            ):
                pass
            else:
                # get information for player
                player_y = self.pyboy.get_memory_value(0xC104)
                player_x = self.pyboy.get_memory_value(0xC106)
                # get the npc who is closest to the player and facing them
                # we go through all npcs because there are npcs like
                # nurse joy who can be across a desk and still talk to you

                # npc_id 0 is the player
                npc_distances = (
                    (
                        self.find_neighboring_npc(npc_id, player_direction, player_x, player_y),
                        npc_id,
                    )
                    for npc_id in range(1, self.pyboy.get_memory_value(0xD4E1))
                )
                npc_candidates = [x for x in npc_distances if x[0]]
                if npc_candidates:
                    _, npc_id = min(npc_candidates, key=lambda x: x[0])
                    self.seen_npcs[(self.pyboy.get_memory_value(0xD35E), npc_id)] = 1

        if self.save_video and self.fast_video:
            self.add_video_frame()

    def agent_stats(self, action):
        x_pos, y_pos, map_n = self.get_game_coords()
        levels = [self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
        return {
            "stats": {
                "step": self.step_count,
                "x": x_pos,
                "y": y_pos,
                "map": map_n,
                "map_location": self.get_map_location(map_n),
                "max_map_progress": self.max_map_progress,
                "last_action": action,
                "pcount": self.read_m(0xD163),
                "levels": levels,
                "levels_sum": sum(levels),
                "ptypes": self.read_party(),
                "hp": self.read_hp_fraction(),
                "coord": sum(self.seen_coords.values()),  # np.sum(self.seen_global_coords),
                "map_id": np.sum(self.seen_map_ids),
                "npc": sum(self.seen_npcs.values()),
                "hidden_obj": sum(self.seen_hidden_objs.values()),
                "deaths": self.died_count,
                "badge": self.get_badges(),
                "event": self.progress_reward["event"],
                "healr": self.total_healing_rew,
                "action_hist": self.action_hist,
                "caught_pokemon": int(sum(self.caught_pokemon)),
                "seen_pokemon": int(sum(self.seen_pokemon)),
                "moves_obtained": int(sum(self.moves_obtained)),
                "opponent_level": self.max_opponent_level,
                "met_bill": int(self.read_bit(0xD7F1, 0)),
                "used_cell_separator_on_bill": int(self.read_bit(0xD7F2, 3)),
                "ss_ticket": int(self.read_bit(0xD7F2, 4)),
                "met_bill_2": int(self.read_bit(0xD7F2, 5)),
                "bill_said_use_cell_separator": int(self.read_bit(0xD7F2, 6)),
                "left_bills_house_after_helping": int(self.read_bit(0xD7F2, 7)),
                "got_hm01": int(self.read_bit(0xD803, 0)),
                "rubbed_captains_back": int(self.read_bit(0xD803, 1)),
                "taught_cut": int(self.check_if_party_has_cut()),
                "cut_coords": sum(self.cut_coords.values()),
            },
            "reward": self.get_game_state_reward(),
            "reward/reward_sum": sum(self.get_game_state_reward().values()),
            "pokemon_exploration_map": self.explore_map,
        }

    def start_video(self):
        if self.full_frame_writer is not None:
            self.full_frame_writer.close()
        if self.model_frame_writer is not None:
            self.model_frame_writer.close()
        if self.map_frame_writer is not None:
            self.map_frame_writer.close()

        base_dir = self.s_path / Path("rollouts")
        base_dir.mkdir(exist_ok=True)
        full_name = Path(f"full_reset_{self.reset_count}_id{self.instance_id}").with_suffix(".mp4")
        model_name = Path(f"model_reset_{self.reset_count}_id{self.instance_id}").with_suffix(
            ".mp4"
        )
        self.full_frame_writer = media.VideoWriter(
            base_dir / full_name, (144, 160), fps=60, input_format="gray"
        )
        self.full_frame_writer.__enter__()
        self.model_frame_writer = media.VideoWriter(
            base_dir / model_name, self.screen_output_shape[:2], fps=60, input_format="gray"
        )
        self.model_frame_writer.__enter__()
        map_name = Path(f"map_reset_{self.reset_count}_id{self.instance_id}").with_suffix(".mp4")
        self.map_frame_writer = media.VideoWriter(
            base_dir / map_name,
            (self.coords_pad * 4, self.coords_pad * 4),
            fps=60,
            input_format="gray",
        )
        self.map_frame_writer.__enter__()

    def add_video_frame(self):
        self.full_frame_writer.add_image(self.render(reduce_res=False)[:, :, 0])
        self.model_frame_writer.add_image(self.render(reduce_res=True)[:, :, 0])

    def get_game_coords(self):
        return (self.read_m(0xD362), self.read_m(0xD361), self.read_m(0xD35E))

    def update_seen_coords(self):
        x_pos, y_pos, map_n = self.get_game_coords()
        self.seen_coords[(x_pos, y_pos, map_n)] = 1
        self.explore_map[local_to_global(y_pos, x_pos, map_n)] = 1
        # self.seen_global_coords[local_to_global(y_pos, x_pos, map_n)] = 1
        self.seen_map_ids[map_n] = 1

    def get_explore_map(self):
        explore_map = np.zeros(GLOBAL_MAP_SHAPE)
        for (x, y, map_n), v in self.seen_coords.items():
            gy, gx = local_to_global(y, x, map_n)
            if gy >= explore_map.shape[0] or gx >= explore_map.shape[1]:
                print(f"coord out of bounds! global: ({gx}, {gy}) game: ({x}, {y}, {map_n})")
            else:
                explore_map[gy, gx] = v

        return explore_map

    def update_recent_screens(self, cur_screen):
        # self.recent_screens = np.roll(self.recent_screens, 1, axis=2)
        # self.recent_screens[:, :, 0] = cur_screen[:, :, 0]
        self.recent_screens.append(cur_screen)
        if len(self.recent_screens) > self.frame_stacks:
            self.recent_screens.popleft()

    def update_recent_actions(self, action):
        # self.recent_actions = np.roll(self.recent_actions, 1)
        # self.recent_actions[0] = action
        self.recent_actions.append(action)
        if len(self.recent_actions) > self.frame_stacks:
            self.recent_actions.popleft()

    def update_reward(self):
        # compute reward
        self.progress_reward = self.get_game_state_reward()
        new_total = sum([val for _, val in self.progress_reward.items()])
        new_step = new_total - self.total_reward

        self.total_reward = new_total
        return new_step

    def read_m(self, addr):
        return self.pyboy.get_memory_value(addr)

    def read_bit(self, addr, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))[-bit - 1] == "1"

    def read_event_bits(self):
        return [
            int(bit)
            for i in range(EVENT_FLAGS_START, EVENT_FLAGS_START + EVENTS_FLAGS_LENGTH)
            for bit in f"{self.read_m(i):08b}"
        ]

    def get_levels_reward(self):
        party_size = self.read_m(PARTY_SIZE)
        party_levels = [
            x for x in [self.read_m(addr) for addr in PARTY_LEVEL_ADDRS[:party_size]] if x > 0
        ]
        self.max_level_sum = max(self.max_level_sum, sum(party_levels))
        if self.max_level_sum < 30:
            return self.max_level_sum
        else:
            return 30 + (self.max_level_sum - 30) / 4
        # return 1.0 / (1 + 1000 * abs(max(party_levels) - self.max_opponent_level))

    def get_badges(self):
        return self.bit_count(self.read_m(0xD356))

    def read_party(self):
        return [self.read_m(addr) for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]]

    def get_all_events_reward(self):
        # adds up all event flags, exclude museum ticket
        return max(
            sum(
                [
                    self.bit_count(self.read_m(i))
                    for i in range(EVENT_FLAGS_START, EVENT_FLAGS_START + EVENTS_FLAGS_LENGTH)
                ]
            )
            - self.base_event_flags
            - int(self.read_bit(*MUSEUM_TICKET)),
            0,
        )

    def get_game_state_reward(self, print_stats=False):
        # addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm
        state_scores = {
            "event": 4 * self.update_max_event_rew(),
            "explore_npcs": sum(self.seen_npcs.values()) * 0.02,
            # "seen_pokemon": sum(self.seen_pokemon) * 0.000010,
            # "caught_pokemon": sum(self.caught_pokemon) * 0.000010,
            "moves_obtained": sum(self.moves_obtained) * 0.000010,
            "explore_hidden_objs": sum(self.seen_hidden_objs.values()) * 0.02,
            "level": self.get_levels_reward(),
            # "opponent_level": self.max_opponent_level,
            # "death_reward": self.died_count,
            "badge": self.get_badges() * 5,
            "heal": self.total_healing_rew,
            "explore": sum(self.seen_coords.values()) * 0.01,
            "explore_maps": np.sum(self.seen_map_ids) * 0.0001,
            "taught_cut": 4 * int(self.check_if_party_has_cut()),
            "cut_coords": sum(self.cut_coords.values()) * 0.001,
        }

        return state_scores

    def update_max_op_level(self):
        # opp_base_level = 5
        opponent_level = (
            max([self.read_m(a) for a in [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]])
            # - opp_base_level
        )
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level

    def update_max_event_rew(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew

    def update_heal_reward(self):
        cur_health = self.read_hp_fraction()
        # if health increased and party size did not change
        if cur_health > self.last_health and self.read_m(0xD163) == self.party_size:
            if self.last_health > 0:
                self.total_healing_rew += cur_health - self.last_health
            else:
                self.died_count += 1

    def update_pokedex(self):
        for i in range(0xD30A - 0xD2F7):
            caught_mem = self.pyboy.get_memory_value(i + 0xD2F7)
            seen_mem = self.pyboy.get_memory_value(i + 0xD30A)
            for j in range(8):
                self.caught_pokemon[8 * i + j] = 1 if caught_mem & (1 << j) else 0
                self.seen_pokemon[8 * i + j] = 1 if seen_mem & (1 << j) else 0

    def update_moves_obtained(self):
        # Scan party
        for i in [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247]:
            if self.pyboy.get_memory_value(i) != 0:
                for j in range(4):
                    move_id = self.pyboy.get_memory_value(i + j + 8)
                    if move_id != 0:
                        if move_id != 0:
                            self.moves_obtained[move_id] = 1
        # Scan current box (since the box doesn't auto increment in pokemon red)
        num_moves = 4
        box_struct_length = 25 * num_moves * 2
        for i in range(self.pyboy.get_memory_value(0xDA80)):
            offset = i * box_struct_length + 0xDA96
            if self.pyboy.get_memory_value(offset) != 0:
                for j in range(4):
                    move_id = self.pyboy.get_memory_value(offset + j + 8)
                    if move_id != 0:
                        self.moves_obtained[move_id] = 1

    def read_hp_fraction(self):
        hp_sum = sum(
            [self.read_hp(add) for add in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]]
        )
        max_hp_sum = sum(
            [self.read_hp(add) for add in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]]
        )
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum / max_hp_sum

    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start + 1)

    # built-in since python 3.10
    def bit_count(self, bits):
        return bin(bits).count("1")
        # return bits.bit_count()

    def fourier_encode(self, val):
        return np.sin(val * 2 ** np.arange(self.enc_freqs))

    def update_map_progress(self):
        map_idx = self.read_m(0xD35E)
        self.max_map_progress = max(self.max_map_progress, self.get_map_progress(map_idx))

    def get_map_progress(self, map_idx):
        if map_idx in self.essential_map_locations.keys():
            return self.essential_map_locations[map_idx]
        else:
            return -1

    def get_map_location(self, map_idx):
        map_locations = {
            0: {"name": "Pallet Town", "coordinates": np.array([70, 7])},
            1: {"name": "Viridian City", "coordinates": np.array([60, 79])},
            2: {"name": "Pewter City", "coordinates": np.array([60, 187])},
            3: {"name": "Cerulean City", "coordinates": np.array([240, 205])},
            62: {
                "name": "Invaded house (Cerulean City)",
                "coordinates": np.array([290, 227]),
            },
            63: {
                "name": "trade house (Cerulean City)",
                "coordinates": np.array([290, 212]),
            },
            64: {
                "name": "Pokémon Center (Cerulean City)",
                "coordinates": np.array([290, 197]),
            },
            65: {
                "name": "Pokémon Gym (Cerulean City)",
                "coordinates": np.array([290, 182]),
            },
            66: {
                "name": "Bike Shop (Cerulean City)",
                "coordinates": np.array([290, 167]),
            },
            67: {
                "name": "Poké Mart (Cerulean City)",
                "coordinates": np.array([290, 152]),
            },
            35: {"name": "Route 24", "coordinates": np.array([250, 235])},
            36: {"name": "Route 25", "coordinates": np.array([270, 267])},
            12: {"name": "Route 1", "coordinates": np.array([70, 43])},
            13: {"name": "Route 2", "coordinates": np.array([70, 151])},
            14: {"name": "Route 3", "coordinates": np.array([100, 179])},
            15: {"name": "Route 4", "coordinates": np.array([150, 197])},
            33: {"name": "Route 22", "coordinates": np.array([20, 71])},
            37: {"name": "Red house first", "coordinates": np.array([61, 9])},
            38: {"name": "Red house second", "coordinates": np.array([61, 0])},
            39: {"name": "Blues house", "coordinates": np.array([91, 9])},
            40: {"name": "oaks lab", "coordinates": np.array([91, 1])},
            41: {
                "name": "Pokémon Center (Viridian City)",
                "coordinates": np.array([100, 54]),
            },
            42: {
                "name": "Poké Mart (Viridian City)",
                "coordinates": np.array([100, 62]),
            },
            43: {"name": "School (Viridian City)", "coordinates": np.array([100, 79])},
            44: {"name": "House 1 (Viridian City)", "coordinates": np.array([100, 71])},
            47: {
                "name": "Gate (Viridian City/Pewter City) (Route 2)",
                "coordinates": np.array([91, 143]),
            },
            49: {"name": "Gate (Route 2)", "coordinates": np.array([91, 115])},
            50: {
                "name": "Gate (Route 2/Viridian Forest) (Route 2)",
                "coordinates": np.array([91, 115]),
            },
            51: {"name": "viridian forest", "coordinates": np.array([35, 144])},
            52: {"name": "Pewter Museum (floor 1)", "coordinates": np.array([60, 196])},
            53: {"name": "Pewter Museum (floor 2)", "coordinates": np.array([60, 205])},
            54: {
                "name": "Pokémon Gym (Pewter City)",
                "coordinates": np.array([49, 176]),
            },
            55: {
                "name": "House with disobedient Nidoran♂ (Pewter City)",
                "coordinates": np.array([51, 184]),
            },
            56: {"name": "Poké Mart (Pewter City)", "coordinates": np.array([40, 170])},
            57: {
                "name": "House with two Trainers (Pewter City)",
                "coordinates": np.array([51, 184]),
            },
            58: {
                "name": "Pokémon Center (Pewter City)",
                "coordinates": np.array([45, 161]),
            },
            59: {
                "name": "Mt. Moon (Route 3 entrance)",
                "coordinates": np.array([153, 234]),
            },
            60: {"name": "Mt. Moon Corridors", "coordinates": np.array([168, 253])},
            61: {"name": "Mt. Moon Level 2", "coordinates": np.array([197, 253])},
            68: {
                "name": "Pokémon Center (Route 3)",
                "coordinates": np.array([135, 197]),
            },
            193: {
                "name": "Badges check gate (Route 22)",
                "coordinates": np.array([0, 87]),
            },  # TODO this coord is guessed, needs to be updated
            230: {
                "name": "Badge Man House (Cerulean City)",
                "coordinates": np.array([290, 137]),
            },
        }
        if map_idx in map_locations.keys():
            return map_locations[map_idx]
        else:
            return {
                "name": "Unknown",
                "coordinates": np.array([80, 0]),
            }  # TODO once all maps are added this case won't be needed
