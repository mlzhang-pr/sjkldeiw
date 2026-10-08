import numpy as np


shot_cool_down = 5
enemy_move_interval = 12
enemy_shot_interval = 10


class Env:
    def __init__(self, ramping=True, random_state=None, use_minimal_observation=True):
        if use_minimal_observation:
            self.channels = {
                "cannon": 0,
                "alien": 1,
                "alien_left": 2,
                "alien_right": 3,
                "friendly_bullet": 4,
                "enemy_bullet": 5,
            }
        else:
            self.channels = {
                "cannon": 0,
                "alien": 1,
                "alien_left": 2,
                "alien_right": 3,
                "friendly_bullet": 4,
                "enemy_bullet": 5,
                "misc_1": 6,
            }

        self.action_map = ["n", "l", "u", "r", "d", "f"]
        self.ramping = ramping
        if random_state is None:
            self.random = np.random.RandomState()
        else:
            self.random = random_state
        self.reset()

    def act(self, a):
        r = 0
        if self.terminal:
            return r, self.terminal

        a = self.action_map[a]

        if a == "f" and self.shot_timer == 0:
            self.f_bullet_map[9, self.pos] = 1
            self.shot_timer = shot_cool_down
        elif a == "l":
            self.pos = max(0, self.pos - 1)
        elif a == "r":
            self.pos = min(9, self.pos + 1)

        self.f_bullet_map = np.roll(self.f_bullet_map, -1, axis=0)
        self.f_bullet_map[9, :] = 0

        self.e_bullet_map = np.roll(self.e_bullet_map, 1, axis=0)
        self.e_bullet_map[0, :] = 0
        if self.e_bullet_map[9, self.pos]:
            self.terminal = True

        if self.alien_map[9, self.pos]:
            self.terminal = True
        if self.alien_move_timer == 0:
            self.alien_move_timer = min(
                np.count_nonzero(self.alien_map), self.enemy_move_interval
            )
            if (np.sum(self.alien_map[:, 0]) > 0 and self.alien_dir < 0) or (
                np.sum(self.alien_map[:, 9]) > 0 and self.alien_dir > 0
            ):
                self.alien_dir = -self.alien_dir
                if np.sum(self.alien_map[9, :]) > 0:
                    self.terminal = True
                self.alien_map = np.roll(self.alien_map, 1, axis=0)
            else:
                self.alien_map = np.roll(self.alien_map, self.alien_dir, axis=1)
            if self.alien_map[9, self.pos]:
                self.terminal = True
        if self.alien_shot_timer == 0:
            self.alien_shot_timer = enemy_shot_interval
            nearest_alien = self._nearest_alien(self.pos)
            self.e_bullet_map[nearest_alien[0], nearest_alien[1]] = 1

        kill_locations = np.logical_and(
            self.alien_map, self.alien_map == self.f_bullet_map
        )

        r += np.sum(kill_locations)
        self.alien_map[kill_locations] = self.f_bullet_map[kill_locations] = 0

        self.shot_timer -= self.shot_timer > 0
        self.alien_move_timer -= 1
        self.alien_shot_timer -= 1
        if np.count_nonzero(self.alien_map) == 0:
            if self.enemy_move_interval > 6 and self.ramping:
                self.enemy_move_interval -= 1
                self.ramp_index += 1
            self.alien_map[0:4, 2:8] = 1
        return r, self.terminal

    def _nearest_alien(self, pos):
        search_order = [i for i in range(10)]
        search_order.sort(key=lambda x: abs(x - pos))
        for i in search_order:
            if np.sum(self.alien_map[:, i]) > 0:
                return [np.max(np.where(self.alien_map[:, i] == 1)), i]
        return None

    def difficulty_ramp(self):
        return self.ramp_index

    def state(self):
        state = np.zeros((10, 10, len(self.channels)), dtype=bool)
        state[9, self.pos, self.channels["cannon"]] = 1
        state[:, :, self.channels["alien"]] = self.alien_map
        if self.alien_dir < 0:
            state[:, :, self.channels["alien_left"]] = self.alien_map
        else:
            state[:, :, self.channels["alien_right"]] = self.alien_map
        state[:, :, self.channels["friendly_bullet"]] = self.f_bullet_map
        state[:, :, self.channels["enemy_bullet"]] = self.e_bullet_map
        return state

    def reset(self):
        self.pos = 5
        self.f_bullet_map = np.zeros((10, 10))
        self.e_bullet_map = np.zeros((10, 10))
        self.alien_map = np.zeros((10, 10))
        self.alien_map[0:4, 2:8] = 1
        self.alien_dir = -1
        self.enemy_move_interval = enemy_move_interval
        self.alien_move_timer = self.enemy_move_interval
        self.alien_shot_timer = enemy_shot_interval
        self.ramp_index = 0
        self.shot_timer = 0
        self.terminal = False

    def state_shape(self):
        return [10, 10, len(self.channels)]

    def minimal_action_set(self):
        minimal_actions = ["n", "l", "r", "f"]
        return [self.action_map.index(x) for x in minimal_actions]
