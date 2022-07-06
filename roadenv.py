from typing import Optional

import numpy as np
from gym import Env, spaces

import zuobiao

Sequence = 50
Lane = 21


def generate_random_map(sequences=Sequence, lanes=Lane):
    res = np.zeros((sequences, lanes), dtype=np.int64)
    res[0][10] = 0
    res[-1][10] = 2
    res[15][16] = res[30][5] = 1
    return res


class roadenv(Env):
    def __init__(
            self,
            desc=None,
    ):
        self.desc = desc = generate_random_map()
        x_len, y_len = desc.shape
        n_action = y_len
        n_observation = x_len * y_len
        self.c1 = 5
        self.c2 = 3
        self.initial_state = 6
        self.obstacals = [[]]
        self.prob = {o: {a: [] for a in range(n_action)} for o in range(n_observation)}

        def to_state(x, y):
            return x * y_len + y

        def inc(x, y, a):
            if x != x_len - 1:
                x += 1
            y = a
            return x, y

        def distance(x0, y0):
            obstac_x = np.array([2 * 15, 2 * 30])
            obstac_y = np.array([6 * 0.2, -5 * 0.2])
            obstac_y += np.sin(0.04 * np.pi * obstac_x)
            dis = np.array(np.sqrt(np.power(obstac_x - x0, 2) + np.power(obstac_y - y0, 2)))
            return dis.min()

        def update_probability_matrix(x, y, action):
            newx, newy = inc(x, y, action)
            newstate = to_state(newx, newy)
            if desc[newx][newy] == 2:
                done = True
                reward = 1.0
                cost = -1
            elif desc[newx][newy] == 1:
                done = True
                reward = 0.0
                cost = 47
            else:
                done = False
                reward = 0.0
                d1 = distance(2 * newx, 0.2 * (newy - 10))
                s, d2 = zuobiao.cartesian_to_frenet(2 * newx, 0.2 * (newy - 10))
                cost = self.c1 / d1 + self.c2 * d2
            return newstate, reward, cost, done

        for x in range(x_len):
            for y in range(y_len):
                state = to_state(x, y)
                for action in range(n_action):
                    li = self.prob[state][action]
                    li.append((1.0, *update_probability_matrix(x, y, action)))

        self.observation_space = spaces.Discrete(n_observation)
        self.action_space = spaces.Discrete(n_action)

    def step(self, a):
        transitions = self.prob[self.s][a][0]
        p, next_s, r, c, d = transitions
        self.s = next_s
        self.lastaction = a
        return next_s, r, c, d, {"prob": p}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.s = self.initial_state
        self.lastaction = None
        return self.s
