import sys
from contextlib import closing

import numpy as np
from io import StringIO
import gymnasium as gym
from gymnasium import utils, spaces
from gymnasium.envs.toy_text.utils import categorical_sample

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAP = [
    "FFFFFFFFFFFF",
    "FFFFFFFFFFFF",
    "FFFFFFFFFFFF",
    "SHHHHHHHHHHG",
]


class CliffEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "ansi"],
        "render_fps": 1,  # Avoid warning about fps not set.
    }

    def __init__(
        self,
        render_mode: str = "human",
    ):
        self.desc = desc = np.asarray(MAP, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (-100, -1)

        self.render_mode = render_mode

        nA = 4
        nS = nrow * ncol

        self.action_space = spaces.Discrete(nA)
        self.observation_space = spaces.Discrete(nS)

        self.initial_state_distribution = (
            np.array(desc == b"S").astype("float64").ravel()
        )
        self.initial_state_distribution /= self.initial_state_distribution.sum()

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]

            # only modification from frozenlake, other than cosmetics
            done = False
            if newletter == b"G":
                reward = -1
                done = True
            elif newletter == b"H":
                reward = -100
                done = True
            else:
                reward = -1
            return newstate, reward, done

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = desc[row, col]
                    if letter in b"GH":
                        # if terminal, stay in the same state with p==1
                        # important for DP methods
                        li.append((1.0, s, 0, True))
                    else:
                        # no non-determinism in 'the cliff' env
                        li.append((1.0, *update_probability_matrix(row, col, a)))

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.s = categorical_sample(
            prob_n=self.initial_state_distribution,
            np_random=self.np_random,
        )
        self.lastaction = None

        return self.s, {}

    def step(self, action):
        transitions = self.P[self.s][action]
        i = categorical_sample(
            prob_n=[t[0] for t in transitions],
            np_random=self.np_random,
        )
        prob, newstate, reward, done = transitions[i]
        self.s = newstate
        self.lastaction = action
        terminated = done
        truncated = False
        info = {"prob": prob}
        return self.s, reward, terminated, truncated, info

    def render(self):

        mode = self.render_mode
        outfile = StringIO() if mode == "ansi" else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(
                "  ({})\n".format(["Left", "Down", "Right", "Up"][self.lastaction])
            )
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        if mode != "human":
            with closing(outfile):
                return outfile.getvalue()
