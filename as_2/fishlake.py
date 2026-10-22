import sys
from contextlib import closing

import numpy as np
from io import StringIO

import gymnasium as gym
from gymnasium import spaces, utils
from gymnasium.envs.toy_text.utils import categorical_sample
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib import patches
from gymnasium.envs.registration import register
from typing import Optional, List

register(
    id="FishlakeCalm-v0",
    entry_point="fishlake:FishlakeEnv",
    kwargs={
        "map_name": "default",
        "streamy": False,
    },
    max_episode_steps=100,
    reward_threshold=0.70,  # optimum = 0.74
)

register(
    id="FishlakeStreamy-v0",
    entry_point="fishlake:FishlakeEnv",
    kwargs={
        "map_name": "default",
        "streamy": True,
    },
    max_episode_steps=100,
    reward_threshold=0.70,  # optimum = 0.74
)

register(
    id="FishlakeCalmLarger-v0",
    entry_point="fishlake:FishlakeEnv",
    kwargs={
        "map_name": "larger",
        "streamy": False,
    },
    max_episode_steps=100,
    reward_threshold=0.70,  # optimum = 0.74
)

register(
    id="FishlakeStreamyLarger-v0",
    entry_point="fishlake:FishlakeEnv",
    kwargs={
        "map_name": "larger",
        "streamy": True,
    },
    max_episode_steps=100,
    reward_threshold=0.70,  # optimum = 0.74
)


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "default": [
        "WWWM",
        "WRWF",
        "SWWW",
    ],
    "larger": [
        "MWWWWWWWWWWWWM",
        "WFWWWWRWWWWWFW",
        "WWWWRRRRRWWWWW",
        "WWWWWWRWWWWWWW",
        "WWWWWSSSWWWWWW",
    ],
}


def perpendicular(a: int) -> List[int]:
    if a == UP or a == DOWN:
        return [LEFT, RIGHT]
    elif a == LEFT or a == RIGHT:
        return [UP, DOWN]
    else:
        raise ValueError()


class FishlakeEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "ansi"],
        "render_fps": 1,  # Avoid warning about fps not set.
    }

    def __init__(
        self,
        desc: Optional[List[str]] = None,
        map_name: str = "default",
        streamy: bool = False,
        render_mode: Optional[str] = None,
    ) -> None:
        if desc is None:
            desc = MAPS[map_name]
        else:
            raise ValueError()
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (-1, 1)

        nA = 4
        self.nA = nA

        s2rc = dict()
        rc2s = dict()
        state = 0
        P = dict()
        for row in range(self.nrow):
            for col in range(self.ncol):
                letter = self.desc[row, col]
                if letter != b"R":
                    s2rc[state] = (row, col)
                    rc2s[(row, col)] = state
                    P[state] = {a: list() for a in [LEFT, RIGHT, UP, DOWN]}
                    state += 1

        state = None
        nS = len(s2rc)
        self.nS = nS

        # Compute initial state distribution over valid states
        initial_state_distribution = np.zeros(self.nS)
        for state, (row, col) in s2rc.items():
            if self.desc[row, col] == b"S":
                initial_state_distribution[state] = 1.0
        initial_state_distribution /= initial_state_distribution.sum()
        self.initial_state_distribution = initial_state_distribution

        def inc(row, col, a):
            if a == LEFT:
                col = col - 1
            elif a == DOWN:
                row = row + 1
            elif a == RIGHT:
                col = col + 1
            elif a == UP:
                row = row - 1
            return (row, col)

        def upm(row, col, action):
            newrow, newcol = inc(row, col, action)
            if (newrow, newcol) not in rc2s:
                newrow, newcol = row, col

            newstate = rc2s[(newrow, newcol)]
            newletter = desc[newrow, newcol]
            done = bytes(newletter) in b"MF"

            if newletter == b"M":
                reward = 1
            elif newletter == b"F":
                reward = -1
            else:
                reward = 0
            return newstate, reward, done

        for state, (row, col) in s2rc.items():
            letter = self.desc[row, col]
            for action in [LEFT, RIGHT, UP, DOWN]:
                li = P[state][action]
                # probability, newstate, reward, done
                if letter in b"MF":
                    li.append((1.0, state, 0, True))
                else:
                    if streamy:
                        li.append((0.8, *upm(row, col, action)))
                        for perp in perpendicular(action):
                            li.append((0.1, *upm(row, col, perp)))
                    else:
                        li.append((1.0, *upm(row, col, action)))

        self.P = P
        self.s2rc = s2rc
        self.rc2s = rc2s

        self.action_space = spaces.Discrete(nA)
        self.observation_space = spaces.Discrete(nS)

        self.lastaction = None  # for rendering
        self.render_mode = render_mode
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
        if mode == "ansi":
            outfile = StringIO()
        else:
            outfile = sys.stdout

        row, col = self.s2rc[self.s]
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

        if mode == "ansi":
            return outfile.getvalue()


def plot_policy(ax, env, policy):
    def a2dxdy(a, p):
        d = 0.5 * p
        if a == LEFT:
            return -d, 0
        if a == DOWN:
            return 0, d
        if a == RIGHT:
            return d, 0
        if a == UP:
            return 0, -d

    for state, pis in enumerate(policy):
        row, col = env.s2rc[state]
        if env.desc[row, col] not in b"MF":
            for action, probability in enumerate(pis):
                if probability > 0:
                    arrow = patches.Arrow(
                        col, row, *a2dxdy(action, probability), width=0.1, color="k"
                    )
                    ax.add_patch(arrow)

    ax.set_xlim([0.5, env.ncol - 0.5])
    ax.set_ylim([0.5, env.nrow - 0.5])

    # Major ticks
    sx = env.ncol
    sy = env.nrow
    ax.set_xticks(np.arange(0, sx, 1))
    ax.set_yticks(np.arange(0, sy, 1))

    # Minor ticks
    ax.set_xticks(np.arange(-0.5, sx, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, sy, 1), minor=True)

    ax.grid(which="minor", color="gray", linestyle="-", linewidth=1)

    ax.invert_yaxis()
    ax.set_aspect(1)


def plot_value_function(ax, env, V):
    cmap = plt.get_cmap("magma")
    max_value = max(env.reward_range)
    for state, value in enumerate(V):
        row, col = env.s2rc[state]
        xy_center = (col, row)
        xy_ll_corner = (col - 0.5, row - 0.5)
        color = cmap(value)
        rect = patches.Rectangle(
            xy_ll_corner,
            1,
            1,
            fill=True,
            facecolor=color,
        )
        ax.add_patch(rect)
        ax.annotate(
            f"{value:3.2g}",
            xy_center,
            va="center",
            ha="center",
            color=cmap(max_value - value),
            fontsize=8,
        )

    ax.set_xlim([0.5, env.ncol - 0.5])
    ax.set_ylim([0.5, env.nrow - 0.5])

    # Major ticks
    sx = env.ncol
    sy = env.nrow
    ax.set_xticks(np.arange(0, sx, 1))
    ax.set_yticks(np.arange(0, sy, 1))

    # Minor ticks
    ax.set_xticks(np.arange(-0.5, sx, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, sy, 1), minor=True)

    ax.grid(which="minor", color="gray", linestyle="-", linewidth=1)

    ax.invert_yaxis()
    ax.set_aspect(1)


def plot_vf_lines(ax, env, V, color="tab:blue"):
    xs = np.arange(len(V))
    ax.vlines(xs, np.zeros(len(V)), V, label="$v_{\\pi}$", color=color)
    ax.scatter(xs, V, color=color, s=5)

    ax.set_ylabel("$v_{\\pi}(s)$")
    ax.set_xticks(xs)
    ax.set_xticklabels(["$s_{" + str(i) + "}$" for i in range(len(V))])
    ax.legend()
