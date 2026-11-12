"""Strict assignment environments and exact offered MsPacman preprocessing."""
from collections import deque
import numpy as np

MSPACMAN_COLOR = 210 + 164 + 74


def preprocess_observation(obs):
    """The assignment's exact proposed MsPacman algorithm, returning int8 (88,80,1)."""
    obs = np.asarray(obs)
    if obs.shape != (210, 160, 3):
        raise ValueError(f'Expected RGB (210,160,3), got {obs.shape}')
    img = obs[1:176:2, ::2].astype(np.int16).sum(axis=2)
    img[img == MSPACMAN_COLOR] = 0
    img = (img // 3 - 128).astype(np.int8)
    return img.reshape(88, 80, 1)


def image_to_uint8(obs):
    # Lossless offset from assignment's [-128,127] int8 to [0,255].
    return (preprocess_observation(obs)[:, :, 0].astype(np.int16) + 128).astype(np.uint8)


class StateHistory:
    def __init__(self, history=4):
        self.history = history
        self.frames = deque(maxlen=history)

    def reset(self, obs):
        first = image_to_uint8(obs)
        self.frames = deque([first.copy() for _ in range(self.history)], maxlen=self.history)
        return self.state()

    def step(self, obs):
        self.frames.append(image_to_uint8(obs))
        return self.state()

    def state(self):
        return np.stack(list(self.frames)).astype(np.uint8)


def make_env(game, render_mode=None):
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise RuntimeError('Install requirements.txt first: missing gymnasium') from exc
    if game == 'cartpole':
        return gym.make('CartPole-v1', render_mode=render_mode)
    if game == 'mspacman':
        try:
            import ale_py
            gym.register_envs(ale_py)
            env = gym.make('ALE/MsPacman-v5', render_mode=render_mode)
        except Exception as exc:
            raise RuntimeError('MsPacman ROM unavailable. Verify ale-py and Gymnasium Atari installation.') from exc
        if env.action_space.n != 9:
            env.close()
            raise RuntimeError('Wrong game variant: expected nine actions')
        return env
    raise ValueError('Use cartpole or mspacman')


def make_state(game, obs, history=None, reset=False):
    if game == 'cartpole':
        return np.asarray(obs, dtype=np.float32)
    if reset:
        return history.reset(obs)
    return history.step(obs)
