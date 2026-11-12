"""Bounded uniform replay buffer; compress image stacks to avoid gigabytes of RAM."""
import random
import zlib
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, game):
        if capacity < 1:
            raise ValueError('Replay capacity must be positive')
        self.capacity = int(capacity)
        self.game = game
        self.data = [None] * self.capacity
        self.pos = 0
        self.size = 0

    def __len__(self):
        return self.size

    def _encode(self, state):
        if self.game == 'mspacman':
            a = np.asarray(state, dtype=np.uint8)
            if a.shape != (4, 88, 80):
                raise ValueError(f'Expected four 88x80 frames, got {a.shape}')
            return zlib.compress(a.tobytes(), level=1)
        a = np.asarray(state, dtype=np.float32)
        if a.shape != (4,):
            raise ValueError(f'Expected CartPole 4-vector, got {a.shape}')
        return a.copy()

    def _decode(self, state):
        if self.game == 'mspacman':
            return np.frombuffer(zlib.decompress(state), dtype=np.uint8).reshape(4, 88, 80).copy()
        return state.copy()

    def add(self, state, action, reward, next_state, terminated, truncated=False):
        # truncated is recorded for audit, but does NOT zero Bellman bootstrap.
        self.data[self.pos] = (self._encode(state), int(action), float(reward),
                               self._encode(next_state), bool(terminated), bool(truncated))
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, batch_size):
        if self.size < batch_size:
            raise ValueError('Not enough transitions in replay buffer')
        indices = random.sample(range(self.size), batch_size)
        rows = [self.data[i] for i in indices]
        states = np.stack([self._decode(row[0]) for row in rows])
        next_states = np.stack([self._decode(row[3]) for row in rows])
        actions = np.array([row[1] for row in rows], dtype=np.int64)
        rewards = np.array([row[2] for row in rows], dtype=np.float32)
        terminals = np.array([row[4] for row in rows], dtype=np.float32)
        return states, actions, rewards, next_states, terminals

    def snapshot(self):
        return dict(capacity=self.capacity, game=self.game, data=self.data,
                    pos=self.pos, size=self.size)

    @classmethod
    def from_snapshot(cls, snap):
        obj = cls(snap['capacity'], snap['game'])
        obj.data = snap['data']
        obj.pos = snap['pos']
        obj.size = snap['size']
        return obj
