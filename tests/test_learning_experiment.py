"""Behavioral algorithm validation: can DQN learn correct actions from reward?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from agent import DQNAgent
from replay import ReplayBuffer
from plotting import moving_average


def test_learns_optimal_two_state_terminal_bandit():
    # Each state has different optimal action, no helpful next-state bootstraps.
    torch.manual_seed(42)
    agent=DQNAgent('cartpole',device='cpu',lr=0.002,gamma=.95)
    replay=ReplayBuffer(800,'cartpole')
    states=[np.array([0.,0.,0.,0.],dtype='float32'),np.array([1.,0.,0.,0.],dtype='float32')]
    for s_idx, s in enumerate(states):
        for action in (0,1):
            reward=1.0 if action==s_idx else -1.0
            for _ in range(100):
                replay.add(s,action,reward,s,True)
    for _ in range(120):
        agent.train_step(replay,64)
    for state_idx, s in enumerate(states):
        q=agent.q_values(s)
        assert int(q.argmax()) == state_idx
        assert q[state_idx] > q[1-state_idx] + 1.0


def test_moving_average_is_trailing_not_centered():
    assert np.allclose(moving_average([1,2,3,4],width=2),[1,1.5,2.5,3.5])
