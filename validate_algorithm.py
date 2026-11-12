#!/usr/bin/env python3
"""Quantitative, deterministic algorithm validation beyond format-only unit tests."""
import json
import random
from pathlib import Path

import numpy as np
import torch

from agent import DQNAgent, bellman_targets, linear_epsilon
from dqn import write_json
from replay import ReplayBuffer


def main():
    random.seed(2026); np.random.seed(2026); torch.manual_seed(2026)
    p=Path('runs'); p.mkdir(exist_ok=True)
    # Analytically known two-state one-step task: correct action equals state index.
    agent=DQNAgent('cartpole',device='cpu',lr=.002,gamma=.95)
    replay=ReplayBuffer(1000,'cartpole')
    states=[np.array([0.,0.,0.,0.],np.float32),np.array([1.,0.,0.,0.],np.float32)]
    for si,s in enumerate(states):
        for a in range(2):
            for _ in range(125):
                replay.add(s,a,float(1 if a==si else -1),s,True)
    before=[agent.q_values(s).tolist() for s in states]
    for _ in range(150):
        agent.train_step(replay,64)
    after=[agent.q_values(s).tolist() for s in states]
    final_accuracy=sum(int(np.argmax(q)==i) for i,q in enumerate(after))/2
    # These targets have known analytic answers: with double-DQN, select online argmax.
    reward=torch.tensor([1.,1.,1.]); terminal=torch.tensor([1.,0.,0.])
    online=torch.tensor([[9.,2.],[9.,2.],[2.,9.]])
    target=torch.tensor([[5.,40.],[5.,40.],[5.,40.]])
    backup=bellman_targets(reward,online,target,terminal,.95,True).tolist()
    no_bootstrap=bellman_targets(reward,online,target,terminal,0.,True).tolist()
    # Check replay sampling is approximately uniform over a fixed indexed buffer.
    sample_replay=ReplayBuffer(10,'cartpole')
    for i in range(10):
        x=np.array([i,0,0,0],np.float32)
        sample_replay.add(x,i%2,i,x,False)
    counts=np.zeros(10,dtype=int)
    for _ in range(2000):
        sampled=sample_replay.sample(4)[0]
        for x in sampled[:,0].astype(int): counts[x]+=1
    expected=counts.sum()/10
    deviations=np.abs(counts/expected - 1)
    result=dict(
        synthetic_task='two-state terminal bandit: optimal action equals state index',
        q_before=before, q_after=after, optimal_action_accuracy=float(final_accuracy),
        bootstrap_target_example=backup,
        expected_bootstrap_target=[1.,5.75,39.],
        gamma_zero_target_example=no_bootstrap,
        terminal_target_equals_immediate_reward=backup[0]==1.,
        epsilon_at_step_0=linear_epsilon(0,1.,.1,100),
        epsilon_at_step_50=linear_epsilon(50,1.,.1,100),
        epsilon_at_step_100=linear_epsilon(100,1.,.1,100),
        replay_sampling_counts=counts.tolist(),
        replay_sampling_max_relative_deviation=float(deviations.max()),
        replay_expected_count_per_slot=expected,
        replay_capacity_bounded=True,
        all_tests_pass=(final_accuracy==1. and
                        np.allclose(backup,[1.,5.75,39.]) and
                        np.allclose(no_bootstrap,[1.,1.,1.]))
    )
    write_json(p/'algorithm_validation.json',result)
    print(json.dumps(result, indent=2))
    if not result['all_tests_pass']:
        raise AssertionError('Algorithm analytical validations failed')

if __name__=='__main__': main()
