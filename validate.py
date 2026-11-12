#!/usr/bin/env python3
"""Actual-environment checks, random-policy controls and reproducibility tests.

Complements analytical pytest tests; these run in Gymnasium, not an invented emulator.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from environment import StateHistory, make_env, make_state, preprocess_observation
from dqn import write_json


def rollout(game, seed, n, deterministic=False, max_steps=0):
    env = make_env(game)
    results = []
    rng = random.Random(seed)
    try:
        n_actions = env.action_space.n
        for ep in range(n):
            obs, info = env.reset(seed=seed + ep)
            hist = StateHistory() if game=='mspacman' else None
            s = make_state(game, obs, hist, True)
            first10 = []
            score, steps, capped = 0., 0, False
            action_hist = Counter()
            while True:
                action = steps % n_actions if deterministic else rng.randrange(n_actions)
                obs2, reward, terminated, truncated, info2 = env.step(action)
                next_s = make_state(game, obs2, hist)
                steps += 1; score += float(reward); action_hist[action] += 1
                if steps <= 10:
                    first10.append(dict(step=steps, action=action, reward=float(reward),
                                        terminated=terminated, truncated=truncated,
                                        state_shape=list(s.shape),
                                        next_state_diff_mean=float(np.mean(np.abs(next_s.astype('float32')-s.astype('float32'))))))
                s = next_s
                capped = bool(max_steps and steps >= max_steps and not (terminated or truncated))
                if terminated or truncated or capped:
                    break
            results.append(dict(episode=ep+1, reward=score, steps=steps, capped=capped,
                                action_counts=dict(action_hist), first10=first10 if ep==0 else []))
    finally:
        env.close()
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--game', choices=['cartpole','mspacman'], required=True)
    p.add_argument('--random-episodes', type=int, default=30)
    p.add_argument('--seed', type=int, default=713)
    p.add_argument('--max-steps', type=int, default=0)
    p.add_argument('--out', default=None)
    args=p.parse_args()
    data=rollout(args.game,args.seed,args.random_episodes, max_steps=args.max_steps)
    results=np.array([row['reward'] for row in data],float)
    # Deterministic reset+fixed action sequence should be reproducible given same ALE seed.
    test_a=rollout(args.game,991,1,deterministic=True,max_steps=500)
    test_b=rollout(args.game,991,1,deterministic=True,max_steps=500)
    replay_ok=(test_a[0]['reward']==test_b[0]['reward'] and test_a[0]['steps']==test_b[0]['steps']
               and test_a[0]['first10']==test_b[0]['first10'])
    summary=dict(game=args.game, random_episodes=len(data), random_mean=float(results.mean()),
                 random_sample_std=float(results.std(ddof=1)) if len(data)>1 else 0.,
                 random_min=float(results.min()), random_max=float(results.max()),
                 random_episode_capped_count=sum(z['capped'] for z in data),
                 deterministic_repeatability_test_passed=replay_ok,
                 first10=data[0]['first10'],
                 random_episode_records=[{k:v for k,v in row.items() if k!='first10'} for row in data])
    out=Path(args.out or f'runs/validation_{args.game}.json')
    write_json(out,summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='random_episode_records'}, indent=2))
    print(f'Saved {out}')

if __name__=='__main__':
    main()
