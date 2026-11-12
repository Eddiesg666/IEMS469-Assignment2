#!/usr/bin/env python3
"""Assignment 2: DQN with online/target networks, replay, resume and 500-game evaluation.

Examples:
  python dqn.py train --game cartpole --episodes 800 --eval-episodes 500
  python dqn.py train --game mspacman --episodes 5000 --eval-episodes 500
  python dqn.py evaluate --game mspacman --out runs/mspacman --episodes 500
"""
import argparse
import copy
import csv
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from agent import DQNAgent, linear_epsilon
from environment import StateHistory, make_env, make_state
from plotting import evaluation_plot, training_plots
from replay import ReplayBuffer


def defaults(game):
    return dict(
        gamma=.95 if game == 'cartpole' else .99,
        lr=5e-4 if game == 'cartpole' else 1e-4,
        episodes=800 if game == 'cartpole' else 5000,
        batch_size=64 if game == 'cartpole' else 32,
        replay_capacity=30000,
        warmup=1000 if game == 'cartpole' else 10000,
        train_freq=1 if game == 'cartpole' else 4,
        target_sync_steps=500 if game == 'cartpole' else 5000,
        epsilon_end=.05 if game == 'cartpole' else .10,
        epsilon_decay_steps=15000 if game == 'cartpole' else 650000,
        reward_clip='none' if game == 'cartpole' else 'sign',
        out=f'runs/{game}',
    )


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
    os.replace(tmp, path)


def write_csv(path, records):
    if not records:
        return
    path = Path(path)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def atomic_torch_save(payload, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_trusted_checkpoint(path, device='cpu'):
    # Only load locally created checkpoints you trust: torch.load uses pickle.
    return torch.load(path, map_location=device, weights_only=False)


def config_for(args):
    d = defaults(args.game)
    for key in d:
        if hasattr(args, key):
            x = getattr(args, key)
            if x is not None:
                d[key] = x
    d.update(game=args.game, seed=args.seed, device=args.device,
             epsilon_start=args.epsilon_start, eval_epsilon=args.eval_epsilon,
             eval_episodes=args.eval_episodes, checkpoint_every=args.checkpoint_every,
             q_sample_freq=args.q_sample_freq, max_steps=args.max_steps,
             double=not args.no_double)
    if d['epsilon_decay_steps'] <= 0:
        raise ValueError('epsilon_decay_steps must be positive')
    if d['batch_size'] > d['replay_capacity']:
        raise ValueError('batch_size must be <= replay_capacity')
    return d


def safe_load_agent(agent, state):
    agent.load_model_state(state)
    for group in agent.optimizer.state.values():
        for k, v in group.items():
            if isinstance(v, torch.Tensor):
                group[k] = v.to(agent.device)


def save_progress(out, cfg, agent, buffer, history, episode, global_step, best_mean):
    snap = {'config': cfg, 'agent': agent.model_state(), 'replay': buffer.snapshot(),
            'history': history, 'episode': episode, 'global_step': global_step,
            'best_mean': best_mean, 'random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'torch_random_state': torch.get_rng_state()}
    atomic_torch_save(snap, out / 'checkpoint_latest.pt')
    write_json(out / 'train_history.json', history)
    write_csv(out / 'train_episodes.csv', history)
    training_plots(history, cfg['game'], out)


def make_agent(cfg):
    return DQNAgent(cfg['game'], cfg['device'], cfg['lr'], cfg['gamma'], cfg['double'])


def train(args):
    cfg = config_for(args)
    out = Path(cfg['out']); out.mkdir(parents=True, exist_ok=True)
    set_all_seeds(cfg['seed'])
    agent = make_agent(cfg)
    env = make_env(cfg['game'])
    buffer = ReplayBuffer(cfg['replay_capacity'], cfg['game'])
    history, global_step, first_episode, best_mean = [], 0, 0, float('-inf')
    if args.resume:
        path = out / 'checkpoint_latest.pt'
        if not path.exists():
            raise FileNotFoundError(f'Cannot resume: {path} does not exist')
        snap = load_trusted_checkpoint(path)
        if snap['config']['game'] != cfg['game']:
            raise ValueError('Checkpoint environment does not match')
        if snap['config']['replay_capacity'] != cfg['replay_capacity']:
            raise ValueError('Replay capacity must match checkpoint')
        # All learning hyperparameters are inherited from original checkpoint on resume.
        for k in ['gamma', 'lr', 'batch_size', 'warmup', 'train_freq',
                  'target_sync_steps', 'epsilon_start', 'epsilon_end',
                  'epsilon_decay_steps', 'reward_clip', 'double', 'q_sample_freq']:
            if cfg[k] != snap['config'][k]:
                raise ValueError(f'Resume setting {k} differs; use same settings as initial run')
        safe_load_agent(agent, snap['agent'])
        buffer = ReplayBuffer.from_snapshot(snap['replay'])
        history, global_step = snap['history'], snap['global_step']
        first_episode, best_mean = snap['episode'], snap['best_mean']
        random.setstate(snap['random_state'])
        np.random.set_state(snap['numpy_random_state'])
        torch.set_rng_state(snap['torch_random_state'])
        print(f'Resumed {first_episode}/{cfg["episodes"]} episodes, step {global_step}', flush=True)
    print(f'Environment={cfg["game"]}, device={agent.device}, '
          f'double DQN={cfg["double"]}, replay={cfg["replay_capacity"]}, '
          f'gamma={cfg["gamma"]}, seed={cfg["seed"]}', flush=True)
    write_json(out / 'config.json', cfg)
    try:
        for ep in range(first_episode, cfg['episodes']):
            obs, _ = env.reset(seed=cfg['seed'] + ep)
            hs = StateHistory() if cfg['game'] == 'mspacman' else None
            state = make_state(cfg['game'], obs, hs, reset=True)
            total_reward, length, capped = 0.0, 0, False
            max_q_values, q_gaps, td_samples = [], [], []
            action_counts = Counter()
            trace = []
            while True:
                epsilon = linear_epsilon(global_step, cfg['epsilon_start'],
                                         cfg['epsilon_end'], cfg['epsilon_decay_steps'])
                if length % cfg['q_sample_freq'] == 0:
                    qs = agent.q_values(state)
                    max_q_values.append(float(np.max(qs)))
                    q_gaps.append(float(np.partition(qs, -2)[-1] - np.partition(qs, -2)[-2]))
                action = agent.act(state, epsilon)
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_state = make_state(cfg['game'], next_obs, hs)
                store_reward = float(np.sign(reward) if cfg['reward_clip'] == 'sign' else reward)
                # Gym time-limit truncations are recorded but not treated as MDP terminations.
                buffer.add(state, action, store_reward, next_state, terminated, truncated)
                if length < 10 and ep == first_episode:
                    trace.append(dict(step=length + 1, action=action, raw_reward=float(reward),
                                      qmax=float(np.max(agent.q_values(state))),
                                      state=(state.astype(float).round(4).tolist() if cfg['game']=='cartpole'
                                             else {'image_nonzero':int(np.count_nonzero(state[-1])),
                                                   'frame_difference_pixels':int(np.count_nonzero(state[-1]!=state[-2]))}),
                                      epsilon=epsilon, terminated=bool(terminated),
                                      truncated=bool(truncated)))
                state = next_state
                total_reward += float(reward)
                action_counts[action] += 1
                length += 1
                global_step += 1
                if len(buffer) >= max(cfg['warmup'], cfg['batch_size']) and global_step % cfg['train_freq'] == 0:
                    metrics = agent.train_step(buffer, cfg['batch_size'])
                    td_samples.append(metrics)
                if global_step % cfg['target_sync_steps'] == 0:
                    agent.sync_target()
                if cfg['max_steps'] and length >= cfg['max_steps'] and not (terminated or truncated):
                    capped = True
                if terminated or truncated or capped:
                    break
            recent = [r['reward'] for r in history[-99:]] + [total_reward]
            avg100 = float(np.mean(recent))
            if ep + 1 >= 100 and avg100 > best_mean + (0.5 if cfg['game'] == 'cartpole' else 1.0):
                best_mean = avg100
                # Model selection uses training rewards, not evaluation episodes.
                atomic_torch_save({'game': cfg['game'], 'episode': ep + 1,
                    'rolling_train_mean': best_mean,
                    'online': {k: v.detach().cpu().clone() for k, v in agent.online.state_dict().items()}},
                    out / 'best_policy.pt')
            def td_mean(key):
                return float(np.mean([m[key] for m in td_samples])) if td_samples else None
            row = dict(episode=ep+1, reward=total_reward, steps=length, capped=capped,
                       epsilon_end=epsilon, mean_sampled_max_q=float(np.mean(max_q_values)),
                       max_sampled_q=float(np.max(max_q_values)),
                       mean_q_gap=float(np.mean(q_gaps)), replay_size=len(buffer),
                       updates=agent.updates, mean_td_loss=td_mean('loss'),
                       td_p10=td_mean('td_abs_p10'), td_p50=td_mean('td_abs_p50'),
                       td_p90=td_mean('td_abs_p90'),
                       action_counts=dict(action_counts), trace_first10=trace)
            history.append(row)
            if (ep+1) % (10 if cfg['game'] == 'cartpole' else 20) == 0 or ep == first_episode:
                print(f'  Episode {ep+1}/{cfg["episodes"]}: reward={total_reward:.1f} '
                      f'avg100={avg100:.2f}, maxQ={row["max_sampled_q"]:.2f}, '
                      f'eps={epsilon:.3f}, steps={length}, updates={agent.updates}', flush=True)
            if (ep+1) % cfg['checkpoint_every'] == 0 or ep+1 == cfg['episodes']:
                save_progress(out, cfg, agent, buffer, history, ep+1, global_step, best_mean)
                print(f'  Saved checkpoint at episode {ep+1}', flush=True)
    except KeyboardInterrupt:
        print('Interrupted. Resume from most recent completed checkpoint using --resume.', flush=True)
        raise
    finally:
        env.close()
    print('Training complete; running evaluation...', flush=True)
    checkpoint = out / 'best_policy.pt'
    if not checkpoint.exists():
        checkpoint = out / 'checkpoint_latest.pt'
    evaluate_policy(cfg['game'], checkpoint, out, cfg['eval_episodes'], cfg['eval_epsilon'], cfg['seed'])


def evaluate_policy(game, checkpoint, out, episodes, epsilon, seed):
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    data = load_trusted_checkpoint(checkpoint)
    config = data.get('config', {})
    resolved = select_eval_device(config)
    agent = DQNAgent(game, device=resolved, lr=1e-4)
    if 'agent' in data:
        agent.online.load_state_dict(data['agent']['online'])
    else:
        if data['game'] != game:
            raise ValueError('Evaluation checkpoint game mismatch')
        agent.online.load_state_dict(data['online'])
    agent.online.eval()
    env = make_env(game)
    rewards, records = [], []
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + 100000 + ep)
            hist = StateHistory() if game == 'mspacman' else None
            state = make_state(game, obs, hist, reset=True)
            score, steps = 0.0, 0
            actions = Counter()
            while True:
                a = agent.act(state, epsilon)
                obs, r, terminated, truncated, _ = env.step(a)
                score += float(r)
                actions[a] += 1
                steps += 1
                if terminated or truncated:
                    break
                state = make_state(game, obs, hist)
            rewards.append(score)
            records.append(dict(episode=ep+1, reward=score, steps=steps, action_counts=dict(actions)))
            if (ep+1) % 25 == 0 or ep+1 == episodes:
                print(f'  Evaluation completed {ep+1}/{episodes}', flush=True)
    finally:
        env.close()
    r = np.asarray(rewards, dtype=float)
    mean = float(r.mean()); sd = float(r.std(ddof=1)) if len(r)>1 else 0.0
    se = sd / math.sqrt(len(r))
    stats = dict(game=game, checkpoint=str(checkpoint), eval_episode_count=len(r),
                 eval_epsilon=epsilon, eval_mean=mean, eval_std_sample=sd,
                 eval_ci95_normal=[mean-1.96*se, mean+1.96*se],
                 eval_min=float(r.min()), eval_max=float(r.max()),
                 eval_rewards=rewards,
                 action_counts=dict(sum((Counter(z['action_counts']) for z in records), Counter())))
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'evaluation_metrics.json', stats)
    write_csv(out / 'evaluation_episodes.csv', records)
    evaluation_plot(rewards, game, out)
    print(json.dumps({k:v for k,v in stats.items() if k!='eval_rewards'}, indent=2))
    return stats


def select_eval_device(config):
    from agent import select_device
    return select_device(config.get('device', 'auto'))


def cmd_evaluate(args):
    out = Path(args.out or f'runs/{args.game}')
    ckpt = Path(args.checkpoint) if args.checkpoint else (out / 'best_policy.pt')
    if not ckpt.exists():
        ckpt = out / 'checkpoint_latest.pt'
    eps = args.epsilon if args.epsilon is not None else (0.0 if args.game == 'cartpole' else .01)
    evaluate_policy(args.game, ckpt, out, args.episodes, eps, args.seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    tr = sub.add_parser('train', help='Train and then evaluate a DQN model')
    tr.add_argument('--game', choices=['cartpole','mspacman'], required=True)
    tr.add_argument('--seed', type=int, default=2026)
    tr.add_argument('--device', default='auto')
    tr.add_argument('--episodes', type=int, default=None)
    tr.add_argument('--eval-episodes', type=int, default=500)
    tr.add_argument('--eval-epsilon', type=float, default=None)
    tr.add_argument('--out', default=None)
    tr.add_argument('--gamma', type=float, default=None)
    tr.add_argument('--lr', type=float, default=None)
    tr.add_argument('--batch-size', type=int, default=None)
    tr.add_argument('--replay-capacity', type=int, default=None)
    tr.add_argument('--warmup', type=int, default=None)
    tr.add_argument('--train-freq', type=int, default=None)
    tr.add_argument('--target-sync-steps', type=int, default=None)
    tr.add_argument('--epsilon-start', type=float, default=1.0)
    tr.add_argument('--epsilon-end', type=float, default=None)
    tr.add_argument('--epsilon-decay-steps', type=int, default=None)
    tr.add_argument('--reward-clip', choices=['none', 'sign'], default=None)
    tr.add_argument('--checkpoint-every', type=int, default=25)
    tr.add_argument('--q-sample-freq', type=int, default=32)
    tr.add_argument('--max-steps', type=int, default=0)
    tr.add_argument('--resume', action='store_true')
    tr.add_argument('--no-double', action='store_true')
    ev = sub.add_parser('evaluate', help='Evaluate an existing checkpoint; no training')
    ev.add_argument('--game', choices=['cartpole','mspacman'], required=True)
    ev.add_argument('--out', default=None)
    ev.add_argument('--checkpoint', default=None)
    ev.add_argument('--episodes', type=int, default=500)
    ev.add_argument('--seed', type=int, default=2026)
    ev.add_argument('--epsilon', type=float, default=None)
    args = ap.parse_args()
    if args.command == 'train':
        if args.eval_epsilon is None:
            args.eval_epsilon = 0.0 if args.game == 'cartpole' else .01
        train(args)
    else:
        cmd_evaluate(args)

if __name__ == '__main__':
    main()
