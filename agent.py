"""DQN Bellman targets, epsilon exploration and optimization."""
import random
import numpy as np
import torch
import torch.nn.functional as F
from models import make_model


def select_device(spec='auto'):
    if spec != 'auto':
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def linear_epsilon(step, start, end, decay_steps):
    fraction = min(max(int(step), 0) / max(int(decay_steps), 1), 1.0)
    return float(start + fraction * (end - start))


def bellman_targets(rewards, next_online, next_target, terminals, gamma, double=True):
    """Bootstrap through truncations; stop on actual MDP terminations."""
    if double:
        actions = next_online.argmax(dim=1, keepdim=True)
        future = next_target.gather(1, actions).squeeze(1)
    else:
        future = next_target.max(dim=1).values
    return rewards + gamma * (1.0 - terminals) * future


class DQNAgent:
    def __init__(self, game, device='auto', lr=1e-3, gamma=0.99, double=True, grad_clip=10.0):
        self.game = game
        self.device = select_device(device)
        self.online = make_model(game).to(self.device)
        self.target = make_model(game).to(self.device)
        self.sync_target()
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.gamma, self.double, self.grad_clip = gamma, bool(double), grad_clip
        self.n_actions = 2 if game == 'cartpole' else 9
        self.updates = 0

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)

    def q_values(self, state):
        with torch.no_grad():
            x = torch.as_tensor(np.array(state, copy=True), device=self.device).unsqueeze(0)
            return self.online(x)[0].cpu().numpy()

    def act(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.q_values(state)))

    def train_step(self, replay, batch_size):
        states, actions, rewards, next_states, terminals = replay.sample(batch_size)
        s = torch.as_tensor(states, device=self.device)
        ns = torch.as_tensor(next_states, device=self.device)
        a = torch.as_tensor(actions, device=self.device).long()
        r = torch.as_tensor(rewards, device=self.device)
        t = torch.as_tensor(terminals, device=self.device)
        chosen = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            online_next = self.online(ns) if self.double else None
            target_next = self.target(ns)
            y = bellman_targets(r, online_next, target_next, t, self.gamma, self.double)
        td = y - chosen
        loss = F.smooth_l1_loss(chosen, y)  # Huber loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip))
        self.optimizer.step()
        self.updates += 1
        td_abs = td.detach().abs().cpu().numpy()
        return {'loss': float(loss.item()), 'td_abs_p10': float(np.percentile(td_abs, 10)),
                'td_abs_p50': float(np.percentile(td_abs, 50)),
                'td_abs_p90': float(np.percentile(td_abs, 90)), 'grad_norm': grad_norm,
                'q_batch': float(chosen.detach().mean())}

    def model_state(self):
        return dict(online=self.online.state_dict(), target=self.target.state_dict(),
                    optimizer=self.optimizer.state_dict(), updates=self.updates)

    def load_model_state(self, d):
        self.online.load_state_dict(d['online'])
        self.target.load_state_dict(d['target'])
        self.optimizer.load_state_dict(d['optimizer'])
        self.updates = d['updates']
        for p in self.target.parameters():
            p.requires_grad_(False)
