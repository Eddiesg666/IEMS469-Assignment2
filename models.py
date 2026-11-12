"""Networks used by DQN. No environment-specific shortcuts or game RAM."""
import torch
from torch import nn


class CartPoleQ(nn.Module):
    def __init__(self, n_actions=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x.float())


class MsPacmanQ(nn.Module):
    """Nature-style CNN; four 88x80 uint8 luminance frames, 9 action values."""
    def __init__(self, n_actions=9, history=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(history, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        with torch.no_grad():
            size = self.conv(torch.zeros(1, history, 88, 80)).flatten(1).shape[1]
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(size, 512), nn.ReLU(), nn.Linear(512, n_actions))

    def forward(self, x):
        # uint8 0..255. Luminance has same spatial layout as prescribed preprocessor.
        return self.head(self.conv(x.float() / 255.0))


def make_model(game):
    if game == 'cartpole':
        return CartPoleQ(2)
    if game == 'mspacman':
        return MsPacmanQ(9)
    raise ValueError(f'Unknown game: {game}')
