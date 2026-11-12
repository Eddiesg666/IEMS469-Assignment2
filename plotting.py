"""Assignment plots plus two nonstandard, testable DQN diagnostics."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)


def moving_average(values, width=100):
    a = np.asarray(values, dtype=float)
    return np.asarray([a[max(0, i - width + 1):i + 1].mean() for i in range(len(a))])


def training_plots(history, game, folder):
    folder = Path(folder)
    if not history:
        return
    x = np.array([r['episode'] for r in history])
    reward = np.array([r['reward'] for r in history])
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(x, reward, alpha=.27, lw=.8, label='Individual episode')
    ax.plot(x, moving_average(reward), lw=2.0, label='Last up to 100 episodes')
    ax.set(xlabel='Training episode', ylabel='Undiscounted episode reward', title=f'{game}: training rewards')
    ax.grid(alpha=.2); ax.legend()
    _save(fig, folder / 'training_rewards.png')

    qmax = np.array([r['max_sampled_q'] for r in history], dtype=float)
    qmean = np.array([r['mean_sampled_max_q'] for r in history], dtype=float)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(x, qmax, lw=1.2, alpha=.6, label='Maximum sampled max_a Q(s,a)')
    ax.plot(x, qmean, lw=1.5, label='Mean sampled max_a Q(s,a)')
    ax.set(xlabel='Training episode', ylabel='Predicted discounted action-value',
           title=f'{game}: Q-values on states sampled during each episode')
    ax.grid(alpha=.2); ax.legend()
    _save(fig, folder / 'max_q_values.png')

    # NONSTANDARD diagnostic 1: action decisiveness measures collapse risk.
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    gap = np.array([r['mean_q_gap'] for r in history], dtype=float)
    ax.plot(x, gap, label='Average best-vs-second-best Q-value gap', lw=1.5)
    ax.set(xlabel='Training episode', ylabel='Top-1 minus top-2 Q-value',
           title=f'{game}: policy decisiveness vs epsilon (nonstandard diagnostic)')
    a2 = ax.twinx()
    a2.plot(x, [r['epsilon_end'] for r in history], color='darkorange', alpha=.75, label='Exploration epsilon')
    a2.set_ylabel('Epsilon')
    ax.grid(alpha=.2)
    h, l = ax.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    ax.legend(h + h2, l + l2, loc='best')
    _save(fig, folder / 'policy_decisiveness.png')

    # NONSTANDARD diagnostic 2: distribution of absolute Bellman prediction errors.
    mask = [i for i, r in enumerate(history) if r['td_p50'] is not None and np.isfinite(r['td_p50'])]
    if mask:
        xx = x[mask]
        fig, ax = plt.subplots(figsize=(9.5, 4.6))
        for key, label in [('td_p10', '10th percentile'), ('td_p50', 'Median'), ('td_p90', '90th percentile')]:
            ax.plot(xx, [history[i][key] for i in mask], label=label, lw=1.4)
        ax.set(xlabel='Training episode', ylabel='Absolute TD error',
               title=f'{game}: distribution of Bellman errors (nonstandard diagnostic)')
        ax.legend(); ax.grid(alpha=.2)
        _save(fig, folder / 'td_error_percentiles.png')


def evaluation_plot(rewards, game, folder):
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.hist(rewards, bins=min(35, max(6, int(np.sqrt(len(rewards))))),
            edgecolor='white', linewidth=.6)
    ax.axvline(np.mean(rewards), color='black', linestyle='--', lw=1.5,
               label=f'Mean {np.mean(rewards):.2f}')
    ax.set(xlabel='Undiscounted episode reward', ylabel='Number of episodes',
           title=f'{game}: evaluation score distribution ({len(rewards)} episodes)')
    ax.grid(alpha=.15); ax.legend()
    _save(fig, Path(folder) / 'evaluation_histogram.png')
