# IEMS 469 — Assignment 2: Deep Q-Network


This is an original implementation of Deep Q-Network for **CartPole-v1** and **ALE/MsPacman-v5**. DQN uses a PyTorch neural network, a bounded uniform experience-replay buffer, and a periodically synchronized target network. By default the Q-target uses the Double-DQN selection rule (a documented DQN refinement), Huber loss, and gradient clipping. There is **no** PPO, policy gradient or copied RL training framework.

> **Evidence status:** Unit tests and end-to-end synthetic training/evaluation/resume workflows have been run. The actual CartPole and MsPacman environments cannot be run in the model's offline runtime because Gymnasium/ALE package downloads are blocked. The included PDF is consequently a **DRAFT**, not a fully completed submission. You must run actual training and final evaluation; don't submit the draft.

## 1. Project layout

| File | Purpose |
|---|---|
| `dqn.py` | Training, epsilon-greedy action selection, replay sampling, checkpoint/resume, evaluation |
| `models.py` | CartPole MLP and 9-output MsPacman CNN |
| `agent.py` | DQN/Double-DQN Bellman targets, optimizer, target network |
| `replay.py` | Uniform replay ring buffer, compressed uint8 image stacks |
| `environment.py` | Strict environment IDs and assignment-specified image preprocessing |
| `plotting.py` | Assignment-required plots plus nonstandard policy-gap and TD-error plots |
| `validate.py` | Real random-policy controls, seeded determinism and first ten steps |
| `validate_algorithm.py` | Analytical TD targets, toy-learning task, replay-sampling uniformity |
| `build_report.py` | PDF report from actual result files; final mode refuses missing evaluations |
| `tests/` | Algorithm tests and behavior-driven learning experiments |
| `runs/` | Models, logs, CSV, figures and evaluation metrics generated locally |
| `assignment2_report_DRAFT.pdf` | Explicitly incomplete report generated without training results |

## 2. Setup on Mac

Download and extract the full solution ZIP in a **new** folder, separate from Assignment 1. For example place it on your Desktop beside the previous project.

Open Terminal inside the extracted `iems469_assignment2` folder:

```bash
cd ~/Desktop/NU/2026fall/'IEMS 469'/iems469_assignment2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the project is in Downloads, change the first command to `cd ~/Downloads/iems469_assignment2`. The code works with macOS Python 3.9 and later when matching package wheels are available; `reportlab==4.2.5` avoids the Python 3.9 incompatibility experienced in Assignment 1. Do not copy Assignment 1's `pg.py` into this project.

Verify installed versions:

```bash
python -c "import torch, gymnasium, ale_py; print('torch', torch.__version__, 'gymnasium', gymnasium.__version__, 'ale', ale_py.__version__); print('MPS:', torch.backends.mps.is_available())"
```

## 3. Run the offline algorithm tests (done here: 14 passed)

```bash
python -m pytest -q tests/
python validate_algorithm.py
```

`validate_algorithm.py` creates `runs/algorithm_validation.json` and quantitatively tests: exact Double-DQN targets (including gamma=0 and terminal states), uniform sampling in a bounded buffer, and a known two-state reward task on which DQN must learn the correct action. These tests assess actual algorithm behavior, not just data dimensions.

## 4. Check the actual games first

```bash
python validate.py --game cartpole --random-episodes 30
python validate.py --game mspacman --random-episodes 10
```

The scripts execute the exact Gymnasium environments. They report untrained random-policy mean/std, the first ten steps' actions/state changes/rewards, and reproducibility of a seeded deterministic action sequence. In MsPacman, an initial random episode may yield a score but it is **not** evidence of trained learning.

MsPacman-v5 is the nine-action `ALE/MsPacman-v5` variant with four emulator frames per step and 0.25 sticky-action probability. The action head outputs **9 Q-values** and selects one of all nine actions; it does not restrict the actions to Pong's two-action subset.

## 5. Train CartPole (required, prioritize getting it to score 500)

```bash
python dqn.py train --game cartpole --episodes 800 --eval-episodes 500 --checkpoint-every 25
```

Training logs include episode score, trailing-100 mean, maximum sampled Q-value, epsilon and gradient-update count. At the end, the agent automatically uses the best training-window checkpoint for 500 **independent** greedy evaluation episodes. It reports the actual mean/std and saves:

- `runs/cartpole/train_history.json`, `train_episodes.csv`, `config.json`
- `runs/cartpole/evaluation_metrics.json`, `evaluation_episodes.csv`
- `runs/cartpole/training_rewards.png`
- `runs/cartpole/max_q_values.png`
- `runs/cartpole/evaluation_histogram.png`
- `runs/cartpole/policy_decisiveness.png` (nonstandard)
- `runs/cartpole/td_error_percentiles.png` (nonstandard)

CartPole is considered strictly solved in an evaluation rollout if it reaches the full 500 steps. The report counts how many of the 500 evaluation episodes satisfy this condition. `--episodes 800` is a starting training budget, not a guarantee of convergence. If the result is poor, keep the logs and tune using a new experiment directory rather than silently claiming success.

## 6. Train MsPacman — long experiment; start ASAP

The assignment specifically mentions **4,000–6,000** MsPacman training episodes and requires real learning/improvement. We use 5,000 as an initial budget. **This could require many hours or days on a laptop.** No actual runtime or convergence claim is implied.

```bash
python dqn.py train \
  --game mspacman \
  --episodes 5000 \
  --eval-episodes 500 \
  --checkpoint-every 25
```

The key default parameters are gamma=0.99, Adam LR=0.0001, batch=32, replay capacity 30,000 transitions, replay warmup 10,000 decisions, train every 4 decisions, target sync every 5,000 decisions, epsilon 1.0→0.10 over 650,000 decisions, raw evaluation score, and sign-clipped **training rewards only**. The screen is preprocessed as the assignment specifies and **four** 88x80 frames are stacked as image input.

If your Mac is slower or you want a short functional test **without overwriting the final experiment**, use a different output directory and shorter replay warmup:

```bash
python dqn.py train \
  --game mspacman \
  --episodes 3 \
  --warmup 100 \
  --batch-size 8 \
  --eval-episodes 3 \
  --checkpoint-every 1 \
  --out runs/mspacman_probe
```


### Resume after an interruption

The `checkpoint_latest.pt` file stores both Q networks, optimizer, the replay buffer, RNG states, episode number and learning history. Interruptions can lose work since the most recent completed checkpoint, not every prior episode. Resume with:

```bash
python dqn.py train \
  --game mspacman \
  --episodes 5000 \
  --eval-episodes 500 \
  --checkpoint-every 25 \
  --resume
```

`--episodes` means **total target episodes**, not extra episodes. For example, if it resumes from 125/5000, it completes the remaining 4,875. When resuming custom hyperparameters, supply the same custom settings as your initial run. You can increase the target episode count while keeping other settings fixed. Avoid killing Python with `kill -9`, especially while a checkpoint is saving.

### Evaluation alone without re-training

```bash
python dqn.py evaluate --game cartpole --episodes 500
python dqn.py evaluate --game mspacman --episodes 500
```

Evaluations use the saved best policy if one exists, otherwise the latest checkpoint. `CartPole` uses greedy epsilon=0; `MsPacman` uses epsilon=0.01 to test near-greedy play under sticky actions. The complete reward list, histogram, mean and sample standard deviation are saved. Progress prints every 25 episodes, **not continuously**.

## 7. Figures and their interpretation

The three assignment-required visualizations for **each game** are:

1. `max_q_values.png`: episode-wise maximum over **sampled** states of `max_a Q(s,a)` (states are sampled every 32 environment decisions). The mean of those state maxima is overlaid. These values are **discounted predictions**, not final undiscounted game scores.
2. `training_rewards.png`: every episode's undiscounted return and the trailing-100 average.
3. `evaluation_histogram.png`: the 500 final complete rollout returns, with mean/std in the report.

**Nonstandard diagnostic 1, not the prohibited replay-age video:** `policy_decisiveness.png` overlays the mean gap between the highest and second-highest Q-value with epsilon across training episodes. A large gap without reward improvement may indicate overconfident but bad actions.

**Nonstandard diagnostic 2:** `td_error_percentiles.png` plots the 10th/50th/90th percentiles of absolute Bellman residuals from gradient steps as training proceeds. This helps identify instability and rare high-error samples.

## 8. Edge cases and experiments

- Replay not full: no training until the minimum warmup + batch size is available.
- Replay full: old transitions overwritten while sampling still covers only current entries.
- Terminal transition: Bellman target contains no bootstrapped future term.
- Gym time-limit truncation: episode stops but TD bootstrap uses the observed next state (not mistaken for physical termination).
- Image boundaries: history resets on every episode; early states repeat the initial frame; prevents information leaking from a previous life/game.
- Sticky Atari actions: deterministic policy output does not imply deterministic environment transitions; only seeds plus action sequence can support reproducibility checks.
- Long games: no artificial maximum game length by default. An optional `--max-steps N` is for diagnostics only, and the cap is logged; never describe capped games as completed evaluations.
- Q inflation: inspect Q-value curves and TD-error percentiles for training instability; use Double DQN and target synchronization.
- Sparse/staged reward: MsPacman uses sign clipping in TD targets but reports raw game scores.
- Fixed seed: experiment uses 2026 for initial network/training, and independent seeds for evaluation. More seeds are needed for reliable generalization claims.

Run `python validate_algorithm.py` and both `python validate.py` commands to obtain **quantitative evidence** of these claims.

## 9. Create your report

Before training finishes, only a draft can be generated:

```bash
python build_report.py --username Eddiesg666 --repo IEMS469-Assignment2
```

After BOTH games finish with 500 evaluation episodes, generate the final report:

```bash
python build_report.py --username Eddiesg666 --repo IEMS469-Assignment2 --final
```

The generator refuses `--final` if any required evaluation is missing or has fewer than 500 games or any required plot is missing. It includes algorithm equations, actual hyperparameters, performance statistics, edge cases, diagnostics and the extra-three-months plan. Open the PDF and review all plots and text before submitting.


## Official links

- https://ale.farama.org/environments/ms_pacman/
- https://gymnasium.farama.org/environments/classic_control/cart_pole/
