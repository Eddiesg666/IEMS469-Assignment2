#!/usr/bin/env python3
"""Create transparent draft or submission-ready PDF from *actual* saved experiment outputs.

Draft: python build_report.py --username YOUR_NAME --repo IEMS469-Assignment2
Final: python build_report.py --username YOUR_NAME --repo IEMS469-Assignment2 --final
"""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, KeepTogether)

ROOT=Path(__file__).resolve().parent
BLUE=colors.HexColor('#12304F')
INK=colors.HexColor('#243243')
PALE=colors.HexColor('#EFF4F9')


def load(path):
    p=ROOT/path
    return json.loads(p.read_text()) if p.exists() else None


def styles():
    s=getSampleStyleSheet()
    s.add(ParagraphStyle(name='TitleNew', parent=s['Title'], fontName='Helvetica-Bold', fontSize=19,
                         leading=23, textColor=BLUE, spaceAfter=14))
    s.add(ParagraphStyle(name='HdrNew', parent=s['Heading2'], fontSize=12.3, leading=15.5,
                         textColor=BLUE, spaceBefore=14, spaceAfter=7))
    s.add(ParagraphStyle(name='SubNew', parent=s['Heading3'], fontSize=10.1,
                         textColor=BLUE, spaceBefore=10, spaceAfter=4))
    s.add(ParagraphStyle(name='BodyNew', fontName='Helvetica', fontSize=9.1,
                         leading=13.6, textColor=INK, spaceAfter=6))
    s.add(ParagraphStyle(name='TinyNew', fontName='Helvetica', fontSize=7.8,
                         leading=10.7, textColor=INK, spaceAfter=5))
    s.add(ParagraphStyle(name='WarnNew', fontName='Helvetica-Bold', fontSize=10,
                         leading=14, textColor=colors.HexColor('#B44024'), spaceAfter=8))
    s.add(ParagraphStyle(name='CaptionNew', parent=s['TinyNew'], alignment=TA_CENTER,
                         textColor=colors.HexColor('#596A7E'), spaceAfter=9))
    return s


def paragraph(text, st):
    return Paragraph(escape(str(text)), st)


def table(rows, widths=None):
    content=[[Paragraph(escape(str(cell)),HEADER_CELL if idx == 0 else CELL) for cell in row] for idx, row in enumerate(rows)]
    t=Table(content, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),BLUE),('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, PALE]),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),9),
        ('RIGHTPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),7),
        ('BOTTOMPADDING',(0,0),(-1,-1),7),
        ('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#D4DEE8'))]))
    return t


def figure(path, label, sty, maxw=6.75*inch, maxh=3.2*inch):
    if not path.exists():
        return paragraph('MISSING FIGURE: '+str(path.name),sty['WarnNew'])
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w,h=im.size
    scale=min(maxw/w,maxh/h)
    return KeepTogether([Image(str(path),width=w*scale,height=h*scale),
                         paragraph(label,sty['CaptionNew'])])


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#D8E2EB'))
    canvas.line(43,43,doc.pagesize[0]-43,43)
    canvas.setFont('Helvetica',8)
    canvas.setFillColor(colors.HexColor('#687B90'))
    canvas.drawString(44,31,'IEMS 469 | Advanced Algorithms for Machine Learning | Assignment 2')
    canvas.drawRightString(doc.pagesize[0]-44,31, f'Page {doc.page}')
    canvas.restoreState()

CELL=ParagraphStyle('cell', fontName='Helvetica',fontSize=8.3,leading=11.4,textColor=INK)
HEADER_CELL=ParagraphStyle('headercell', fontName='Helvetica-Bold',fontSize=8.3,leading=11.4,textColor=colors.white)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--username', default='Eddiesg666')
    ap.add_argument('--repo', default='IEMS469-Assignment2')
    ap.add_argument('--final', action='store_true', help='Require both completed 500-episode evaluations')
    args=ap.parse_args()
    st=styles()
    experiments={game:(load(Path('runs')/game/'train_history.json'),
                       load(Path('runs')/game/'evaluation_metrics.json'),
                       load(Path('runs')/game/'config.json'))
                 for game in ('cartpole','mspacman')}
    missing=[]
    if load(Path('runs') / 'algorithm_validation.json') is None:
        missing.append('analytical algorithm validation')
    for name in ('cartpole', 'mspacman'):
        if load(Path('runs') / f'validation_{name}.json') is None:
            missing.append(f'{name} real-environment random-policy validation')
    for game,(train,ev,cfg) in experiments.items():
        if train is None: missing.append(f'{game} training history')
        if ev is None or ev.get('eval_episode_count')!=500: missing.append(f'{game} 500-episode evaluation')
        for image in ['max_q_values.png','training_rewards.png','evaluation_histogram.png','policy_decisiveness.png']:
            if not (ROOT/'runs'/game/image).exists():
                missing.append(f'{game} {image}')
    if args.final and missing:
        raise RuntimeError('Cannot produce final report; missing required actual results: '+', '.join(missing))
    is_draft=bool(missing)
    pdf=ROOT/('assignment2_report_DRAFT.pdf' if is_draft else 'assignment2_report.pdf')
    story=[]
    def add(text,fmt='BodyNew'): story.append(paragraph(text,st[fmt]))
    def header(text): add(text,'HdrNew')
    def sub(text): add(text,'SubNew')
    story.append(paragraph('Assignment 2: Deep Q-Network',st['TitleNew']))
    add('IEMS 469 - Advanced Algorithms for Machine Learning')
    add(f'GitHub username: {args.username} | Repository: {args.repo}')
    add(f'Public source code: https://github.com/{args.username}/{args.repo}')
    if is_draft:
        add('DRAFT - NOT READY FOR SUBMISSION. The items listed below have not been generated from real training/evaluation yet.','WarnNew')
        for item in missing: add('Missing: '+item,'TinyNew')
    else:
        add('All six required training/evaluation plots and both 500-episode evaluations were found. Confirm the public GitHub repository before submitting.')
    header('1. Reproduction and execution')
    add('Environment: Python 3.9 or later, pip-installable PyTorch, Gymnasium, ale-py, NumPy, Matplotlib, ReportLab, pytest. Run from repository root. Single MPS GPU on Apple Silicon or a single CUDA GPU where available.')
    for cmd in ['python3 -m venv .venv; source .venv/bin/activate',
                'python -m pip install -r requirements.txt',
                'python -m pytest -q tests/; python validate_algorithm.py',
                'python validate.py --game cartpole --random-episodes 30',
                'python validate.py --game mspacman --random-episodes 10',
                'python dqn.py train --game cartpole --episodes 800 --eval-episodes 500',
                'python dqn.py train --game mspacman --episodes 2000 --eval-episodes 500',
                'python dqn.py train --game mspacman --episodes 2000 --resume',
                'python build_report.py --username '+args.username+' --repo '+args.repo+' --final']:
        add(cmd,'TinyNew')
    add('The optional --resume command is used only if the previous run was interrupted. It restores the replay buffer, optimizer, target network, epsilon schedule, and training counters from the latest complete checkpoint.')
    header('2. DQN formulation and implementation')
    add('State, action and reward: CartPole-v1 uses a four-dimensional observation and two actions, reward +1 per successful step, with gamma=0.95. ALE/MsPacman-v5 uses RGB image observations (210x160x3), nine available actions and gamma=0.99. The goal is 500 CartPole steps or high MsPacman score.')
    add('Image preprocessing matches the provided algorithm: rows [1:176:2], every second column, sum RGB channels, set color sum 448 to zero, integer divide by 3 and subtract 128. We store those int8 pixel values with a lossless +128 offset in uint8 and stack four observations, obtaining a 4x88x80 image state. The CNN predicts nine Q-values. The raw game reward is preserved for reporting, while optional sign clipping stabilizes the MsPacman TD target.')
    add('Online Q network selects actions epsilon-greedily. The target network is copied periodically. A bounded replay buffer uniformly samples stored (state,action,reward,next state,termination,truncation) transitions; compressed image histories keep memory practical. We use Adam, Huber loss, gradient clipping, linear epsilon decay and optional Double DQN (default enabled). The target is y = r + gamma*(1-terminated)*Q_target(next_state,argmax_a Q_online(next_state,a)). The terminal flag masks bootstrapping; time-limit truncation alone does not.')
    add('Q-value plots show the maximum, over sampled training states in each episode, of max_a Q(s,a), alongside the mean of those sampled state maxima. Sampling is performed every 32 environment decisions, not a complete enumeration of every state.')
    validation=load(Path('runs')/'algorithm_validation.json')
    header('3. Quantitative algorithm tests, edge cases, and controls')
    add('Tests include: exact offered image preprocessing, stacked-frame reset and history boundaries, nine Q-values, ReplayBuffer overwrite/wrap/compression, terminal versus time-limit Bellman targets, Double DQN online-action selection, target-network freezing and copying, epsilon endpoints, random action coverage, and actual reward-driven learning on a two-state known-optimal terminal task.')
    if validation:
        story.append(table([['Quantitative controlled test','Measured result'],
             ['Two-state analytic task: optimal-action accuracy', f"{validation['optimal_action_accuracy']:.1%}"],
             ['Double DQN analytically predicted TD targets', str(validation['bootstrap_target_example'])],
             ['With gamma=0, TD targets', str(validation['gamma_zero_target_example'])],
             ['Replay sampling: maximum proportional count deviation', f"{validation['replay_sampling_max_relative_deviation']:.1%}"],
             ['Tests satisfying analytical expectations', str(validation['all_tests_pass'])]],widths=[3.65*inch,2.9*inch]))
    else:
        add('Quantitative algorithm validation has not been executed. Run python validate_algorithm.py.','WarnNew')
    for game in ('cartpole','mspacman'):
        controls=load(Path('runs')/f'validation_{game}.json')
        if controls:
            add(f"{game}: untrained random policy over {controls['random_episodes']} episodes: mean {controls['random_mean']:.2f}, sample std {controls['random_sample_std']:.2f}. Deterministic seeded action-sequence reproducibility test: {controls['deterministic_repeatability_test_passed']}. {controls['random_episode_capped_count']} artificially capped random-policy games.")
        else:
            add(f'{game}: random-policy validation not yet run; execute python validate.py --game {game} --random-episodes 30.','WarnNew')
    add('Additional edge-case handling: unavailable replay samples defer training until warmup; stale ring-buffer entries are overwritten; no bootstrap on true terminal states; truncation is separately tracked; frame histories reset at each episode; the action set is enforced from Gymnasium; Q-values are evaluated without reward clipping. Deterministic seeded replay checks are documented in validation JSON.')
    add('First-ten-step diagnostic traces, with the selected action, environment reward, epsilon, estimated maximum Q-value, and termination flags, are included in the saved episode history. The JSON enables inspection of how state, Q-values and actions interact at the beginning of training.')
    for game,(train,ev,cfg) in experiments.items():
        header('4. '+('CartPole-v1' if game=='cartpole' else 'MsPacman-v5')+' experiments')
        if cfg:
            rows=[['Hyperparameter','Actual saved setting']]
            for key in ('episodes','gamma','lr','batch_size','replay_capacity','warmup','train_freq',
                        'target_sync_steps','epsilon_end','epsilon_decay_steps','reward_clip','double'):
                rows.append([key,str(cfg.get(key,'unavailable'))])
            story.append(table(rows,widths=[2.7*inch,3.85*inch]))
        if train:
            rewards=np.array([r['reward'] for r in train])
            add(f"Training: {len(train)} episodes. First 100 average {rewards[:100].mean():.2f}; last 100 average {rewards[-100:].mean():.2f}. Maximum individual training score {rewards.max():.2f}. Q-values monitored every {cfg.get('q_sample_freq',32) if cfg else 32} decisions. Any observed poor or unstable learning should be reported rather than hidden.")
        else: add('Training statistics pending actual execution.','WarnNew')
        if ev:
            add(f"Evaluation ({ev['eval_episode_count']} independent seeded episodes): mean {ev['eval_mean']:.3f}, sample std {ev['eval_std_sample']:.3f}; 95% normal approximation interval [{ev['eval_ci95_normal'][0]:.3f}, {ev['eval_ci95_normal'][1]:.3f}], range {ev['eval_min']:.1f} to {ev['eval_max']:.1f}. Evaluation epsilon {ev['eval_epsilon']}.")
            if game=='cartpole':
                successes=sum(r>=500 for r in ev['eval_rewards'])
                add(f'Strict full-500-step successes: {successes} / {ev["eval_episode_count"]}.')
        else: add('Required 500-episode evaluation results are pending; do not substitute training rewards.','WarnNew')
        base=ROOT/'runs'/game
        for fn,caption in [('max_q_values.png','Maximum predicted Q-values versus episode (assignment required).'),
                           ('training_rewards.png','Episode score with trailing 100-episode moving average (assignment required).'),
                           ('evaluation_histogram.png','Histogram of independent post-training episode rewards (assignment required).'),
                           ('policy_decisiveness.png','Nonstandard diagnostic: Q-action gap versus epsilon; monitors confident-action collapse.'),
                           ('td_error_percentiles.png','Nonstandard diagnostic: Bellman prediction error distribution across training.')]:
            story.append(figure(base/fn,caption,st))
    header('5. Interpreting Q-values, reward learning and limitations')
    add('The Q-value maximum is a predicted *discounted* future return, not the same as an undiscounted game score. MsPacman uses sign-clipped rewards for TD optimization, so Q predictions and raw episode scores differ in scale. Reward curves measure real game outcomes. The action-gap plot helps diagnose overconfidence during epsilon decay; it does not independently prove policy improvement. The TD-error quantile plot reveals whether a small set of transitions dominates learning.')
    add('One seed per method does not establish algorithmic superiority across random initializations. Any claim of learning must be supported by increasing trailing reward, independent evaluation, comparison with the untrained control, and training stability. MsPacman-v5 includes sticky actions (25% repeat probability), so environment transitions are stochastic even with a nominally deterministic greedy policy.')
    header('6. What I would investigate with three more months')
    for item in [
        'Run 10-20 distinct training seeds per game, with confidence intervals across independently trained models, not just across evaluation episodes.',
        'Ablate replay capacity (5k/30k/100k), warmup, update frequency, and target synchronization period, holding environment steps fixed.',
        'Compare vanilla DQN and Double DQN; compare Huber and MSE, reward clipping on/off, and epsilon decay schedules.',
        'Evaluate prioritized replay, n-step returns and dueling architectures, with separate checks that the core DQN requirements remain satisfied.',
        'Use Q calibration checks: from fixed probe states, compare predicted Q with empirical discounted Monte Carlo returns under the trained greedy policy.',
        'Test network robustness to ALE sticky-action settings and different seeds while preserving ALE/MsPacman-v5 as the main reported environment.',
        'Measure GPU memory, compression ratio, frames per second, replay action diversity, gradient stability, and value overestimation.',
        'Test 4,000/5,000/6,000 MsPacman episode budgets, monitor generalization across checkpoints and report elapsed compute as well as reward.'
    ]: add('- '+item)
    header('7. Source and reproducibility notes')
    add('Assignment specification: IEMS 469, Assignment 2 - Deep Q-Network, due November 11, 2026. MsPacman official environment: https://ale.farama.org/environments/ms_pacman/. CartPole-v1: Gymnasium Classic Control documentation. Full source, requirements, unit tests, environment probes and recorded per-episode data are included in the GitHub repository.')
    add('Submission verification: confirm the GitHub repository is public and accessible and the report shows 500 evaluation episodes for both required environments. Do not submit a DRAFT report.')
    doc=SimpleDocTemplate(str(pdf),pagesize=(8.5*inch,11*inch),
          leftMargin=.72*inch,rightMargin=.72*inch,topMargin=.72*inch,bottomMargin=.70*inch,
          title='IEMS469 Assignment 2 Deep Q-Network',author=args.username)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(json.dumps(dict(report=str(pdf), submit_ready=not is_draft,
                          missing_results=missing, pages='inspect generated PDF'),indent=2))

if __name__=='__main__': main()
