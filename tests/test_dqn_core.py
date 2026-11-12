import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import random
import numpy as np
import pytest
import torch
from agent import DQNAgent, bellman_targets, linear_epsilon
from environment import preprocess_observation, image_to_uint8, StateHistory
from models import make_model
from replay import ReplayBuffer


def test_exact_offered_preprocessing_shape_range_and_background():
    obs = np.zeros((210, 160, 3), dtype=np.uint8)
    obs[..., 0], obs[..., 1], obs[..., 2] = 210, 164, 74
    obs[101, 42] = [255, 0, 0]
    result = preprocess_observation(obs)
    assert result.shape == (88, 80, 1) and result.dtype == np.int8
    assert result[0, 0, 0] == -128
    assert result[50, 21, 0] == (255//3-128)
    pack = image_to_uint8(obs)
    assert pack.dtype == np.uint8 and pack[0,0] == 0 and pack[50,21] == 85


def test_history_reset_pads_4_and_rolls_without_cross_episode_leakage():
    obs=np.zeros((210,160,3),np.uint8)
    h=StateHistory()
    first=h.reset(obs)
    assert first.shape == (4,88,80)
    assert np.array_equal(first[0],first[-1])
    obs2=obs.copy(); obs2[101,42] = [255,255,255]
    changed=h.step(obs2)
    assert changed[-1,50,21] != changed[-2,50,21]
    assert np.array_equal(h.reset(obs2)[0],h.state()[-1])


def test_cartpole_and_atari_outputs_all_actions():
    q=make_model('cartpole'); p=make_model('mspacman')
    assert q(torch.zeros((3,4))).shape == (3,2)
    assert p(torch.zeros((2,4,88,80), dtype=torch.uint8)).shape == (2,9)


def test_terminal_zeroes_bootstrap_but_time_limit_does_not():
    r=torch.tensor([1.,1.]); t=torch.tensor([1.,0.])
    q1=torch.tensor([[10.,20.],[10.,20.]])
    q2=torch.tensor([[30.,40.],[30.,40.]])
    tar=bellman_targets(r,q1,q2,t,.95,True)
    assert torch.allclose(tar,torch.tensor([1.,39.]))


def test_double_dqn_uses_online_argmax_not_target_max():
    r=torch.zeros(1); t=torch.zeros(1)
    q_online=torch.tensor([[10.,5.]])
    q_target=torch.tensor([[2.,20.]])
    assert bellman_targets(r,q_online,q_target,t,.9,True).item() == pytest.approx(1.8)
    assert bellman_targets(r,q_online,q_target,t,.9,False).item() == pytest.approx(18.)


def test_epsilon_schedule_including_edges():
    assert linear_epsilon(-10, 1.,.1,100) == 1.
    assert linear_epsilon(50,1.,.1,100) == pytest.approx(.55)
    assert linear_epsilon(100,1.,.1,100) == pytest.approx(.1)
    assert linear_epsilon(300,1.,.1,100) == pytest.approx(.1)


def test_target_network_is_frozen_and_sync_exact():
    agent=DQNAgent('cartpole',device='cpu')
    assert all(not p.requires_grad for p in agent.target.parameters())
    with torch.no_grad():
        next(agent.online.parameters()).add_(3.)
    assert not torch.equal(next(agent.online.parameters()),next(agent.target.parameters()))
    agent.sync_target()
    assert torch.equal(next(agent.online.parameters()),next(agent.target.parameters()))


def test_replay_wrap_sampling_without_stale_transitions():
    b=ReplayBuffer(4,'cartpole')
    for i in range(10):
        b.add(np.ones(4)*i, i%2, i, np.ones(4)*(i+1), i==9, i==8)
    assert len(b)==4 and b.pos==2
    s,a,r,ns,t=b.sample(4)
    assert set(r.tolist()) == {6.,7.,8.,9.}
    assert t.sum() == 1
    b2=ReplayBuffer.from_snapshot(b.snapshot())
    assert len(b2)==4 and b2.game=='cartpole'


def test_compressed_image_replay_roundtrip():
    rng=np.random.default_rng(0)
    s=rng.integers(0,256,size=(4,88,80),dtype=np.uint8)
    ns=np.roll(s,1,axis=0)
    b=ReplayBuffer(2,'mspacman')
    b.add(s, 8, 10, ns, False)
    s2, a, r, ns2, t=b.sample(1)
    assert np.array_equal(s2[0],s) and np.array_equal(ns2[0],ns)
    assert a[0]==8
    assert len(b.data[0][0]) < s.nbytes+500  # zlib can expand random data slightly


def test_replay_rejects_invalid_batch():
    b=ReplayBuffer(2,'cartpole')
    with pytest.raises(ValueError): b.sample(1)


def test_random_epsilon_explores_all_discrete_actions():
    random.seed(12)
    a=DQNAgent('mspacman',device='cpu')
    s=np.zeros((4,88,80),dtype=np.uint8)
    choices={a.act(s,epsilon=1.) for _ in range(180)}
    assert choices == set(range(9))


def test_a_bellman_update_actually_changes_online_not_target():
    torch.manual_seed(11)
    agent=DQNAgent('cartpole',device='cpu',lr=1e-2)
    b=ReplayBuffer(100,'cartpole')
    for k in range(20):
        s=np.array([k%2,0,0,0],dtype=np.float32)
        b.add(s,k%2,float((k%2)*4),s,True)
    t_before=[p.detach().clone() for p in agent.target.parameters()]
    o_before=[p.detach().clone() for p in agent.online.parameters()]
    m=agent.train_step(b,16)
    assert np.isfinite(m['loss']) and m['td_abs_p90'] >= 0
    assert any(not torch.allclose(x,y) for x,y in zip(o_before,agent.online.parameters()))
    assert all(torch.allclose(x,y) for x,y in zip(t_before,agent.target.parameters()))
