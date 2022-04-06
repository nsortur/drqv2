# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
from torch import nn
import e2cnn.nn as enn
from e2cnn import gspaces
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
from collections import OrderedDict

torch.backends.cudnn.benchmark = True
# group action acting on all networks
g = gspaces.FlipRot2dOnR2(4)

def enc_net(obs_shape, act, load_weights):
    n_out = 128
    net = nn.Sequential(
            enn.R2Conv(enn.FieldType(act, obs_shape[0] * [act.trivial_repr]),
                      enn.FieldType(act, n_out//8 * \
                                   [act.regular_repr]),
                      kernel_size=3, stride=2, padding=1),
            enn.ReLU(enn.FieldType(act, n_out//8 * \
                    [act.regular_repr]), inplace=True),
            enn.PointwiseMaxPool(enn.FieldType(
                act, n_out//8 * [act.regular_repr]), 2),

            enn.R2Conv(enn.FieldType(act, n_out//8 * [act.regular_repr]),
                      enn.FieldType(act, n_out//4 * \
                                   [act.regular_repr]),
                      kernel_size=3, stride=2, padding=1),
            enn.ReLU(enn.FieldType(act, n_out//4 * \
                    [act.regular_repr]), inplace=True),
            enn.PointwiseMaxPool(enn.FieldType(
                act, n_out//4 * [act.regular_repr]), 2),


            enn.R2Conv(enn.FieldType(act, n_out//4 * [act.regular_repr]),
                      enn.FieldType(act, n_out//2 * \
                                   [act.regular_repr]),
                      kernel_size=3, stride=2, padding=1),
            enn.ReLU(enn.FieldType(act, n_out//2 * \
                    [act.regular_repr]), inplace=True),
            enn.PointwiseMaxPool(enn.FieldType(
                act, n_out//2 * [act.regular_repr]), 2),

            enn.R2Conv(enn.FieldType(act, n_out//2 * [act.regular_repr]),
                       enn.FieldType(act, n_out * [act.regular_repr]),
                       kernel_size=1),
            enn.ReLU(enn.FieldType(act, n_out * [act.regular_repr]),
                       inplace=True)
    )
    if load_weights:
        dict_init = torch.load(os.path.join(Path.cwd(), 'encWeights.pt'))
        dict_ad = {k.replace('convnet.', ''): v for k, v in dict_init.items()}
        net.load_state_dict(dict_ad)
    net.to('cuda')
    return net

def act_net(repr_dim, act, load_weights):
    
    net = enn.R2Conv(
        enn.FieldType(act, repr_dim * [act.regular_repr]), 
        enn.FieldType(act, 1 * [act.irrep(1, 2)]), 
        kernel_size=1, padding=0
    )
    if load_weights:
        dict_init = torch.load(os.path.join(Path.cwd(), 'actWeights.pt'))
        dict_ad = {k.replace('policy.', ''): v for k, v in dict_init.items()}
        net.load_state_dict(dict_ad)
    net.to('cuda')
    return net

def crit_net(repr_dim, action_shape, act, load_weights):

    hidden_dim = 64
    net1 = nn.Sequential(
        enn.R2Conv(enn.FieldType(act, repr_dim * [act.regular_repr] + 1 * [act.irrep(1, 2)]),
                   enn.FieldType(act, hidden_dim * [act.regular_repr]),
                   kernel_size=1, padding=0),
        enn.ReLU(enn.FieldType(act, hidden_dim * [act.regular_repr]), inplace=True),
        enn.GroupPooling(enn.FieldType(act, hidden_dim * [act.regular_repr])),
        enn.R2Conv(enn.FieldType(act, hidden_dim * [act.trivial_repr]),
                   enn.FieldType(act, 1 * [act.trivial_repr]),
                   kernel_size=1, padding=0)
    )
    net2 = nn.Sequential(
        enn.R2Conv(enn.FieldType(act, repr_dim * [act.regular_repr] + 1 * [act.irrep(1, 2)]),
                   enn.FieldType(act, hidden_dim * [act.regular_repr]),
                   kernel_size=1, padding=0),
        enn.ReLU(enn.FieldType(act, hidden_dim * [act.regular_repr]), inplace=True),
        enn.GroupPooling(enn.FieldType(act, hidden_dim * [act.regular_repr])),
        enn.R2Conv(enn.FieldType(act, hidden_dim * [act.trivial_repr]),
                   enn.FieldType(act, 1 * [act.trivial_repr]),
                   kernel_size=1, padding=0)
    )
    netT1 = deepcopy(net1)
    netT2 = deepcopy(net2)
    if load_weights:
        dict_init1 = torch.load(os.path.join(Path.cwd(), 'critWeights1.pt'))
        dict_init2 = torch.load(os.path.join(Path.cwd(), 'critWeights2.pt'))
        dict_initT1 = torch.load(os.path.join(Path.cwd(), 'critTargWeights1.pt'))
        dict_initT2 = torch.load(os.path.join(Path.cwd(), 'critTargWeights2.pt'))
        dict_ad1 = {k.replace('Q1.', ''): v for k, v in dict_init1.items()}
        dict_ad2 = {k.replace('Q2.', ''): v for k, v in dict_init2.items()}
        dict_adT1 = {k.replace('Q1.', ''): v for k, v in dict_initT1.items()}
        dict_adT2 = {k.replace('Q2.', ''): v for k, v in dict_initT2.items()}
        net1.load_state_dict(dict_ad1)
        net2.load_state_dict(dict_ad2)
        netT1.load_state_dict(dict_adT1)
        netT2.load_state_dict(dict_adT2)
    
    net1.to('cuda')
    net2.to('cuda')
    netT1.to('cuda')
    netT2.to('cuda')
    return net1, net2, netT1, netT2

def make_agent(obs_spec, action_spec, cfg):
    global g
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    agent = hydra.utils.instantiate(cfg)

    # don't load weights because we're not loading from pickle, instead initialize
    enc = enc_net(cfg.obs_shape, g, load_weights=False)
    act = act_net(enc.repr_dim, g, load_weights=False)
    q1, q2, qt1, qt2 = crit_net(enc.repr_dim, cfg.action_shape, g, load_weights=False)
    agent.set_networks(g, enc, act, q1, q2, qt1, qt2)
    agent.encoder.apply(utils.weight_init)
    agent.actor.apply(utils.weight_init)
    agent.critic.apply(utils.weight_init)
    agent.critic_target.apply(utils.weight_init)

    return agent

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        # files to save e2cnn network weights because they're unpicklable
        self.enc_weight_dir = "encWeights.pt"
        self.actor_weight_dir = "actWeights.pt"
        self.critic_weight_dir1 = "critWeights1.pt"
        self.critic_weight_dir2 = "critWeights2.pt"
        self.criticT_weight_dir1 = "critTargWeights1.pt"
        self.criticT_weight_dir2 = "critTargWeights2.pt"

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed)
        self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed)
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                if self.global_frame % 100000 == 0:
                    # save vid every 100k frames instead of every 10k
                    self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)
        self.agent.save_enc(self.enc_weight_dir)
        self.agent.save_actor(self.actor_weight_dir)
        self.agent.save_critic(self.critic_weight_dir1, self.critic_weight_dir2,
                               self.criticT_weight_dir1, self.criticT_weight_dir2)

    def load_snapshot(self):
        global g
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v
        
        # load weights from pickle state dict
        enc = enc_net(cfg.obs_shape, g, load_weights=True)
        act = act_net(enc.repr_dim, g, load_weights=True)
        q1, q2, qt1, qt2 = crit_net(enc.repr_dim, cfg.action_shape, g, load_weights=True)
        self.agent.set_networks(g, enc, act, q1, q2, qt1, qt2)

@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    from train import Workspace as W
    root_dir = Path.cwd()
    print('root:', root_dir)
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
