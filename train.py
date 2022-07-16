# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dm_env import specs
from e2cnn import gspaces
import e2cnn.nn as enn
from torch import nn
import torch
import numpy as np
import hydra
from pathlib import Path
from copy import deepcopy
import warnings
import os
os.environ['MUJOCO_GL'] = 'egl'
import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
from collections import OrderedDict

warnings.filterwarnings('ignore', category=DeprecationWarning)

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'


torch.backends.cudnn.benchmark = True
# group action acting on all networks
g = gspaces.Flip2dOnR2()


def enc_net(obs_shape, act, load_weights):
    n_out = 512
    chan_up = n_out // 6
    net = nn.Sequential(
        # 84x84
        enn.R2Conv(enn.FieldType(act, obs_shape[0] * [act.trivial_repr]),
                   enn.FieldType(act, chan_up*1 *
                                 [act.regular_repr]),
                   kernel_size=3, padding=1),
        enn.ReLU(enn.FieldType(act, chan_up*1 *
                               [act.regular_repr]), inplace=True),
        enn.PointwiseMaxPool(enn.FieldType(
            act, chan_up*1 * [act.regular_repr]), 2),

        # 42x42
        enn.R2Conv(enn.FieldType(act, chan_up*1 * [act.regular_repr]),
                   enn.FieldType(act, chan_up*2 *
                                 [act.regular_repr]),
                   kernel_size=3, padding=0),
        enn.ReLU(enn.FieldType(act, chan_up*2 *
                               [act.regular_repr]), inplace=True),
        enn.PointwiseMaxPool(enn.FieldType(
            act, chan_up*2 * [act.regular_repr]), 2),

        # 20x20
        enn.R2Conv(enn.FieldType(act, chan_up*2 * [act.regular_repr]),
                   enn.FieldType(act, chan_up*3 *
                                 [act.regular_repr]),
                   kernel_size=3, padding=1),
        enn.ReLU(enn.FieldType(act, chan_up*3 *
                               [act.regular_repr]), inplace=True),
        enn.PointwiseMaxPool(enn.FieldType(
            act, chan_up*3 * [act.regular_repr]), 2),

        # 10x10
        enn.R2Conv(enn.FieldType(act, chan_up*3 * [act.regular_repr]),
                   enn.FieldType(act, chan_up*4 *
                                 [act.regular_repr]),
                   kernel_size=3, padding=1),
        enn.ReLU(enn.FieldType(act, chan_up*4 *
                               [act.regular_repr]), inplace=True),
        enn.PointwiseMaxPool(enn.FieldType(
            act, chan_up*4 * [act.regular_repr]), 2),

        # 5x5
        enn.R2Conv(enn.FieldType(act, chan_up*4 * [act.regular_repr]),
                   enn.FieldType(act, chan_up*5 *
                                 [act.regular_repr]),
                   kernel_size=3, padding=0),
        enn.ReLU(enn.FieldType(act, chan_up*5 *
                               [act.regular_repr]), inplace=True),

        # 3x3
        enn.R2Conv(enn.FieldType(act, chan_up*5 * [act.regular_repr]),
                   enn.FieldType(act, n_out *
                                 [act.regular_repr]),
                   kernel_size=3, padding=0),
        enn.ReLU(enn.FieldType(act, n_out *
                               [act.regular_repr]), inplace=True),
        # 1x1
    )
    if load_weights:
        dict_init = torch.load(os.path.join(Path.cwd(), 'encWeights.pt'))
        net.load_state_dict(dict_init)
    return net, 512


def act_net(repr_dim, action_shape, act, load_weights):

    # hardcoded from cfg to test backing up to only equi encoder
    feature_dim = 50
    hidden_dim = 1024
    net = nn.Sequential(
#         enn.R2Conv(
#             enn.FieldType(act, repr_dim * [act.regular_repr]),
#             enn.FieldType(act, feature_dim * [act.regular_repr]),
#             kernel_size=1, padding=0
#         ),
#         enn.InnerBatchNorm(enn.FieldType(act, feature_dim * [act.regular_repr])),
#         enn.ReLU(enn.FieldType(act, feature_dim * [act.regular_repr])),
#         enn.R2Conv(
#             enn.FieldType(act, feature_dim * [act.regular_repr]),
#             enn.FieldType(act, hidden_dim * [act.regular_repr]),
#             kernel_size=1, padding=0
#         ),
#         enn.ReLU(enn.FieldType(act, hidden_dim * [act.regular_repr])),
        enn.R2Conv(
            enn.FieldType(act, feature_dim * [act.irrep(1)]),
            enn.FieldType(act, hidden_dim * [act.irrep(1)]),
            kernel_size=1, padding=0
        ),
        enn.R2Conv(
            enn.FieldType(act, hidden_dim * [act.irrep(1)]),
            enn.FieldType(act, 1 * [act.irrep(1)]),
            kernel_size=1, padding=0
        )
#         enn.ReLU(enn.FieldType(act, hidden_dim * [act.regular_repr])),
#         enn.R2Conv(
#             enn.FieldType(act, hidden_dim * [act.regular_repr]),
#             enn.FieldType(act, action_shape[0] * [act.irrep(1)]),
#             kernel_size=1, padding=0
#         ),
    )
#     net = nn.Sequential(nn.Linear(repr_dim, feature_dim),
#                         nn.LayerNorm(feature_dim),
#                         nn.Tanh(),
#                         nn.Linear(feature_dim, hidden_dim),
#                         nn.ReLU(inplace=True),
#                         nn.Linear(hidden_dim, hidden_dim),
#                         nn.ReLU(inplace=True),
#                         nn.Linear(hidden_dim, 1))

    trunk = nn.Sequential(
        enn.R2Conv(enn.FieldType(act, repr_dim * [act.regular_repr]),
                   enn.FieldType(act, feature_dim * [act.irrep(1)]),
                   kernel_size=1)
    )

    if load_weights:
        dict_init = torch.load(os.path.join(Path.cwd(), 'actWeights.pt'))
        net.load_state_dict(dict_init)
    return net, trunk


def crit_net(repr_dim, action_shape, act, load_weights, target):
    hidden_dim = 1024
    feature_dim = 50
#     net1 = nn.Sequential(
#         enn.R2Conv(enn.FieldType(act, repr_dim * [act.irrep(1)]+ action_shape[0] * [act.irrep(1)]),
#                    enn.FieldType(act, hidden_dim * [act.regular_repr]),
#                    kernel_size=1, padding=0),
#         enn.ReLU(enn.FieldType(act, hidden_dim *
#                  [act.regular_repr]), inplace=True),
#         enn.R2Conv(enn.FieldType(act, hidden_dim * [act.regular_repr]),
#                    enn.FieldType(act, hidden_dim * [act.regular_repr]),
#                    kernel_size=1, padding=0),
#         enn.ReLU(enn.FieldType(act, hidden_dim *
#                  [act.regular_repr]), inplace=True),
#         enn.GroupPooling(enn.FieldType(act, hidden_dim * [act.regular_repr])),
#         enn.R2Conv(enn.FieldType(act, hidden_dim * [act.trivial_repr]),
#                    enn.FieldType(act, 1 * [act.trivial_repr]),
#                    kernel_size=1, padding=0)
#     )
#     net2 = nn.Sequential(
#         enn.R2Conv(enn.FieldType(act, repr_dim * [act.irrep(1)]+ action_shape[0] * [act.irrep(1)]),
#                    enn.FieldType(act, hidden_dim * [act.regular_repr]),
#                    kernel_size=1, padding=0),
#         enn.ReLU(enn.FieldType(act, hidden_dim *
#                  [act.regular_repr]), inplace=True),
#         enn.R2Conv(enn.FieldType(act, hidden_dim * [act.regular_repr]),
#                    enn.FieldType(act, hidden_dim * [act.regular_repr]),
#                    kernel_size=1, padding=0),
#         enn.ReLU(enn.FieldType(act, hidden_dim *
#                  [act.regular_repr]), inplace=True),
#         enn.GroupPooling(enn.FieldType(act, hidden_dim * [act.regular_repr])),
#         enn.R2Conv(enn.FieldType(act, hidden_dim * [act.trivial_repr]),
#                    enn.FieldType(act, 1 * [act.trivial_repr]),
#                    kernel_size=1, padding=0)
#     )
#     trunk = nn.Sequential(
#         enn.R2Conv(
#             enn.FieldType(act, repr_dim * [act.regular_repr]),
#             enn.FieldType(act, repr_dim * [act.regular_repr]),
#             kernel_size=1, padding=0
#         ),
#         enn.InnerBatchNorm(enn.FieldType(act, repr_dim * [act.regular_repr])),
#         enn.ReLU(enn.FieldType(act, repr_dim * [act.regular_repr])),
#     )
    net1 = nn.Sequential(
        nn.Linear(feature_dim + action_shape[0], hidden_dim),
        nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1)
    )
    net2 = nn.Sequential(
        nn.Linear(feature_dim + action_shape[0], hidden_dim),
        nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1)
    )
    trunk = nn.Sequential(
        enn.R2Conv(enn.FieldType(act, repr_dim * [act.regular_repr]),
                   enn.FieldType(act, feature_dim * [act.irrep(1)]),
                   kernel_size=1)
    )
    if load_weights:
        if target:
            dict_init1 = torch.load(os.path.join(
                Path.cwd(), 'critTargWeights1.pt'))
            dict_init2 = torch.load(os.path.join(
                Path.cwd(), 'critTargWeights2.pt'))
            dict_init_trunk = torch.load(os.path.join(
                Path.cwd(), 'critTargWeightsTrunk.pt'))
        else:
            dict_init1 = torch.load(os.path.join(
                Path.cwd(), 'critWeights1.pt'))
            dict_init2 = torch.load(os.path.join(
                Path.cwd(), 'critWeights2.pt'))
            dict_init_trunk = torch.load(
                os.path.join(Path.cwd(), 'critWeightsTrunk.pt'))

        net1.load_state_dict(dict_init1)
        net2.load_state_dict(dict_init2)
        trunk.load_state_dict(dict_init_trunk)

    return net1, net2, trunk


def make_agent(obs_spec, action_spec, cfg):
    global g
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    agent = hydra.utils.instantiate(cfg)

    # don't load weights because we're not loading from pickle, instead initialize
    enc, repr_dim = enc_net(cfg.obs_shape, g, load_weights=False)
    act, actTrunk = act_net(repr_dim, cfg.action_shape, g, load_weights=False)

    q1, q2, trunk = crit_net(
        repr_dim, cfg.action_shape, g, load_weights=False, target=False)
    qt1, qt2, trunkT = crit_net(
        repr_dim, cfg.action_shape, g, load_weights=False, target=True)
    # set networks in agent
    agent.set_networks(g, repr_dim, enc, act, actTrunk, q1, q2, qt1, qt2, trunk, trunkT)
    agent.encoder.apply(utils.weight_init)
    agent.actor.apply(utils.weight_init)
    agent.critic.apply(utils.weight_init)
    agent.critic_target.apply(utils.weight_init)

    agent.encoder.to('cuda')
    agent.actor.to('cuda')
    agent.critic.to('cuda')
    agent.critic_target.to('cuda')

    agent.critic_target.load_state_dict(agent.critic.state_dict())
    agent.encoder_opt = torch.optim.Adam(agent.encoder.parameters(), lr=cfg.lr)
    agent.actor_opt = torch.optim.Adam(agent.actor.parameters(), lr=cfg.lr)
    agent.critic_opt = torch.optim.Adam(agent.critic.parameters(), lr=cfg.lr)
    agent.train()
    agent.critic_target.train()

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

        self.critic_weight_dirTrunk = "critWeightsTrunk.pt"
        self.critic_weight_dirTrunkT = "critTargWeightsTrunk.pt"

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
                               self.criticT_weight_dir1, self.criticT_weight_dir2,
                               self.critic_weight_dirTrunk, self.critic_weight_dirTrunkT)

    def load_snapshot(self):
        global g
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v

        # load weights from pickle state dict
        obs_shape = self.train_env.observation_spec().shape
        action_shape = self.train_env.action_spec().shape
        enc, repr_dim = enc_net(obs_shape, g, load_weights=True)
        act, actTrunk = act_net(repr_dim, action_shape, g, load_weights=True)
        q1, q2, trunk = crit_net(
            repr_dim, action_shape, g, load_weights=True, target=False)
        qt1, qt2, trunkT = crit_net(
            repr_dim, action_shape, g, load_weights=True, target=True)
        self.agent.set_networks(g, repr_dim, enc, act, actTrunk,
                                q1, q2, qt1, qt2, trunk, trunkT)
        self.agent.encoder.to(self.device)
        self.agent.actor.to(self.device)
        self.agent.critic.to(self.device)
        self.agent.critic_target.to(self.device)


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
