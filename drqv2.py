# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import OrderedDict
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import e2cnn.nn as enn
from e2cnn import gspaces

import utils


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
#         self.repr_dim = 32 * 10 * 10
        self.repr_dim = None
        self.c4_act = None
        self.convnet = None

    def __getstate__(self):
        """Overriden to handle not being able to pickle e2cnn network"""
        res = {k: v for (k, v) in self.__dict__.items()
               if self._should_pickle(k)}
        return res

    def _should_pickle(self, val):
        return val != 'c4_act' and val != '_modules'

    def forward(self, obs):
        assert(len(obs.shape) == 4)
        obs = obs / 255.0 - 0.5
        geo = enn.GeometricTensor(obs, enn.FieldType(self.c4_act,
                                                     obs.shape[1] * [self.c4_act.trivial_repr]))
        h = self.convnet(geo)
#         h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.c4_act = None
        self.policy = None
        self.trunk = None
#         self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
#                                    nn.LayerNorm(feature_dim), nn.Tanh())
#         self.c4_act = gspaces.FlipRot2dOnR2(4)
#         self.policy = enn.R2Conv(enn.FieldType(self.c4_act, repr_dim * [self.c4_act.regular_repr]),
#                                  enn.FieldType(self.c4_act, 1 * [self.c4_act.irrep(1, 2)]),
#                                  kernel_size=1, padding=0)


#         self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
#                                     nn.ReLU(inplace=True),
#                                     nn.Linear(hidden_dim, hidden_dim),
#                                     nn.ReLU(inplace=True),
#                                     nn.Linear(hidden_dim, action_shape[0]))

#         self.apply(utils.weight_init)

    def forward(self, obs, std):
#                           TODO change trunk output to regular
#         h = self.trunk(obs).tensor.view(obs.shape[0], -1)
#         h = torch.tanh(h.tensor)
#         h = enn.GeometricTensor(h, enn.FieldType(self.c4_act,
#                                               256 * [self.c4_act.regular_repr]))
        mu = self.policy(obs.tensor.view(obs.shape[0], -1))#.tensor.view(obs.shape[0], -1)
        assert mu.shape[1:] == torch.Size(
            [1]), f'Action output not correct shape: {mu.shape}'
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist

    def __getstate__(self):
        """Overriden to handle not being able to pickle e2cnn network"""
        res = {k: v for (k, v) in self.__dict__.items()
               if self._should_pickle(k)}
        return res

    def _should_pickle(self, val):
        return val != 'c4_act' and val != '_modules'


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.action_shape = action_shape
        self.trunk = None
        self.Q1 = None
        self.Q2 = None
        self.c4_act = None
        self.repr_dim = None
#         self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
#                                    nn.LayerNorm(feature_dim), nn.Tanh())

#         self.Q1 = nn.Sequential(
#             nn.Linear(feature_dim + action_shape[0], hidden_dim),
#             nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
#
#         self.Q2 = nn.Sequential(
#             nn.Linear(feature_dim + action_shape[0], hidden_dim),
#             nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
#         self.c4_act = gspaces.FlipRot2dOnR2(4)
#         self.repr_dim = repr_dim
#         self.Q1 = nn.Sequential(
#             enn.R2Conv(enn.FieldType(self.c4_act, repr_dim * [self.c4_act.regular_repr] + 1 * [self.c4_act.irrep(1, 2)]),
#                        enn.FieldType(self.c4_act, hidden_dim * [self.c4_act.regular_repr]),
#                        kernel_size=1, padding=0),
#             enn.ReLU(enn.FieldType(self.c4_act, hidden_dim * [self.c4_act.regular_repr]), inplace=True),
#             enn.GroupPooling(enn.FieldType(self.c4_act, hidden_dim * [self.c4_act.regular_repr])),
#             enn.R2Conv(enn.FieldType(self.c4_act, hidden_dim * [self.c4_act.trivial_repr]),
#                        enn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
#                        kernel_size=1, padding=0)
#         )
#         self.Q2 = nn.Sequential(
#             enn.R2Conv(enn.FieldType(self.c4_act, repr_dim * [self.c4_act.regular_repr] + 1 * [self.c4_act.irrep(1, 2)]),
#                        enn.FieldType(self.c4_act, hidden_dim * [self.c4_act.regular_repr]),
#                        kernel_size=1, padding=0),
#             enn.ReLU(enn.FieldType(self.c4_act, hidden_dim * [self.c4_act.regular_repr]), inplace=True),
#             enn.GroupPooling(enn.FieldType(self.c4_act, hidden_dim * [self.c4_act.regular_repr])),
#             enn.R2Conv(enn.FieldType(self.c4_act, hidden_dim * [self.c4_act.trivial_repr]),
#                        enn.FieldType(self.c4_act, 1 * [self.c4_act.trivial_repr]),
#                        kernel_size=1, padding=0)
#         )
#
#         self.apply(utils.weight_init)

    def forward(self, obs, action):

        h = self.trunk(obs.tensor.view(obs.shape[0], -1))#.tensor
#         h = torch.tanh(h)
#         h = h.view(h.shape[0], -1)
        obs_action = torch.cat(
            [h, action], dim=1)#.unsqueeze(2).unsqueeze(3)
#         obs_action = enn.GeometricTensor(
#             obs_action, enn.FieldType(self.c4_act,
#                                     1024 * [self.c4_act.irrep(1)] + self.action_shape[0] * [self.c4_act.irrep(1)]))
        q1 = self.Q1(obs_action)#.tensor.reshape(obs.shape[0], 1)
        q2 = self.Q2(obs_action)#.tensor.reshape(obs.shape[0], 1)
        return q1, q2

    def __getstate__(self):
        """Overriden to handle not being able to pickle e2cnn network"""
        res = {k: v for (k, v) in self.__dict__.items()
               if self._should_pickle(k)}
        return res

    def _should_pickle(self, val):
        return val != 'c4_act' and val != '_modules'


class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        # models
        # parameters don't matter here, refactor later because instantiating
        # in train class
        self.encoder = Encoder(obs_shape).to(device)
        self.actor = Actor(0, action_shape, feature_dim,
                           hidden_dim).to(device)

        self.critic = Critic(0, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(0, action_shape,
                                    feature_dim, hidden_dim).to(device)
#         self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
#         self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
#         self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
#         self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
#
        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

#         self.train()
#         self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def save_enc(self, subdir):
        """Saves encoder weights to given directory"""
        torch.save(self.encoder.convnet.eval().state_dict(), subdir)

    def save_actor(self, subdir):
        """Saves actor weights to given directory"""
        torch.save(self.actor.policy.eval().state_dict(), subdir)

    def save_critic(self, subdirq1, subdirq2, subdirqT1, subdirqT2, subdirqTrunk, subdirqTrunkTarg):
        """Saves critic and critic target weights to given directory"""
        torch.save(self.critic.Q1.eval().state_dict(), subdirq1)
        torch.save(self.critic.Q2.eval().state_dict(), subdirq2)
        torch.save(self.critic.trunk.eval().state_dict(), subdirqTrunk)
        torch.save(self.critic_target.Q1.eval().state_dict(), subdirqT1)
        torch.save(self.critic_target.Q2.eval().state_dict(), subdirqT2)
        torch.save(self.critic_target.trunk.eval(
        ).state_dict(), subdirqTrunkTarg)

    def set_networks(self, group, repr_dim, encNet, actNet, actTrunk, critQ1, critQ2, critQT1, critQT2, trunk, trunkT):
        """
        Sets the network and group for encoder, agent, and critic for pickling purposes
        MUST be called immediately after initialization
        """
        self.encoder.c4_act = group
        self.actor.c4_act = group
        self.critic.c4_act = group
        self.critic_target.c4_act = group

        self.encoder.repr_dim = repr_dim
        self.critic.repr_dim = repr_dim
        self.critic_target.repr_dim = repr_dim

        # TODO manually set modules with orderdict, but pass in networks w params loaded already
        odEnc = OrderedDict()
        odEnc['convnet'] = encNet
        self.encoder._modules = odEnc
        self.encoder.convnet = encNet

        odAct = OrderedDict()
        odAct['policy'] = actNet
        odAct['trunk'] = actTrunk
        self.actor._modules = odAct
        self.actor.policy = actNet
        self.actor.trunk = actTrunk

        odCrit = OrderedDict()
        odCrit['Q1'] = critQ1
        odCrit['Q2'] = critQ2
        odCrit['trunk'] = trunk
        self.critic._modules = odCrit
        self.critic.Q1 = critQ1
        self.critic.Q2 = critQ2
        self.critic.trunk = trunk

        odCritTarg = OrderedDict()
        odCritTarg['Q1'] = critQT1
        odCritTarg['Q2'] = critQT2
        odCritTarg['trunk'] = trunkT
        self.critic_target._modules = odCritTarg
        self.critic_target.Q1 = critQT1
        self.critic_target.Q2 = critQT2
        self.critic_target.trunk = trunkT

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        # TODO ask about detaching like this, bc don't packprop through encoder
        obs = enn.GeometricTensor(obs.tensor.detach(), obs.type)
        metrics.update(self.update_actor(obs, step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
