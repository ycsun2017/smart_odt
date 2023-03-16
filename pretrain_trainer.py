"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import torch
import time
from torch.nn import functional as F
from decision_transformer.models.pretrain_decision_transformer import PretrainDecisionTransformer

class PretrainSequenceTrainer:
    def __init__(
        self,
        model,
        optimizer,
        log_temperature_optimizer,
        scheduler=None,
        device="cuda",
    ):
        self.model = model
        self.optimizer = optimizer
        self.log_temperature_optimizer = log_temperature_optimizer
        self.scheduler = scheduler
        self.device = device
        self.start_time = time.time()
    
    def unsup_train_iteration(
        self,
        loss_fn,
        dataloader,
        train_weights
    ):
        losses, nlls, entropies = [], [], []
        losses_fwd, losses_inv, losses_rew, losses_randinv = [], [], [], []
        logs = dict()
        train_start = time.time()

        self.model.train()
        for _, trajs in enumerate(dataloader):
            loss, loss_fwd, loss_inv, loss_rew, loss_randinv, nll, entropy = self.unsup_train_step(loss_fn, trajs, train_weights)
            losses.append(loss)
            losses_fwd.append(loss_fwd)
            losses_inv.append(loss_inv)
            losses_rew.append(loss_rew)
            losses_randinv.append(loss_randinv)
            nlls.append(nll)
            entropies.append(entropy)

        logs["time/training"] = time.time() - train_start
        logs["training/train_loss_mean"] = np.mean(losses)
        logs["training/train_loss_std"] = np.std(losses)
        logs["training/forward_loss_mean"] = np.mean(losses_fwd)
        logs["training/inverse_loss_mean"] = np.mean(losses_inv)
        logs["training/reward_loss_mean"] = np.mean(losses_rew)
        logs["training/randinv_loss_mean"] = np.mean(losses_randinv)
        logs["training/nll"] = nlls[-1]
        logs["training/entropy"] = entropies[-1]
        logs["training/temp_value"] = self.model.temperature().detach().cpu().item()

        return logs
    
    def unsup_train_step(self, loss_fn, trajs, train_weights):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            timesteps,
            ordering,
            padding_mask,
        ) = trajs

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        rtg = rtg.to(self.device)
        timesteps = timesteps.to(self.device)
        ordering = ordering.to(self.device)
        padding_mask = padding_mask.to(self.device)

        action_target = torch.clone(actions)

        preds = self.model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            ordering,
            padding_mask=padding_mask,
            unsup_losses=True
        )

        # action loss
        loss, nll, entropy = loss_fn(
            preds['action_preds'],  # a_hat_dist
            action_target,
            padding_mask,
            self.model.temperature().detach(),  # no gradient taken here
        )

        # other self-supervised loss
        loss_fwd = F.mse_loss(preds['state_preds'][:, :-1, :], states[:, 1:, :])
        loss_rew = F.mse_loss(preds['reward_preds'], rewards)
        loss_inv = F.mse_loss(preds['inverse_preds'][:, 1:, :], actions[:, :-1, :])
        loss_randinv = F.mse_loss(preds['randinv_preds'], actions)
        print("loss forward", loss_fwd.item(), "loss reward", loss_rew.item(), "loss inverse", loss_inv.item(), "loss randinv", loss_randinv.item())
        
        all_loss = train_weights['dt'] * loss \
                + train_weights['forward'] * loss_fwd \
                + train_weights['reward'] * loss_rew \
                + train_weights['inverse'] * loss_inv \
                + train_weights['randinv'] * loss_randinv

        self.optimizer.zero_grad()
        all_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        self.log_temperature_optimizer.zero_grad()
        temperature_loss = (
            self.model.temperature() * (entropy - self.model.target_entropy).detach()
        )
        temperature_loss.backward()
        self.log_temperature_optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return (
            loss.detach().cpu().item(),
            loss_fwd.detach().cpu().item(),
            loss_inv.detach().cpu().item(),
            loss_rew.detach().cpu().item(),
            loss_randinv.detach().cpu().item(),
            nll.detach().cpu().item(),
            entropy.detach().cpu().item(),
        )
        

    def train_iteration(
        self,
        loss_fn,
        dataloader,
    ):

        losses, nlls, entropies = [], [], []
        logs = dict()
        train_start = time.time()

        self.model.train()
        for _, trajs in enumerate(dataloader):
            loss, nll, entropy = self.train_step_stochastic(loss_fn, trajs)
            losses.append(loss)
            nlls.append(nll)
            entropies.append(entropy)

        logs["time/training"] = time.time() - train_start
        logs["training/train_loss_mean"] = np.mean(losses)
        logs["training/train_loss_std"] = np.std(losses)
        logs["training/nll"] = nlls[-1]
        logs["training/entropy"] = entropies[-1]
        logs["training/temp_value"] = self.model.temperature().detach().cpu().item()

        return logs

    def train_step_stochastic(self, loss_fn, trajs):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            timesteps,
            ordering,
            padding_mask,
        ) = trajs

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        rtg = rtg.to(self.device)
        timesteps = timesteps.to(self.device)
        ordering = ordering.to(self.device)
        padding_mask = padding_mask.to(self.device)

        action_target = torch.clone(actions)

        if isinstance(self.model, PretrainDecisionTransformer):
            action_preds = self.model.forward(
                states,
                actions,
                rewards,
                rtg[:, :-1],
                timesteps,
                ordering,
                padding_mask=padding_mask,
                unsup_losses=False
            )['action_preds']
        else:
            _, action_preds, _ = self.model.forward(
                states,
                actions,
                rewards,
                rtg[:, :-1],
                timesteps,
                ordering,
                padding_mask=padding_mask
            )

        loss, nll, entropy = loss_fn(
            action_preds,  # a_hat_dist
            action_target,
            padding_mask,
            self.model.temperature().detach(),  # no gradient taken here
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        self.log_temperature_optimizer.zero_grad()
        temperature_loss = (
            self.model.temperature() * (entropy - self.model.target_entropy).detach()
        )
        temperature_loss.backward()
        self.log_temperature_optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return (
            loss.detach().cpu().item(),
            nll.detach().cpu().item(),
            entropy.detach().cpu().item(),
        )
