"""Masked on-policy advantage actor-critic for joint market rollouts."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, actions, learning_rate=3e-4):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.actor = nn.Linear(64, actions)
        self.critic = nn.Linear(64, 1)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.rollout = []
        self.pending = None
        self.updates = 0
        self.last_loss = 0.0
        self.gamma = 0.99

    @staticmethod
    def features(obs):
        x = np.zeros(64, dtype=np.float32)
        prices = np.asarray(obs['current_price'], dtype=float)
        hist = np.asarray(obs['price_history'], dtype=float)
        active = np.asarray(obs['available_assets']) > 0
        values = max(float(obs['agent_total_value']), 1e-8)
        for i in np.flatnonzero(active):
            p = hist[:, i]
            p = p[p > 0]
            r = np.diff(np.log(p)) if len(p) > 1 else np.zeros(1)
            x[i] = np.mean(r[-5:]) / .01
            x[10+i] = np.mean(r[-20:]) / .01
            x[20+i] = np.std(r[-20:]) / .01
            x[30+i] = obs['agent_holdings'][i] * prices[i] / values
            x[40+i] = 1
        x[50] = float(obs['agent_cash']) / values
        x[51] = float(obs.get('is_black_box', 0))
        x[52] = float(obs.get('villain_points', 0)) / 10
        x[53] = float(np.mean(obs.get('active_events', [0])))
        x[54:64] = np.asarray(obs.get('triggerable_event_mask', np.zeros(10)))[:10]
        return np.clip(x, -10, 10)

    def forward(self, x):
        h = self.body(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def select(self, obs, mask, training=True):
        x = self.features(obs)
        with torch.no_grad():
            logits, value = self(torch.from_numpy(x))
            distribution = Categorical(logits=logits.masked_fill(~torch.as_tensor(mask), -1e9))
            choice = int(distribution.sample() if training else distribution.probs.argmax())
        self.pending = (x, choice, mask.copy(), float(value)) if training else None
        return choice

    def observe(self, reward, next_obs, terminated, truncated):
        if self.pending is None:
            return
        with torch.no_grad():
            _, next_value = self(torch.from_numpy(self.features(next_obs)))
        self.rollout.append((*self.pending, float(reward), 0.0 if terminated else float(next_value), terminated or truncated))
        self.pending = None
        if len(self.rollout) >= 32 or terminated or truncated:
            self.update()

    def update(self):
        if not self.rollout:
            return
        advantages, gae = [], 0.0
        for _, _, _, value, reward, next_value, end in reversed(self.rollout):
            delta = reward + self.gamma * next_value - value
            gae = delta + self.gamma * .95 * (not end) * gae
            advantages.append(gae)
        advantages = torch.tensor(advantages[::-1], dtype=torch.float32)
        targets = advantages + torch.tensor([r[3] for r in self.rollout])
        obs = torch.from_numpy(np.stack([r[0] for r in self.rollout]))
        actions = torch.tensor([r[1] for r in self.rollout])
        masks = torch.from_numpy(np.stack([r[2] for r in self.rollout]))
        logits, values = self(obs)
        distribution = Categorical(logits=logits.masked_fill(~masks, -1e9))
        # Keep advantages in reward units: normalization of a constant negative
        # batch would erase its learning signal.
        actor_loss = -(distribution.log_prob(actions) * advantages.detach()).mean()
        critic_loss = nn.functional.smooth_l1_loss(values, targets.detach())
        loss = actor_loss + .5 * critic_loss - .01 * distribution.entropy().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite actor-critic loss')
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), .5, error_if_nonfinite=True)
        self.optimizer.step()
        self.last_loss = float(loss.detach())
        self.updates += 1
        self.rollout.clear()

    def snapshot(self):
        return {'model': self.state_dict(), 'optimizer': self.optimizer.state_dict(),
                'updates': self.updates}

    def restore(self, state):
        self.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.updates = state['updates']
        self.pending = None
        self.rollout.clear()
