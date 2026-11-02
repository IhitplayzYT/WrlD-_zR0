"""Masked on policy advantage actor critic for joint market rollouts"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

FEATURE_DIM = 66

class ActorCritic(nn.Module):
    def __init__(self, actions, learning_rate=3e-4):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(FEATURE_DIM, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.actor = nn.Linear(64, actions)
        self.critic = nn.Linear(64, 1)
        nn.init.orthogonal_(self.actor.weight, gain=0.01) # wTw = wwT = I where we find w and set w = 0.01w  // This is to avoid exploding grads
        nn.init.zeros_(self.actor.bias)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.rollout = []
        self.pending = None
        self.updates = 0
        self.last_loss = 0.0
        self.gamma = 0.99

    # From obs builds a FEATURE_DIM sized vector as input for Actor citic nn
    @staticmethod
    def features(obs):
        x = np.zeros(FEATURE_DIM, dtype=np.float32)
        prices = np.asarray(obs['current_price'], dtype=float)
        hist = np.asarray(obs['price_history'], dtype=float)
        active = np.asarray(obs['available_assets']) > 0
        values = max(float(obs['agent_total_value']), 1e-8)
        for i in np.flatnonzero(active):
            p = hist[:, i]
            p = p[p > 0]
            r = np.diff(np.log(p)) if len(p) > 1 else np.zeros(1) # LOg return btwn consecutive pari of assets
            x[i] = np.mean(r[-5:]) / .01
            x[10+i] = np.mean(r[-20:]) / .01
            x[20+i] = np.std(r[-20:]) / .01
            x[30+i] = obs['agent_holdings'][i] * prices[i] / values
            x[40+i] = 1
        x[50] = float(obs['agent_cash']) / values
        x[51] = float(obs.get('is_black_box', 0))
        x[52] = float(obs.get('villain_points', 0)) / 10
        x[53] = float(np.mean(obs.get('active_events', [0])))
        mask = np.asarray(obs.get('triggerable_event_mask', np.zeros(10)), dtype=np.float32)
        x[54:64] = [float(b.mean()) if len(b) else 0. for b in np.array_split(mask, 10)]
        x[64] = float(obs.get('bbox_consensus', 0.)) / .01   # ensemble forecast return (1% units)
        x[65] = float(obs.get('kaiser_direction', 0.))        # Kaiser's last action direction
        return np.clip(x, -10, 10)

    def forward(self, x):
        h = self.body(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def select(self, obs, mask, training=True, bias=None):
        x = self.features(obs)
        with torch.no_grad():
            logits, value = self(torch.from_numpy(x))
            if bias is not None: logits += torch.as_tensor(bias, dtype=logits.dtype)
            distribution = Categorical(logits=logits.masked_fill(~torch.as_tensor(mask), -1e9))
            choice = int(distribution.sample() if training else distribution.probs.argmax())
        self.pending = (x, choice, mask.copy(), float(value), None if bias is None else np.asarray(bias, dtype=np.float32).copy()) if training else None
        return choice

    def override_choice(self, choice):
        if self.pending is not None:
            x, _, mask, value, bias = self.pending
            self.pending = (x, int(choice), mask, value, bias)

    def observe(self, reward, next_obs, terminated, truncated):
        if self.pending is None: return
        with torch.no_grad(): _, next_value = self(torch.from_numpy(self.features(next_obs)))
        self.rollout.append((*self.pending, float(reward), 0.0 if terminated else float(next_value), terminated or truncated))
        self.pending = None
        if len(self.rollout) >= 32 or terminated or truncated: self.update()

    def update(self):
        if not self.rollout: return
        adv, gae = [], 0.0
        for _, _, _, value, _, reward, next_value, end in reversed(self.rollout):
            delta = reward + self.gamma * next_value - value
            gae = delta + self.gamma * .95 * (not end) * gae
            adv.append(gae)
        adv = torch.tensor(adv[::-1], dtype=torch.float32)
        tgt = adv + torch.tensor([r[3] for r in self.rollout])
        obs = torch.from_numpy(np.stack([r[0] for r in self.rollout]))
        actions = torch.tensor([r[1] for r in self.rollout])
        masks = torch.from_numpy(np.stack([r[2] for r in self.rollout]))
        logits, values = self(obs)
        if any(r[4] is not None for r in self.rollout):  # same behaviour policy as at sampling time
            logits += torch.from_numpy(np.stack([r[4] if r[4] is not None else np.zeros(logits.shape[1], np.float32) for r in self.rollout]))
        distrib = Categorical(logits=logits.masked_fill(~masks, -1e9))
        actor_loss = -(distrib.log_prob(actions) * adv.detach()).mean()
        critic_loss = nn.functional.smooth_l1_loss(values, tgt.detach())
        loss = actor_loss + .5 * critic_loss - .01 * distrib.entropy().mean()
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite actor-critic loss')
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), .5, error_if_nonfinite=True)
        self.optimizer.step()
        self.last_loss = float(loss.detach())
        self.updates += 1
        self.rollout.clear()

    def snapshot(self):return {'model': self.state_dict(), 'optimizer': self.optimizer.state_dict(),'updates': self.updates}

    def restore(self, state):
        model = dict(state['model'])
        weight = model.get('body.0.weight')
        if weight is not None and weight.shape[1] < FEATURE_DIM:
            padded = torch.zeros(weight.shape[0], FEATURE_DIM, dtype=weight.dtype)
            padded[:, :weight.shape[1]] = weight
            model['body.0.weight'] = padded
            state = {**state, 'optimizer': self.optimizer.state_dict()}  # old Adam moments no longer match shapes
        self.load_state_dict(model)
        self.optimizer.load_state_dict(state['optimizer'])
        self.updates = state['updates']
        self.pending = None
        self.rollout.clear()