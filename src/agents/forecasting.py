"""Per-agent probabilistic recurrent model for black-box price forecasts."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ProbabilisticForecastRNN(nn.Module):
    """Encode recent returns and agent behavior, then predict return distributions."""

    def __init__(self, behavior_dim: int, hidden_dim: int = 32, learning_rate: float = 3e-4):
        super().__init__()
        self.behavior_dim = behavior_dim
        self.encoder = nn.GRU(input_size=3, hidden_size=hidden_dim, batch_first=True)
        self.context = nn.Sequential(nn.Linear(hidden_dim + behavior_dim, 64), nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(65, 64), nn.Tanh(), nn.Linear(64, 2))
        final = self.decoder[-1]
        nn.init.normal_(final.weight, mean=0.0, std=0.01)
        nn.init.zeros_(final.bias)
        with torch.no_grad():
            final.bias[1] = -4.6  # initial daily return uncertainty is about one percent
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.updates = 0
        self.loss_sum = 0.0
        self.last_loss = 0.0

    def distribution(self, market_sequence, behavior, horizon: int):
        sequence = torch.as_tensor(market_sequence, dtype=torch.float32).unsqueeze(0)
        traits = torch.as_tensor(behavior, dtype=torch.float32).unsqueeze(0)
        _, hidden = self.encoder(sequence)
        context = self.context(torch.cat((hidden[-1], traits), dim=-1))
        time = torch.linspace(1.0 / horizon, 1.0, horizon).reshape(1, horizon, 1)
        repeated = context.unsqueeze(1).expand(-1, horizon, -1)
        output = self.decoder(torch.cat((repeated, time), dim=-1)).squeeze(0)
        mean = output[:, 0]
        std = F.softplus(output[:, 1]).clamp(5e-4, 0.08)
        return mean, std

    def predict(self, market_sequence, behavior, horizon: int):
        self.eval()
        with torch.no_grad():
            mean, std = self.distribution(market_sequence, behavior, horizon)
        return mean.numpy(), std.numpy()

    def train_forecast(self, market_sequence, behavior, target_returns):
        self.train()
        target = torch.as_tensor(target_returns, dtype=torch.float32)
        mean, std = self.distribution(market_sequence, behavior, len(target_returns))
        loss = (torch.log(std) + 0.5 * ((target - mean) / std).square()).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        self.last_loss = float(loss.detach())
        self.loss_sum += self.last_loss
        self.updates += 1
        return self.last_loss

    def snapshot(self):
        return {'model': self.state_dict(), 'optimizer': self.optimizer.state_dict(), 'updates': self.updates, 'loss_sum': self.loss_sum, 'last_loss': self.last_loss}

    def restore(self, state):
        self.load_state_dict(state['model'])
        if 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
        self.updates = int(state.get('updates', 0))
        self.loss_sum = float(state.get('loss_sum', 0.0))
        self.last_loss = float(state.get('last_loss', 0.0))
