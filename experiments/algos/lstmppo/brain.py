import torch
from torch import nn
from torch.distributions import Categorical
import sys

sys.path.append("../../")
from eutils.mutil import initialize_orthogonal


class Brain(nn.Module):
    def __init__(
        self,
        state_size,
        hidden_size,
        observation_dim,
        action_dim,
        inp_parser_features: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_size = state_size

        self.input_parser = nn.Sequential(
            nn.LayerNorm(observation_dim),
            nn.Linear(observation_dim, inp_parser_features),
            nn.ReLU(),
            nn.Linear(inp_parser_features, inp_parser_features),
            nn.ReLU(),
        )
        initialize_orthogonal(self.input_parser, 2.0)

        self.rnn_layer = nn.LSTM(
            inp_parser_features,
            self.state_size,
            batch_first=True,
        )

        initialize_orthogonal(self.rnn_layer, 2.0)

        self.hidden_layer = nn.Sequential(
            nn.Linear(state_size, self.hidden_size), nn.ReLU()
        )
        initialize_orthogonal(self.hidden_layer, 2.0)

        self.policy_hidden = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU()
        )
        initialize_orthogonal(self.policy_hidden, 2.0)

        self.value_hidden = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU()
        )
        initialize_orthogonal(self.value_hidden, 2.0)

        self.policy_function = nn.Linear(self.hidden_size, action_dim)
        initialize_orthogonal(self.policy_function, 0.01)
        self.value_function = nn.Linear(self.hidden_size, 1)
        initialize_orthogonal(self.policy_function, 1)

    def forward(
        self,
        obs: torch.tensor,
        hidden_state: torch.tensor,
    ):
        batch_size, sequences, input_dim = obs.shape
        flat_obs = obs.reshape(-1, input_dim)
        features = self.input_parser(flat_obs)

        features_seq = features.view(batch_size, sequences, -1)

        hsn, hidden_state = self.rnn_layer(features_seq, hidden_state)

        flat_hsn = hsn.reshape(batch_size * sequences, -1)

        hidden_logits = self.hidden_layer(flat_hsn)
        hid_policy = self.policy_hidden(hidden_logits)
        hid_value = self.value_hidden(hidden_logits)

        policy_logits = self.policy_function(hid_policy)
        value = self.value_function(hid_value).reshape(-1)

        return Categorical(logits=policy_logits), value, hidden_state

    def init_states(self, batch_size: int, device: torch.device) -> tuple:
        hxs = torch.zeros(
            (batch_size, self.state_size),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        cxs = torch.zeros(
            (batch_size, self.state_size),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        return (hxs, cxs)
