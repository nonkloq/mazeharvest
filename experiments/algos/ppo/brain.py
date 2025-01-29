"""
Agent Network with Conv Layers, observation will be scalled to 16x16 Image (imagination)
and then it will work like CNN Policy
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
    pad_sequence,
)

from torch.distributions import Categorical
import sys


sys.path.append("../../")

from eutils import PERCEPTION_SIZE, STATE_SIZE, ACTION_SPACE


class Brain(nn.Module):
    def __init__(
        self,
    ) -> None:
        super(Brain, self).__init__()

        # Initial Observation Process
        self.bilstm_block = nn.Sequential(
            nn.GRU(
                input_size=PERCEPTION_SIZE,
                hidden_size=128,
                num_layers=1,
                # dropout=0.,
                bidirectional=True,
                batch_first=True,
                bias=True,
            ),
        )

        self.state_feats = nn.Sequential(
            nn.Linear(STATE_SIZE, 64),
            nn.GELU(),
        )

        self.final_fc = nn.Sequential(
            nn.Linear(320, 512, bias=True),
            nn.GELU(),
            nn.Linear(512, 512, bias=True),
            nn.GELU(),
        )

        self.policy_logits = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, ACTION_SPACE),
        )

        self.value_function = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def _forw_rnn(self, perception: list[torch.Tensor]):
        # Pad & Pack Perception sequence for batch processing
        lengths = torch.tensor([perc.size(0) for perc in perception], dtype=torch.uint8)
        paded_perception = pad_sequence(perception, batch_first=True)

        packed_perception = pack_padded_sequence(
            paded_perception, lengths, batch_first=True, enforce_sorted=False
        )

        out, _ = self.bilstm_block(packed_perception)
        out, _ = pad_packed_sequence(out, batch_first=True)

        return torch.tanh(out[:, -1, :])

    def forward(self, obs):
        perception, state = obs

        percim = self._forw_rnn(perception)
        stateim = self.state_feats(state)

        obs = torch.cat((percim, stateim), dim=-1)
        features = self.final_fc(obs)

        policy_logits = self.policy_logits(features)
        value = self.value_function(features).reshape(-1)  # (batch,1) -> (batch,)

        return Categorical(logits=policy_logits), value


if __name__ == "__main__":
    from eutils.mutil import get_param_count, print_param_counts

    testm = Brain()
    print("Trainable Param Count:", get_param_count(testm)[0])

    perceptions = [torch.randn(10, 6), torch.randn(5, 6)] + [
        torch.randn(5, 6) for _ in range(62)
    ]
    stats = torch.randn(64, 29)

    with torch.no_grad():
        out = testm((perceptions, stats))
        print(out[0])
        print(out[1].shape)

    print("\n")
    print_param_counts(testm)
