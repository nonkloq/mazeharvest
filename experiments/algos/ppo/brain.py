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
        self.bilstm_blocks = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=PERCEPTION_SIZE,
                    hidden_size=8,
                    num_layers=2,
                    # dropout=.3,
                    bidirectional=True,
                    batch_first=True,
                    bias=True,
                )
                for _ in range(16)
            ]
        )

        self._bilstm_outs = [None] * 16

        self.state_feats = nn.Sequential(
            nn.Linear(STATE_SIZE, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.Tanh(),
        )

        # 256 / 16 = 16, 16x16
        self.img_height, self.img_width = 16, 16

        self.conv = nn.Sequential(
            # 16x16 -> 10x10
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=8,
                stride=1,
                padding=4,
                dilation=2,
                padding_mode="circular",
                bias=True,
            ),
            nn.ReLU(),
            # 10x10 -> 8x8
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=0,
                dilation=1,
                bias=True,
            ),
            nn.ReLU(),
            # 8x8 -> 6x6
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.ReLU(),
        )

        self.final_fc = nn.Sequential(
            nn.Linear(6 * 6 * 32, 512, bias=True),
            nn.GELU(),
            nn.Linear(512, 256, bias=True),
            nn.Tanh(),
        )

        self.policy_logits = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_SPACE),
        )

        self.value_function = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def _forw_bilstm(self, perception: list[torch.Tensor]):
        # Pad & Pack Perception sequence for batch processing
        lengths = torch.tensor([perc.size(0) for perc in perception], dtype=torch.uint8)
        paded_perception = pad_sequence(perception, batch_first=True)

        packed_perception = pack_padded_sequence(
            paded_perception, lengths, batch_first=True, enforce_sorted=False
        )

        # num_layers *2
        # hn = torch.zeros(4, len(perception), 8)
        # cn = torch.zeros(4, len(perception), 8)

        for i, bilstm in enumerate(self.bilstm_blocks):
            out, (hn, cn) = bilstm(packed_perception)  # , (hn, cn))
            out, _ = pad_packed_sequence(out, batch_first=True)
            self._bilstm_outs[i] = out[:, -1, :]

            # no backprob through the previous blocks
            # hn = hn.detach()
            # cn = cn.detach()

        return torch.tanh(torch.cat(self._bilstm_outs, dim=-1))

    def forward(self, obs):
        perception, state = obs

        percim = self._forw_bilstm(perception)
        stateim = self.state_feats(state)

        img = percim + stateim

        img = img.reshape(-1, 1, self.img_height, self.img_width)
        img = self.conv(img)
        observation = img.view(img.size(0), -1)  # batch, 6*6*32

        features = self.final_fc(observation)  # + percim + stateim

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
