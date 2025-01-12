import sys

import torch
from torch import nn

sys.path.append("../../")


from eutils import ACTION_SPACE, PERCEPTION_SIZE, STATE_SIZE
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


class Brain(nn.Module):
    def __init__(self, n_atoms=51, v_min=-100, v_max=100):
        super().__init__()
        self.n_atoms = n_atoms
        self.n = ACTION_SPACE

        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))

        self.bilstm_blocks = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=PERCEPTION_SIZE,
                    hidden_size=32,
                    num_layers=1,
                    # dropout=.3,
                    bidirectional=True,
                    batch_first=True,
                    bias=True,
                )
                for _ in range(4)
            ]
        )

        self.state_feats = nn.Sequential(
            nn.Linear(STATE_SIZE, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.Tanh(),
        )

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

        self.extract_features = nn.Sequential(
            nn.Linear(6 * 6 * 32, 512), nn.GELU(), nn.Linear(512, 256), nn.Tanh()
        )

        self.advantage_function = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, self.n * self.n_atoms)
        )

        self.value_function = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, self.n_atoms)
        )

        self._bilstm_outs = [None] * 4

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
            out, (hn, cn) = bilstm(packed_perception)  # (hn, cn))
            out, _ = pad_packed_sequence(out, batch_first=True)
            self._bilstm_outs[i] = out[:, -1, :]

            # no backprob through the previous blocks
            # hn = hn.detach()
            # cn = cn.detach()

        return torch.tanh(torch.cat(self._bilstm_outs, dim=-1))

    def get_action(self, obs, actions=None):
        q_values, pmfs = self(obs)

        if actions is None:
            actions = torch.argmax(q_values, 1)

        batch_idx = torch.arange(q_values.size(0))
        return q_values[batch_idx, actions], actions, pmfs[batch_idx, actions]

    def forward(self, obs):
        perception, state = obs

        percim = self._forw_bilstm(perception)
        stateim = self.state_feats(state)

        img = percim + stateim
        img = img.reshape(-1, 1, 16, 16)
        img = self.conv(img)
        observation = img.view(img.size(0), -1)

        features = self.extract_features(observation) + percim + stateim

        advantages = self.advantage_function(features).view(-1, self.n, self.n_atoms)
        values = self.value_function(features).view(-1, 1, self.n_atoms)

        logits = values + advantages - advantages.mean(dim=1, keepdim=True)

        pmfs = torch.softmax(logits, dim=-1)

        q_values = (pmfs * self.atoms).sum(dim=2)

        return q_values, pmfs


if __name__ == "__main__":
    import numpy as np
    from eutils.mutil import get_param_count, print_param_counts

    testm = Brain()
    print("Trainable Param Count:", get_param_count(testm)[0])

    perceptions = [torch.randn(np.random.randint(7, 21), 6) for _ in range(64)]
    stats = torch.randn(64, 29)

    with torch.no_grad():
        out = testm((perceptions, stats))
        print(out[0].shape)
        print(out[1].shape)

        out = testm.get_action((perceptions, stats))
        print(out[0].shape)
        print(out[1].shape)
        print(out[2].shape)

    print("\n")
    print_param_counts(testm)
