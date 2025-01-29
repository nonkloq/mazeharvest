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

        self.bilstm_block = nn.Sequential(
            nn.GRU(
                input_size=PERCEPTION_SIZE,
                hidden_size=128,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
                bias=True,
            ),
        )

        self.state_feats = nn.Sequential(
            nn.Linear(STATE_SIZE, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
        )

        self.extract_features = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
        )

        self.advantage_function = nn.Sequential(
            nn.Linear(256, 256), nn.GELU(), nn.Linear(256, self.n * self.n_atoms)
        )

        self.value_function = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Linear(128, self.n_atoms)
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
        return torch.nn.functional.gelu((out[:, -1, :]))

    def get_action(self, obs, actions=None):
        q_values, pmfs = self(obs)

        if actions is None:
            actions = torch.argmax(q_values, 1)

        batch_idx = torch.arange(q_values.size(0))
        return q_values[batch_idx, actions], actions, pmfs[batch_idx, actions]

    def forward(self, obs):
        perception, state = obs

        percim = self._forw_rnn(perception)
        stateim = self.state_feats(state)

        obs = torch.cat((percim, stateim), dim=-1)
        features = self.extract_features(obs)

        advantages = self.advantage_function(features).view(-1, self.n, self.n_atoms)
        values = self.value_function(features).view(-1, 1, self.n_atoms)

        logits = values + advantages - advantages.mean(dim=1, keepdim=True)

        pmfs = torch.softmax(logits, dim=1)

        q_values = (pmfs * self.atoms).sum(dim=2)

        return q_values, pmfs


if __name__ == "__main__":
    import numpy as np
    from eutils.mutil import get_param_count, print_param_counts

    testm = Brain(n_atoms=51)
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
