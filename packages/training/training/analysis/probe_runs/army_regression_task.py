from torch import Tensor
import torch.nn as nn

from training.analysis.probes.cross_player import army_totals_ch_indices
from training.bc.model.heads.pool import masked_mean_pool


# TODO: how do we get the indices for these channels robustly??
#   We need to know the number of channels that are inserted before
#   the 8 "army totals" channels (1 per player)
# TODO: this isn't valid syntax? how do I specify a slice
# leaderboard_army_ch = [0, :, :, :]


class ArmyRegressionTask:

    def __init__(self, dense_history_n: int):
        self.army_totals_ch = army_totals_ch_indices(dense_history_n)

    def extract_target(self, frame: dict[str, Tensor]):
        obs = frame["obs"]
        board_mask = frame["valid_mask"]
        army_totals = obs[self.army_totals_ch]
        # army_totals is broadcast. return first cell as 1-d tensor (len = player_count)
        return board_cells_flat(army_totals, board_mask)[:, 0]

    def loss(self, pred: Tensor, target: Tensor, alive: Tensor) -> Tensor:
        return ((pred[alive] - target[alive]) ** 2).mean()


class ArmyRegressionHead(nn.Module):

    def __init__(self, in_ch, n_players=8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, n_players, kernel_size=3, padding=1)

    def forward(self, x, board_mask):
        return masked_mean_pool(self.conv(x), board_mask)


# NOTE (2026-06-16): `board_mask` is the new canonical name for `valid_mask`
# TODO: Standardize this the canonical way to extract board cells ("unpadded")
# for leaf-operations like this
def board_cells_flat(x: Tensor, board_mask: Tensor) -> Tensor:
    """Gather the real (non-padding) cells of x, row-major.
    x: [..., H, W], board_mask: [1, H, W] bool  -> [..., n_valid]."""
    # [n_valid] flat indices
    idx = board_mask.reshape(-1).nonzero(as_tuple=True)[0]
    return x.flatten(-2)[..., idx] # [..., n_valid]
