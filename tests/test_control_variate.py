import torch

from drgw.model import DRGW
from drgw.training import training_objective


def test_zero_edit_control_variate_cannot_reward_detector_bias():
    torch.manual_seed(41)
    model = DRGW(hidden_dim=16, watermark_dim=8, gin_layers=1, flow_layers=2)
    adj = (torch.rand(2, 12, 12) < 0.25).float().triu(1)
    adj = adj + adj.transpose(-1, -2)
    mask = torch.ones(2, 12, dtype=torch.bool)
    cfg = dict(alpha=0.1, edit_budget=0, robustness_rates=[0.3],
               robust_control_variate=True, augmentation_rate=0.1,
               orthogonality_weight=0.1, feature_weight=1., variance_weight=1.,
               nll_weight=5., stage3_aux_weight=0.1)
    for step in (0, 1):  # edge flips and node deletion share the null plan.
        model.zero_grad(set_to_none=True)
        _, parts = training_objective(model, adj, mask, "stage3", cfg, step)
        assert parts["robustness"].item() == 0.0
        parts["robustness"].backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                assert torch.count_nonzero(parameter.grad) == 0
