"""Correctness tests for the x_0 prediction returned by `training_losses`.

`training_losses` returns `terms["pred_xstart"]`, which downstream code
(`SpectrumDiffLoss.forward` -> `recover_spectrum_from_tokens` -> `SpectrumChunkLoss`)
treats as a *clean spectrum* estimate.  It therefore has to be an x_0 prediction
under every `model_mean_type`, not just `START_X`.

Regression guard for the bug where `pred_xstart` was set to the raw network
output while `model_mean_type == EPSILON`, so the training path fed pure noise
into the spectral reconstruction loss.
"""

import numpy as np
import pytest
import torch as th

from src.model.diffusion import gaussian_diffusion as gd
from src.model.diffusion.respace import SpacedDiffusion, space_timesteps


def _make_diffusion(predict_xstart, num_timesteps=1000):
    betas = gd.get_named_beta_schedule("cosine", num_timesteps)
    # Mirror create_diffusion(): an empty respacing string means "use every step".
    return SpacedDiffusion(
        use_timesteps=space_timesteps(num_timesteps, [num_timesteps]),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=gd.ModelVarType.FIXED_LARGE,
        loss_type=gd.LossType.MSE,
    )


@pytest.mark.parametrize("predict_xstart", [True, False])
def test_pred_xstart_is_clean_x0_for_a_perfect_model(predict_xstart):
    """A model that predicts its target perfectly must yield pred_xstart == x_0.

    Tolerance: in EPSILON mode the conversion is
        x_0 = x_t / sqrt(alpha_bar) - sqrt(1/alpha_bar - 1) * eps
    and at t = 999 the cosine schedule gives alpha_bar ~ 1e-5, so both terms are
    O(300) and cancel down to an O(1) result.  `_predict_xstart_from_eps` also
    casts the schedule coefficients to float32 (`diffusion_utils.py`), so the
    cancellation costs ~1e-3 absolute.  That is a precision limit of the
    schedule dtype, not a correctness bug, hence the looser atol here.
    """
    th.manual_seed(0)
    diffusion = _make_diffusion(predict_xstart)

    x_start = th.randn(4, 8, 6)
    t = th.tensor([0, 1, 250, 999])
    noise = th.randn_like(x_start)

    # The target the network is trained against for this model_mean_type.
    if predict_xstart:
        perfect_output = x_start
    else:
        perfect_output = noise

    def model(x_t, timesteps, **kwargs):
        return perfect_output

    terms = diffusion.training_losses(model, x_start, t, noise=noise)

    assert terms["mse"].max().item() == pytest.approx(0.0, abs=1e-6)
    atol = 1e-5 if predict_xstart else 5e-3
    th.testing.assert_close(terms["pred_xstart"], x_start, atol=atol, rtol=1e-4)


def test_epsilon_mode_pred_xstart_reconstructs_x0_from_noise():
    """eps -> x_0 conversion must invert q_sample exactly."""
    th.manual_seed(1)
    diffusion = _make_diffusion(predict_xstart=False)

    x_start = th.randn(3, 5, 6)
    t = th.tensor([10, 500, 990])
    noise = th.randn_like(x_start)

    x_t = diffusion.q_sample(x_start, t, noise=noise)
    recovered = diffusion._predict_xstart_from_eps(x_t, t, noise)

    th.testing.assert_close(recovered, x_start, atol=1e-4, rtol=1e-4)


def test_epsilon_mode_pred_xstart_is_not_the_raw_output():
    """Guards the exact regression: pred_xstart must not equal the eps output."""
    th.manual_seed(2)
    diffusion = _make_diffusion(predict_xstart=False)

    x_start = th.randn(2, 6, 6)
    t = th.tensor([100, 700])
    noise = th.randn_like(x_start)

    captured = {}

    def model(x_t, timesteps, **kwargs):
        captured["out"] = noise
        return noise

    terms = diffusion.training_losses(model, x_start, t, noise=noise)

    assert not th.allclose(terms["pred_xstart"], captured["out"], atol=1e-3), (
        "pred_xstart is the raw eps output; it should have been converted to x_0"
    )
    # And it must be a genuine x_0 estimate, i.e. close to the real clean signal.
    th.testing.assert_close(terms["pred_xstart"], x_start, atol=1e-4, rtol=1e-4)
