"""A zero prediction must not produce a NaN gradient in the rigid alignment.

`compute_spectrum_mse_mae_after_rigid_align` (used by the `spectrum_mse` and
`spectrum_mae` terms of `SpectrumChunkLoss`) first estimates a rotation from the
DC / 0 Hz component. That goes through `align_pred_to_true`, which forms the
covariance A^T A and takes its SVD.

When the prediction is exactly zero the covariance is the zero matrix, all three
singular values are equal, and `torch.linalg.svd`'s backward divides by
(s_i^2 - s_j^2) = 0. The forward value stays finite, so the loss *looks* fine --
but the gradient becomes NaN and poisons every parameter on the first optimizer
step.

A freshly-initialised denoiser emits exactly 0 (adaLN-Zero), so this fires on
step 0 of every run. Guard added in `rigid_align_spectrum_by_dc`.
"""

import torch

from src.model.loss.spectrum_loss import (
    compute_spectrum_mse_mae_after_rigid_align,
    rigid_align_spectrum_by_dc,
)

N_ATOM, L = 7, 11


def _zeros():
    return (torch.zeros(N_ATOM, 3, L), torch.zeros(N_ATOM, 3, L))


def _true():
    torch.manual_seed(0)
    return (torch.randn(N_ATOM, 3, L), torch.randn(N_ATOM, 3, L))


def test_zero_prediction_gives_finite_gradients():
    """The regression: pred == 0 must not NaN the gradient."""
    true_real, true_imag = _true()
    pred_real, pred_imag = _zeros()
    pred_real = pred_real.requires_grad_(True)
    pred_imag = pred_imag.requires_grad_(True)

    mse, mae, _ = compute_spectrum_mse_mae_after_rigid_align(
        pred_real, pred_imag, true_real, true_imag, allowing_reflection=False
    )
    mse.backward()

    assert torch.isfinite(mse)
    assert torch.isfinite(pred_real.grad).all(), "NaN gradient on a zero prediction"
    # With pred == 0 the aligned prediction is still 0, so the MSE reduces to
    # mean(|true|^2).
    expected = (true_real ** 2 + true_imag ** 2).mean()
    torch.testing.assert_close(mse, expected, rtol=1e-5, atol=1e-6)


def test_zero_prediction_returns_identity_rotation():
    """A zero prediction carries no orientation, so alignment is a no-op."""
    true_real, true_imag = _true()
    pred_real, pred_imag = _zeros()

    out_real, out_imag, rot = rigid_align_spectrum_by_dc(
        pred_real, pred_imag, true_real, true_imag, allowing_reflection=False
    )
    torch.testing.assert_close(rot, torch.eye(3), rtol=0, atol=0)
    torch.testing.assert_close(out_real, pred_real, rtol=0, atol=0)
    torch.testing.assert_close(out_imag, pred_imag, rtol=0, atol=0)


def test_nonzero_prediction_still_aligns():
    """The guard must not disable alignment for ordinary predictions."""
    true_real, true_imag = _true()
    torch.manual_seed(1)
    pred_real = torch.randn(N_ATOM, 3, L).requires_grad_(True)
    pred_imag = torch.randn(N_ATOM, 3, L).requires_grad_(True)

    out_real, _, rot = rigid_align_spectrum_by_dc(
        pred_real, pred_imag, true_real, true_imag, allowing_reflection=False
    )
    # A genuine rotation was applied: not the identity, and still a rotation.
    assert not torch.allclose(rot, torch.eye(3), atol=1e-6)
    torch.testing.assert_close(
        rot @ rot.transpose(-1, -2), torch.eye(3), rtol=1e-5, atol=1e-6
    )
    assert torch.isfinite(out_real).all()


def test_batched_zero_prediction_gives_finite_gradients():
    """Same guard on the batched (B, N, 3, L) path."""
    true_real, true_imag = _true()
    pred_real = torch.zeros(2, N_ATOM, 3, L, requires_grad=True)
    pred_imag = torch.zeros(2, N_ATOM, 3, L, requires_grad=True)

    mse, _, rot = compute_spectrum_mse_mae_after_rigid_align(
        pred_real, pred_imag, true_real.unsqueeze(0), true_imag.unsqueeze(0),
        allowing_reflection=False,
    )
    mse.backward()

    assert torch.isfinite(mse)
    assert torch.isfinite(pred_real.grad).all()
    assert rot.shape == (2, 3, 3)
    torch.testing.assert_close(rot[0], torch.eye(3), rtol=0, atol=0)
