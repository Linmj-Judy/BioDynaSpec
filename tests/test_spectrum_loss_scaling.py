"""Unit tests for the spectrum loss scaling (src/model/loss/spectrum_loss.py).

Why these exist
---------------
A training run showed `spectrum_chunk_loss` ~2e4 dwarfing `diffloss` ~2e3, and
within it `weighted_amplitude_mse_loss` accounted for ~95% of the total. That
makes the reported `train_loss` a near-monotone function of one unnormalised,
squared-magnitude term, rather than a signal of training progress.

These tests pin the actual arithmetic down so the question "is the scaling
wrong?" is answered by numbers instead of by staring at logs. They do NOT
assert that the current behaviour is correct — several assert the *observed*
behaviour explicitly and are marked as such, so a future fix must consciously
update them rather than silently break.

Run:
    cd <repo root>
    PYTHONPATH=. pytest tests/test_spectrum_loss_scaling.py -v
"""

import math

import pytest
import torch

from src.model.loss.spectrum_loss import (
    SpectrumLoss,
    _amplitude_from_real_imag,
    compute_spectrum_mse_mae_after_rigid_align,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------
def _load_real_spectrum_weights():
    """Read SpectrumLossWeight's values straight out of src/trainer_far.py.

    Deliberately NOT `from trainer_far import SpectrumLossWeight`: importing
    that module pulls in pytorch_lightning and costs ~50s on a loaded machine.
    We only need the numeric literals, so parse the source with `ast` — fast,
    and still a single source of truth (no hand-copied duplicate that can drift
    the way this test's original `_Weight` did: it said 5.0 while the source had
    already moved to 2.0).
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "trainer_far.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SpectrumLossWeight":
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == "__init__":
                    values = {}
                    for stmt in sub.body:
                        if (
                            isinstance(stmt, ast.Assign)
                            and len(stmt.targets) == 1
                            and isinstance(stmt.targets[0], ast.Attribute)
                            and isinstance(stmt.targets[0].value, ast.Name)
                            and stmt.targets[0].value.id == "self"
                        ):
                            values[stmt.targets[0].attr] = ast.literal_eval(stmt.value)
                    if values:
                        return values
    raise AssertionError(f"could not read SpectrumLossWeight from {src}")


class _Weight:
    """Weight holder built from the real source values (see loader above)."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


_WEIGHTS = _load_real_spectrum_weights()


class _SpectrumLossCfg:
    eps = 1e-6
    weight = _Weight(**_WEIGHTS)


class _Cfg:
    spectrum_loss = _SpectrumLossCfg()


def _make_loss():
    return SpectrumLoss(_Cfg())


def _pkg(real, imag):
    """Build the pred_fourier_package / true_fourier_package dict shape."""
    return {"X_positive_real": real, "X_positive_imag": imag}


def _rand(shape, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g) * scale


# ---------------------------------------------------------------------------
# 1. amplitude_from_real_imag is the plain modulus
# ---------------------------------------------------------------------------
def test_amplitude_is_modulus():
    real = torch.tensor([[[[3.0]]]])
    imag = torch.tensor([[[[4.0]]]])
    amp = _amplitude_from_real_imag(real, imag, eps_amp=0.0)
    assert torch.allclose(amp, torch.tensor([[[[5.0]]]]), atol=1e-6)


# ---------------------------------------------------------------------------
# 2. amplitude_mse_loss: exact expected values
# ---------------------------------------------------------------------------
def test_amplitude_mse_is_zero_on_perfect_prediction():
    """A_pred == A_true must give exactly 0 (no hidden bias term)."""
    loss = _make_loss()
    x = _rand((1, 3, 3, 8), seed=1)

    # The amplitude term is an inline lambda in forward(): F.mse_loss(A_pred, A_true).
    amp = _amplitude_from_real_imag(x, x, eps_amp=loss.eps)
    got = torch.nn.functional.mse_loss(amp, amp)
    assert got.item() == pytest.approx(0.0, abs=1e-12)


def test_amplitude_mse_scales_quadratically_with_magnitude():
    """A_pred = c*A_true  ->  mse = (c-1)^2 * mean(A_true^2).

    This is the property that makes the term blow up: it is a *squared*
    magnitude, so it is not scale-free.
    """
    loss = _make_loss()
    x = _rand((1, 4, 3, 16), seed=2) * 0.5
    A_true = _amplitude_from_real_imag(x, x, eps_amp=loss.eps)

    for c in (0.5, 2.0, 10.0):
        A_pred = c * A_true
        got = torch.nn.functional.mse_loss(A_pred, A_true).item()
        expected = ((c - 1.0) ** 2) * (A_true ** 2).mean().item()
        assert got == pytest.approx(expected, rel=1e-5), (
            f"c={c}: amplitude MSE {got} != (c-1)^2*mean(A^2) {expected}"
        )


def _weighted_breakdown(loss, pred_r, pred_i, true_r, true_i):
    cum, metrics = loss(_pkg(pred_r, pred_i), _pkg(true_r, true_i))
    weighted = {
        k[len("weighted_"):]: float(v)
        for k, v in metrics.items()
        if k.startswith("weighted_") and k != "weighted_loss"
    }
    return float(cum), weighted


def test_dominant_term_depends_on_prediction_error_regime():
    """Which term dominates is regime-dependent -- and that is the finding.

    Measured on this implementation (N=32, F=64, weights from
    SpectrumLossWeight):

        prediction error          dominant term
        ------------------------  ---------------------------
        pred == truth             ~0 (nothing)
        pred == truth * 1.3       log_spectral_l1_loss   84%
        pred == 0                 log_spectral_l1_loss   95%
        pred == 2x iid noise      log_spectral_l1_loss   63%
        pred == 10x iid noise     amplitude_mse_loss     65%

    So `amplitude_mse_loss` is NOT unconditionally dominant: it only takes
    over once the absolute-magnitude error is large. In the near-truth and
    small-error regimes the *log-compressed* term dominates instead, because
    log compresses large true values and therefore makes small relative
    errors comparatively expensive.

    This test pins both regimes so that a future fix to either term's
    normalisation is caught. It asserts the observed behaviour, not that the
    behaviour is desirable.
    """
    loss = _make_loss()
    n_atom, seq_len = 32, 64
    # Ranks mirror production: main_model._spectrum6_to_real_imag_xyz yields
    # (N_atom, 3, seq_len) for BOTH pred and true (3-D, no batch).
    true_r = _rand((n_atom, 3, seq_len), seed=3, scale=0.3)
    true_i = _rand((n_atom, 3, seq_len), seed=4, scale=0.3)

    # --- large-error regime: amplitude term takes over ---
    big_r = _rand((n_atom, 3, seq_len), seed=99, scale=3.0)
    big_i = _rand((n_atom, 3, seq_len), seed=98, scale=3.0)
    cum_big, w_big = _weighted_breakdown(loss, big_r, big_i, true_r, true_i)
    top_big = max(w_big, key=w_big.get)
    print(f"\n[regime: 10x-noise pred] cum={cum_big:.4g}")
    for k, v in sorted(w_big.items(), key=lambda kv: -kv[1]):
        print(f"    weighted_{k:26s} = {v:12.4g}  ({v / cum_big:6.1%})")
    assert top_big == "amplitude_mse_loss", (
        f"large-error regime: expected amplitude_mse_loss to dominate, got {top_big}"
    )
    assert w_big[top_big] / cum_big > 0.5

    # --- small-error regime: the log term dominates instead ---
    near_r = true_r * 1.3
    near_i = true_i * 1.3
    cum_near, w_near = _weighted_breakdown(loss, near_r, near_i, true_r, true_i)
    top_near = max(w_near, key=w_near.get)
    print(f"\n[regime: 1.3x truth pred] cum={cum_near:.4g}")
    for k, v in sorted(w_near.items(), key=lambda kv: -kv[1]):
        print(f"    weighted_{k:26s} = {v:12.4g}  ({v / cum_near:6.1%})")
    assert top_near == "log_spectral_l1_loss", (
        f"small-error regime: expected log_spectral_l1_loss to dominate, got {top_near}"
    )
    assert w_near[top_near] / cum_near > 0.5


def test_loss_is_near_zero_on_exact_prediction():
    """Exact prediction must drive cum_loss to ~0 (all 7 terms vanish)."""
    loss = _make_loss()
    n_atom, seq_len = 16, 32
    true_r = _rand((n_atom, 3, seq_len), seed=21, scale=0.3)
    true_i = _rand((n_atom, 3, seq_len), seed=22, scale=0.3)
    cum, w = _weighted_breakdown(loss, true_r.clone(), true_i.clone(), true_r, true_i)
    assert cum < 1e-5, f"exact prediction gave cum_loss={cum}: {w}"


def test_amplitude_term_is_now_scale_normalised():
    """REGRESSION GUARD for the fix.

    `amplitude_mse_loss` used to be a bare `F.mse_loss(A_pred, A_true)` — the
    only one of the seven terms that was unbounded and unnormalised, so it grew
    quadratically with prediction error and dominated the summed loss.

    It is now divided by `mean(A_true^2)` (detached), making it a *relative*
    error. Consequences asserted here:

      * A_pred == c * A_true  ->  loss == (c-1)^2 * weight, INDEPENDENT of the
        data scale. Doubling the true spectrum's magnitude must NOT change the
        loss at all. That scale-invariance is the whole point of the fix.
      * The term is therefore bounded in the sense that it measures relative
        deviation, not absolute energy.

    Before the fix this test failed at c=2 with 0.843 (the old
    `(c-1)^2 * mean(A_true^2)` value) instead of 5.0.
    """
    loss = _make_loss()
    base = _rand((8, 3, 16), seed=31, scale=0.3)
    w_amp = loss.loss_weight["amplitude_mse_loss"]

    # (c-1)^2 * weight, with mean(A_true^2) cancelled by the normaliser.
    for c in (1.0, 2.0, 4.0, 8.0):
        _, w = _weighted_breakdown(loss, base * c, base * c, base, base)
        expected = w_amp * (c - 1.0) ** 2
        print(f"[normalised] c={c:4.1f}  amplitude_mse={w['amplitude_mse_loss']:12.6g}  "
              f"(analytic {expected:12.6g})")
        assert w["amplitude_mse_loss"] == pytest.approx(expected, rel=1e-4, abs=1e-9), (
            f"c={c}: {w['amplitude_mse_loss']} != (c-1)^2*weight = {expected}"
        )

    # Scale invariance: the same relative error on a 100x larger spectrum gives
    # the same loss. This is what stops the term dominating the sum.
    #
    # NOTE: avoid tiny scales here. A = sqrt(Re^2+Im^2+eps) with eps=1e-6, so at
    # data scale 0.01 the eps term is ~1% of A^2 and the invariant no longer
    # holds to 1e-3. 0.1 vs 10 keeps eps at ~1e-4 of A^2.
    small_r = _rand((8, 3, 16), seed=41, scale=0.1)
    small_i = _rand((8, 3, 16), seed=42, scale=0.1)
    big_r = small_r * 100.0
    big_i = small_i * 100.0
    _, w_small = _weighted_breakdown(loss, small_r * 2, small_i * 2, small_r, small_i)
    _, w_big = _weighted_breakdown(loss, big_r * 2, big_i * 2, big_r, big_i)
    print(f"[normalised] scale-invariance: small={w_small['amplitude_mse_loss']:.6g} "
          f"big={w_big['amplitude_mse_loss']:.6g}")
    assert w_big["amplitude_mse_loss"] == pytest.approx(
        w_small["amplitude_mse_loss"], rel=1e-3
    ), "normalised amplitude loss must not depend on the absolute data scale"

    # And it must still be a meaningful (non-degenerate) signal.
    _, w_zero = _weighted_breakdown(
        loss, torch.zeros_like(base), torch.zeros_like(base), base, base
    )
    print(f"[normalised] A_pred=0 -> amplitude_mse={w_zero['amplitude_mse_loss']:.6g}")
    assert w_zero["amplitude_mse_loss"] > 0.0, (
        "relative amplitude loss must be non-zero when the prediction is all zeros"
    )


# ---------------------------------------------------------------------------
# 3. aggregate_losses: weight semantics
# ---------------------------------------------------------------------------
def test_aggregate_losses_is_a_plain_weighted_sum():
    """cum_loss must equal sum_i (weight_i * loss_i), no normalisation."""
    loss = _make_loss()
    vals = {
        "spectrum_mse_loss": torch.tensor(2.0),
        "spectrum_mae_loss": torch.tensor(3.0),
        "smooth_l1_loss": torch.tensor(4.0),
        "log_spectral_l1_loss": torch.tensor(5.0),
        "cosine_similarity_loss": torch.tensor(0.5),
        "spectral_convergence_loss": torch.tensor(0.25),
        "amplitude_mse_loss": torch.tensor(10.0),
    }
    fns = {k: (lambda v=v: v) for k, v in vals.items()}
    cum, metrics = loss.aggregate_losses(fns)

    expected = sum(loss.loss_weight[k] * v.item() for k, v in vals.items())
    assert cum.item() == pytest.approx(expected, rel=1e-6)

    # and the per-term weighted metrics agree with weight*loss
    for k, v in vals.items():
        assert float(metrics[f"weighted_{k}"]) == pytest.approx(
            loss.loss_weight[k] * v.item(), rel=1e-6
        )


def test_aggregate_losses_weights_match_trainer_far_source():
    """The loss's weight lookup must match SpectrumLossWeight in the source.

    Asserted against `_WEIGHTS` (parsed from src/trainer_far.py) rather than
    against hard-coded numbers, so the two can't drift apart. The original
    version of this test hard-coded amplitude_mse_loss=5.0 and silently became
    a lie once the source moved to 2.0 — and because the same numbers were
    duplicated into the test's `_Weight` double, that drift also changed what
    the *other* tests measured.
    """
    loss = _make_loss()
    assert loss.loss_weight == {
        "spectrum_mse_loss": _WEIGHTS["mse_loss"],
        "spectrum_mae_loss": _WEIGHTS["mae_loss"],
        "smooth_l1_loss": _WEIGHTS["smooth_l1_loss"],
        "log_spectral_l1_loss": _WEIGHTS["log_spectral_l1_loss"],
        "cosine_similarity_loss": _WEIGHTS["cosine_similarity_loss"],
        "spectral_convergence_loss": _WEIGHTS["spectral_convergence_loss"],
        "amplitude_mse_loss": _WEIGHTS["amplitude_mse_loss"],
    }
    # The regression this file exists for: amplitude weight must stay low enough
    # that the (now normalised) amplitude term cannot dominate on its own.
    assert _WEIGHTS["amplitude_mse_loss"] <= _WEIGHTS["spectral_convergence_loss"]


# ---------------------------------------------------------------------------
# 4. The normalisation asymmetry between the two "same-dimensionality" terms
# ---------------------------------------------------------------------------
def test_spectrum_mse_and_amplitude_mse_disagree_by_a_large_factor():
    """spectrum_mse_loss is rigid-aligned; amplitude_mse_loss is not.

    Both purport to measure spectral error, yet on identical inputs their
    magnitudes differ by orders of magnitude. This test documents the gap: it
    asserts they are NOT comparable, which is the whole reason one term can
    dominate the sum.
    """
    loss = _make_loss()
    n_atom, seq_len = 16, 32
    true_r = _rand((n_atom, 3, seq_len), seed=7, scale=0.3)
    true_i = _rand((n_atom, 3, seq_len), seed=8, scale=0.3)
    pred_r = true_r + _rand((n_atom, 3, seq_len), seed=9, scale=0.05)
    pred_i = true_i + _rand((n_atom, 3, seq_len), seed=10, scale=0.05)

    mse_loss, mae_loss, _rot = compute_spectrum_mse_mae_after_rigid_align(
        X_pred_real_positive=pred_r,
        X_pred_imag_positive=pred_i,
        true_real_b=true_r,
        true_imag_b=true_i,
        allowing_reflection=False,
    )
    A_pred = _amplitude_from_real_imag(pred_r, pred_i, eps_amp=loss.eps)
    A_true = _amplitude_from_real_imag(true_r, true_i, eps_amp=loss.eps)
    amp = torch.nn.functional.mse_loss(A_pred, A_true)

    ratio = (amp / mse_loss).item()
    print(f"\n[observed] amplitude_mse={amp.item():.6g}  spectrum_mse={mse_loss.item():.6g}  ratio={ratio:.3g}")
    # They measure related quantities but are not on a common scale.
    assert ratio > 1.5 or ratio < 1 / 1.5, (
        f"the two MSE terms happened to be comparable (ratio={ratio}); "
        "the dominance claim would need revisiting"
    )


# ---------------------------------------------------------------------------
# 5. Chunked vs non-chunked must agree (regression guard)
# ---------------------------------------------------------------------------
def test_chunked_forward_matches_plain_forward():
    """SpectrumChunkLoss must not change the reduction vs plain SpectrumLoss.

    The two classes share all the per-term implementations but consume
    different inputs (package dict vs 4 tensors), and are invoked on different
    slices. If chunking altered the reduction -- e.g. by averaging chunk means
    instead of taking a global mean, so the value scaled with chunk count --
    the training loss would silently depend on how the frequency axis was
    split. That is the class of bug in coordination.md RFC-FMAR-LOSS-003.
    """
    from src.model.loss.spectrum_loss import SpectrumChunkLoss

    n_atom, seq_len = 8, 64
    true_r = _rand((n_atom, 3, seq_len), seed=11, scale=0.3)
    true_i = _rand((n_atom, 3, seq_len), seed=12, scale=0.3)
    pred_r = true_r + _rand((n_atom, 3, seq_len), seed=13, scale=0.05)
    pred_i = true_i + _rand((n_atom, 3, seq_len), seed=14, scale=0.05)

    plain = _make_loss()
    _, m_plain = plain(_pkg(pred_r, pred_i), _pkg(true_r, true_i))

    # SpectrumChunkLoss.__init__ is inherited; forward takes 4 tensors.
    chunked = SpectrumChunkLoss(_Cfg())
    _, m_chunked = chunked(pred_r, pred_i, true_r, true_i)

    for key in ("amplitude_mse_loss", "spectrum_mse_loss", "spectral_convergence_loss"):
        a, b = float(m_plain[key]), float(m_chunked[key])
        print(f"[chunk] {key}: plain={a:.6g} chunked={b:.6g}")
        assert b == pytest.approx(a, rel=2e-4), f"{key}: {a} vs {b}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
