"""Fourier-domain metrics for the frozen SpectrumVAE interface."""

from __future__ import annotations

from typing import Any

import torch


def spectrum_metrics(
    vae: Any,
    recon_tokens: torch.Tensor,
    target_real: torch.Tensor,
    target_imag: torch.Tensor,
) -> dict[str, float]:
    """Report normalized-token and physical positive-frequency MSE."""
    target_tokens = vae.prepare_spectrum_input(target_real, target_imag)
    normalized_mse = torch.mean((recon_tokens.float() - target_tokens.float()) ** 2)
    pred_real, pred_imag = vae.recover_spectrum_from_tokens(recon_tokens)
    physical_mse = torch.mean(
        torch.cat(
            [
                (pred_real.float() - target_real.float()).reshape(-1),
                (pred_imag.float() - target_imag.float()).reshape(-1),
            ]
        )
        ** 2
    )
    return {
        "normalized_token_spectrum_mse": float(normalized_mse.detach().cpu()),
        "physical_positive_spectrum_mse": float(physical_mse.detach().cpu()),
    }
