"""The diffusion head must keep its own initialisation.

`ProteinSpectrumFAR.__init__` builds `SpectrumDiffLoss` (whose denoiser does its
own adaLN-Zero + zero-init of `final_layer.linear`) and *then* calls
`self.initialize_weights()`, which runs `self.apply(self._init_weights)`.

`nn.Module.apply` is recursive, so that blanket xavier re-init also overwrote
`final_layer.linear`. The denoiser therefore started from a large random output
instead of 0, and since `pred_xstart` is what `SpectrumChunkLoss` treats as the
predicted clean spectrum, the reconstruction losses started at ~1e3-1e4 and the
diffusion loss at ~1e3 rather than ~0.07.

These tests pin the invariant: after construction, a freshly built model's
denoiser must still output ~0.
"""

import pytest
import torch

from src.model.diffusion.denoise_net import FrequencyComponentDenoiseNet


def _tiny_denoiser():
    return FrequencyComponentDenoiseNet(
        in_channels=6,
        model_channels=32,
        out_channels=6,
        z_channels=16,
        num_dit_blocks=2,
        num_heads=4,
        mlp_ratio=4,
        block_type="mlp",  # avoid flash-attn so this runs on CPU
        use_checkpoint=False,
    )


def test_fresh_denoiser_outputs_zero():
    """Baseline: the denoiser's own init zeroes its output layer."""
    net = _tiny_denoiser().eval()
    assert net.final_layer.linear.weight.abs().max().item() == 0.0
    assert net.final_layer.linear.bias.abs().max().item() == 0.0

    with torch.no_grad():
        out = net(
            torch.randn(7, 11, 6), torch.tensor([5]),
            torch.randn(7, 11, 16), torch.full((7, 11, 1), 3.0),
        )
    assert out.abs().max().item() == 0.0


def test_blanket_apply_destroys_the_zero_init():
    """Documents the mechanism: a recursive apply() wipes the zero init.

    This is what `initialize_weights()` does to the head when it is not restored,
    so the test guards the *reason* the restore exists.
    """
    net = _tiny_denoiser()

    def reinit(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    net.apply(reinit)
    assert net.final_layer.linear.weight.abs().max().item() > 0.0, (
        "expected the blanket xavier init to overwrite final_layer.linear"
    )


def test_restore_helper_reinstates_the_snapshot():
    """The restore path used by initialize_weights() must undo the clobbering."""
    from src.model.diffusion.denoise_net import FrequencyComponentDenoiseNet as _N

    net = _tiny_denoiser()
    snapshot = {n: p.detach().clone() for n, p in net.named_parameters()}

    def reinit(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    net.apply(reinit)
    assert net.final_layer.linear.weight.abs().max().item() > 0.0

    with torch.no_grad():
        for n, p in net.named_parameters():
            p.copy_(snapshot[n])

    assert net.final_layer.linear.weight.abs().max().item() == 0.0
    assert net.final_layer.linear.bias.abs().max().item() == 0.0


def test_protein_spectrum_far_preserves_head_init():
    """End-to-end: constructing the full model must leave a zero-output head.

    Skipped if the heavy model cannot be built in this environment.
    """
    pytest.importorskip("src.protein_spectrum_far")
    try:
        from src.protein_spectrum_far import protein_spectrum_far_base
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"cannot import ProteinSpectrumFAR: {exc}")

    try:
        model = protein_spectrum_far_base(
            encoder_depth=1, encoder_num_heads=4, decoder_embed_dim=64,
            decoder_depth=1, decoder_num_heads=4, buffer_size=8,
            diffloss_d=2, diffloss_w=64, num_sampling_steps="50",
            denoise_block_type="mlp", denoise_use_checkpoint=False,
            use_checkpoint=False, learn_sigma=False, predict_xstart=True,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"cannot construct ProteinSpectrumFAR: {exc}")

    denoiser = model.spectrum_diffloss.net
    lw = denoiser.final_layer.linear.weight
    assert lw.abs().max().item() == 0.0, (
        "the blanket initialize_weights() clobbered the diffusion head's "
        "zero-initialised output layer; the denoiser will start from a large "
        "random output and the spectral losses will explode from step 0."
    )
