"""`block_type` must actually select the denoiser block class.

Regression guard: `FrequencyComponentDenoiseNet` accepted `block_type` and
validated it, but the `ResidueDiTBlock` branch was commented out and every
setting produced a plain `ResBlock`. With `block_type="residue"` (what all the
configs ask for) the diffusion denoiser therefore had no attention across
residues at all — each token was processed independently.

Construction only, so this needs no GPU and no flash-attn.
"""

import pytest

from src.model.diffusion.denoise_net import (
    FrequencyComponentDenoiseNet,
    ResBlock,
    ResidueDiTBlock,
)


def _build(block_type, num_dit_blocks=3):
    return FrequencyComponentDenoiseNet(
        in_channels=6,
        model_channels=32,
        out_channels=6,
        z_channels=16,
        num_dit_blocks=num_dit_blocks,
        num_heads=4,
        mlp_ratio=4,
        block_type=block_type,
        use_checkpoint=False,
    )


def test_residue_block_type_builds_residue_dit_blocks():
    net = _build("residue")
    assert len(net.res_blocks) == 3
    assert all(isinstance(b, ResidueDiTBlock) for b in net.res_blocks)
    # Residue attention is the whole point: without it the block is a bare MLP.
    assert any(isinstance(b, ResidueDiTBlock) and hasattr(b, "attn") for b in net.res_blocks)


def test_mlp_block_type_builds_res_blocks():
    net = _build("mlp")
    assert len(net.res_blocks) == 3
    assert all(isinstance(b, ResBlock) for b in net.res_blocks)


def test_block_types_build_different_networks():
    """The two settings must not collapse onto the same architecture."""
    residue_kinds = {type(b).__name__ for b in _build("residue").res_blocks}
    mlp_kinds = {type(b).__name__ for b in _build("mlp").res_blocks}
    assert residue_kinds == {"ResidueDiTBlock"}
    assert mlp_kinds == {"ResBlock"}
    assert residue_kinds != mlp_kinds


def test_unknown_block_type_is_rejected():
    with pytest.raises(ValueError, match="block_type"):
        _build("bogus")


def test_block_count_matches_num_dit_blocks():
    net = _build("residue", num_dit_blocks=5)
    assert len(net.res_blocks) == 5
