"""Every shipped training config must use START_X (x_0) prediction.

The diffusion head's `pred_xstart` output is consumed downstream as a clean
spectrum estimate. `predict_xstart: false` selects EPSILON prediction, which is
only self-consistent if every consumer knows to convert; more importantly the
paper's Eq. (diff-loss) is `||F_theta(x_sigma) - x_0||^2`, i.e. the network
regresses x_0 directly, and the reference trainer
(MacroMolecularDynamics src/trainer_far_ray_tune.py) sets
`predict_xstart = True`.

This test pins the configs and `BaseConfig` so the setting cannot silently drift
back to `false` again.
"""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

# Configs that drive a real training or smoke run.
TRAINING_CONFIGS = [
    "trainer_BioDynaSpec.yaml",
    "trainer_BioDynaSpec_wo_irfc.yaml",
    "trainer_far_example.yaml",
    "trainer_smoke.yaml",
    "trainer_smoke_realdata.yaml",
]


def _load(name):
    with open(CONFIG_DIR / name) as fh:
        return yaml.safe_load(fh)


@pytest.mark.parametrize("config_name", TRAINING_CONFIGS)
def test_yaml_enables_predict_xstart(config_name):
    cfg = _load(config_name)
    assert cfg.get("predict_xstart") is True, (
        f"{config_name} sets predict_xstart={cfg.get('predict_xstart')!r}; "
        "it must be true so the network regresses x_0 (paper Eq. diff-loss)."
    )


@pytest.mark.parametrize("config_name", TRAINING_CONFIGS)
def test_primary_smoke_config_stays_bounded(config_name):
    """Guard the smoke configs against accidentally going back to a full run."""
    cfg = _load(config_name)
    if "smoke" not in config_name:
        pytest.skip("only smoke configs are expected to bound the run")
    limit = cfg.get("limit_train_batches", 1.0)
    assert limit <= 10, (
        f"{config_name} has limit_train_batches={limit}; smoke configs must stay "
        "bounded so they can finish in a few minutes."
    )


def test_base_config_default_matches_paper():
    """BaseConfig (src/trainer_far.py) is the fallback when a yaml omits the key."""
    import ast

    src = (REPO_ROOT / "src" / "trainer_far.py").read_text()
    tree = ast.parse(src)

    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "BaseConfig":
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                    target = stmt.targets[0]
                    if isinstance(target, ast.Name):
                        try:
                            found[target.id] = ast.literal_eval(stmt.value)
                        except ValueError:
                            pass

    assert found.get("predict_xstart") is True, (
        "BaseConfig.predict_xstart must be True to match the paper and the "
        f"reference trainer; found {found.get('predict_xstart')!r}."
    )


def test_main_model_fallback_is_x0():
    """main_model's getattr fallback must also default to START_X."""
    src = (REPO_ROOT / "src" / "main_model.py").read_text()
    assert "getattr(config, 'predict_xstart', True)" in src, (
        "main_model.py must fall back to predict_xstart=True when the config "
        "omits the key."
    )
