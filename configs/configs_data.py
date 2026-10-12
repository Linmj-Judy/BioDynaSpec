import os
import json
from .config_utils import EnvConfig

AF3_DATA_ROOT_DIR = os.environ.get("AF3_DATA_ROOT_DIR", "/data/L202500336/database/af3_database/release_data")
MD_DATA_ROOT_DIR = os.environ.get("MD_DATA_ROOT_DIR", "/data/L202500336/database/Dynamics/data")
FEATURE_CACHE_ROOT = os.environ.get("FEATURE_CACHE_ROOT", MD_DATA_ROOT_DIR)
# Optional cache namespace selected by a launcher.  Keep the historical
# ``feature_cache/<dataset>`` layout by default, but allow multi-window
# producers to isolate each physical representation without changing the
# public FEATURE_CACHE_ROOT contract.
FEATURE_CACHE_SUBDIR = os.environ.get("FEATURE_CACHE_SUBDIR", "feature_cache")
FOURIER_DATA_ROOT_DIR = os.environ.get("FOURIER_DATA_ROOT_DIR", MD_DATA_ROOT_DIR)
PAIRFORMER_REPR_ROOT = os.environ.get(
    "PAIRFORMER_REPR_ROOT", os.path.join(MD_DATA_ROOT_DIR, "pairformer_cache")
)

# Use CCD cache created by scripts/gen_ccd_cache.py priority. (without date in filename)
# See: docs/prepare_data.md
CCD_COMPONENTS_FILE_PATH = os.path.join(AF3_DATA_ROOT_DIR, "components.cif")
CCD_COMPONENTS_RDKIT_MOL_FILE_PATH = os.path.join(
    AF3_DATA_ROOT_DIR, "components.cif.rdkit_mol.pkl"
)
PDB_CLUSTER_FILE_PATH = os.path.join(AF3_DATA_ROOT_DIR, "clusters-by-entity-40.txt")

if (not os.path.exists(CCD_COMPONENTS_FILE_PATH)) or (
    not os.path.exists(CCD_COMPONENTS_RDKIT_MOL_FILE_PATH)
):
    CCD_COMPONENTS_FILE_PATH = os.path.join(AF3_DATA_ROOT_DIR, "components.v20240608.cif")
    CCD_COMPONENTS_RDKIT_MOL_FILE_PATH = os.path.join(
        AF3_DATA_ROOT_DIR, "components.v20240608.cif.rdkit_mol.pkl"
    )

if os.uname().sysname == "Darwin":
    CCD_COMPONENTS_FILE_PATH = "/Users/mujielin/Documents/GitHub/MacroMolecularDynamics/test/data/ccd/components.cif"
    CCD_COMPONENTS_RDKIT_MOL_FILE_PATH = "/Users/mujielin/Documents/GitHub/MacroMolecularDynamics/test/data/ccd/components.cif.rdkit_mol.pkl"

env_config = EnvConfig({
    "use_esm": ("USE_ESM",False),
    "use_msa": ("USE_MSA",True),
    "fft_type": ("FFT_TYPE", "fft_window"),  # fft_full | fft_window | stft_window
    "window_size": ("WINDOW_SIZE", 500),
})


# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=C0114,C0301
from copy import deepcopy

from src.config.extend_types import GlobalConfigValue, ListValue

default_test_configs = {
    "sampler_configs": {
        "sampler_type": "uniform",
    },
    "cropping_configs": {
        "method_weights": [
            0.0,  # ContiguousCropping
            0.0,  # SpatialCropping
            1.0,  # SpatialInterfaceCropping
        ],
        "crop_size": -1,
    },
    "lig_atom_rename": GlobalConfigValue("test_lig_atom_rename"),
    "shuffle_mols": GlobalConfigValue("test_shuffle_mols"),
    "shuffle_sym_ids": GlobalConfigValue("test_shuffle_sym_ids"),
    "constraint": {
        "enable": False,
        "fix_seed": False,  # True means use use the same contact in each evaluation.
    },
}

default_weighted_pdb_configs = {
    "sampler_configs": {
        "sampler_type": "weighted",
        "beta_dict": {
            "chain": 0.5,
            "interface": 1,
        },
        "alpha_dict": {
            "prot": 3,
            "nuc": 3,
            "ligand": 1,
        },
        "force_recompute_weight": True,
    },
    "cropping_configs": {
        "method_weights": ListValue([0.2, 0.4, 0.4]),
        "crop_size": GlobalConfigValue("train_crop_size"),
    },
    "sample_weight": 0.5,
    "limits": -1,
    "lig_atom_rename": GlobalConfigValue("train_lig_atom_rename"),
    "shuffle_mols": GlobalConfigValue("train_shuffle_mols"),
    "shuffle_sym_ids": GlobalConfigValue("train_shuffle_sym_ids"),
    # If enabled, the training settings for different constraint types,
    # providing the model a certain proportion of constraints
    # that meet specific conditions.
    "constraint": {
        "enable": False,
        "fix_seed": False,
        "pocket": {
            "prob": 0.0,
            "size": 1 / 3,
            "spec_binder_chain": False,
            "max_distance_range": {"PP": ListValue([6, 20]), "LP": ListValue([6, 20])},
            "group": "complex",
            "distance_type": "center_atom",
        },
        "contact": {
            "prob": 0.0,
            "size": 1 / 3,
            "max_distance_range": {
                "PP": ListValue([6, 30]),
                "PL": ListValue([4, 10]),
            },
            "group": "complex",
            "distance_type": "center_atom",
        },
        "substructure": {
            "prob": 0.0,
            "size": 0.8,
            "mol_type_pairs": {
                "PP": 15,
                "PL": 10,
                "LP": 10,
            },
            "feature_type": "one_hot",
            "ratios": {
                "full": [
                    0.0,
                    0.5,
                    1.0,
                ],  # ratio options of full chain substructure constraint
                "partial": 0.3,  # ratio of partial chain substructure constraint
            },
            "coord_noise_scale": 0.05,
            "spec_asym_id": False,
        },
        "contact_atom": {
            "prob": 0.0,
            "size": 1 / 3,
            "max_distance_range": {
                "PP": ListValue([2, 12]),
                "PL": ListValue([2, 8]),
            },
            "min_distance": -1,
            "group": "complex",
            "distance_type": "atom",
            "feature_type": "continuous",
        },
    },
}

fft_type = env_config.config_dict["fft_type"]
window_size = env_config.config_dict["window_size"]
fourier_dir_name = os.environ.get(
    "FOURIER_DIR_NAME", f"fourier_{fft_type}_window{window_size}"
)


def fourier_dataset_dir(dataset_name: str) -> str:
    return os.path.join(FOURIER_DATA_ROOT_DIR, fourier_dir_name, dataset_name)


def msa_dataset_dir(dataset_name: str) -> str:
    """Return the canonical per-dataset MSA root used by Dynamics data."""
    slug = {"DPF": "dpf"}.get(str(dataset_name), str(dataset_name).lower())
    return os.path.join(MD_DATA_ROOT_DIR, f"mmcif_msa_v2_{slug}")


def feature_cache_dir(dataset_name: str) -> str:
    # Keep the historical path as the default public cache contract.  A
    # launcher may set FEATURE_CACHE_SUBDIR (for example
    # ``feature_cache/fourier_fft_window_window500``) to isolate a physical
    # Fourier variant.  Use the canonical `fourier_{fft_type}_window{N}` form:
    # a short alias such as ``window500`` diverges from the published root name.
    return os.path.join(FEATURE_CACHE_ROOT, FEATURE_CACHE_SUBDIR, dataset_name)


def feature_cache_variant_dir(dataset_name: str) -> str:
    """Return an optional physically namespaced cache root for migrations."""
    return os.path.join(
        FEATURE_CACHE_ROOT, FEATURE_CACHE_SUBDIR, fourier_dir_name, dataset_name
    )


def external_dynamics_dataset_config(dataset_name: str, pdb_list_env: str) -> dict:
    """Build one namespaced external-Dynamics training dataset configuration."""

    return {
        "base_info": {
            "bioassembly_dict_dir": os.path.join(
                MD_DATA_ROOT_DIR, "bioassembly", dataset_name
            ),
            "fourier_output_dir": fourier_dataset_dir(dataset_name),
            "traj_indices_fpath": os.path.join(
                MD_DATA_ROOT_DIR, "indices", f"{dataset_name}_traj_indices.csv"
            ),
            "pdb_indices_fpath": os.path.join(
                MD_DATA_ROOT_DIR, "indices", f"{dataset_name}_pdb_indices.csv"
            ),
            "pdb_list": os.getenv(pdb_list_env, ""),
            "random_sample_if_failed": True,
            "max_n_token": -1,
            "use_reference_chains_only": False,
            "exclusion": {
                "mol_1_type": ListValue(["ions"]),
                "mol_2_type": ListValue(["ions"]),
            },
        },
        "sequence_fpath": os.path.join(
            MD_DATA_ROOT_DIR, "sequences", "pdb_labels_seq.csv"
        ),
        "msa_dir": msa_dataset_dir(dataset_name),
        "esm_embedding_dir": os.path.join(
            MD_DATA_ROOT_DIR, "esm_cache", "esm2_3b_embed"
        ),
        "esm_fusion_rate": {
            "msa": 0.3,
            "esm": 0.3,
            "both": 0.3,
            "neither": 0.1,
        },
        "pairformer_repr_dir": os.path.join(
            PAIRFORMER_REPR_ROOT, dataset_name
        ),
        # BiHiTo is a legacy optional branch.  The current production feature
        # contract uses the regular feature cache and leaves this branch off.
        # Keep a string for strict config merging; use_bihito=False means this
        # legacy path is never probed or loaded.
        "bihito_repr_dir": os.path.join(
            MD_DATA_ROOT_DIR, "bihito_repr_cache", dataset_name
        ),
        "use_bihito": False,
        "use_cache": str(os.getenv("PAIRFORMER_PRECOMPUTED", "False")).lower()
        == "true",
        "cache_dir": feature_cache_dir(dataset_name),
        "return_atom_token_array": True,
        **deepcopy(default_weighted_pdb_configs),
    }

# use_sliding_window and unroll_frames are derived from fft_type:
# "fft_window" / "stft_window" → both True; "fft_full" → both False
use_sliding_window = "window" in fft_type
unroll_frames = "window" in fft_type


def _dataset_name_from_env(variable: str, default: str) -> str:
    """Read one Dynamics dataset name without silently accepting a typo."""
    value = os.getenv(variable, default).strip()
    aliases = {
        "atlas": "atlas",
        "dpf": "DPF",
        "chameleon": "chameleon",
        "chamelon": "chameleon",
        "pareto": "pareto",
        "patero": "pareto",
    }
    normalized = aliases.get(value.lower())
    if normalized is None:
        raise ValueError(
            f"{variable} must be atlas, DPF, or chameleon (pareto is also "
            f"supported for explicit dynamics manifests); got {value!r}"
        )
    return normalized


def _train_variants_from_env() -> list[dict]:
    """Parse an optional JSON manifest for mixed dataset/window training.

    Keeping the default empty preserves every existing single-window run.  A
    manifest entry may point at any registered dataset and override its Fourier
    root/cache/window/dt metadata without duplicating the structural config.
    """
    raw = os.getenv("TRAIN_VARIANTS_JSON", "").strip()
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("TRAIN_VARIANTS_JSON must be valid JSON") from exc
    if not isinstance(value, list):
        raise ValueError("TRAIN_VARIANTS_JSON must be a JSON list")
    return value


train_dataset_name = _dataset_name_from_env("TRAIN_DATASET", "chameleon")
test_dataset_name = _dataset_name_from_env("TEST_DATASET", train_dataset_name)
train_variants_from_env = _train_variants_from_env()
# ConfigManager infers list element types from the first item and therefore
# cannot represent a bare ``[]``.  Keep the empty/default case nullable with a
# dict element type; all consumers already treat ``None`` as an empty manifest.
train_variants_config = (
    ListValue(train_variants_from_env)
    if train_variants_from_env
    else ListValue(None, dtype=dict)
)
manifest_fft_types = {
    str(v.get("fft_type")) for v in train_variants_from_env
    if isinstance(v, dict) and v.get("fft_type")
}
manifest_has_window = any(
    isinstance(v, dict)
    and (v.get("window_size") is not None or v.get("window") is not None)
    for v in train_variants_from_env
)

data_configs = {
    "num_dl_workers": int(os.getenv("NUM_DL_WORKERS", "0")),
    # Keep worker processes alive across epochs so mixed-variant feature
    # caches are not repeatedly reconstructed.  The queue is deliberately
    # small because each prefetched sample can contain a full Fourier pack.
    "persistent_workers": str(os.getenv("PERSISTENT_DL_WORKERS", "True")).lower()
    in {"1", "true", "yes"},
    "prefetch_factor": int(os.getenv("DL_PREFETCH_FACTOR", "2")),
    "epoch_size": int(os.getenv("EPOCH_SIZE", "5500")),
    # Frame-level micro-batches are shape-bucketed by the unroll loader, so
    # equal atom/frequency geometries can be stacked without padding.  Keep
    # the legacy default until a run explicitly opts into the audited path.
    "frame_batch_size": int(os.getenv("FRAME_BATCH_SIZE", "1")),
    # Adaptive frame micro-batching keeps small shape buckets productive while
    # forcing large atom/frequency windows to remain at batch one. A zero
    # budget preserves the fixed legacy behavior.
    "frame_batch_size_max": int(
        os.getenv("FRAME_BATCH_SIZE_MAX", os.getenv("FRAME_BATCH_SIZE", "1"))
    ),
    "frame_batch_cost_budget": float(os.getenv("FRAME_BATCH_COST_BUDGET", "0")),
    "frame_batch_pad_incomplete": str(
        os.getenv("FRAME_BATCH_PAD_INCOMPLETE", "False")
    ).lower() in {"1", "true", "yes"},
    "frame_bucket_lookahead_batches": int(
        os.getenv("FRAME_BUCKET_LOOKAHEAD_BATCHES", "4")
    ),
    # Used only for progress/scheduler length estimation; actual frame counts
    # still come from each Fourier package at iteration time.
    "estimated_frames_per_sample": int(
        os.getenv("FRAME_ESTIMATED_FRAMES_PER_PACKAGE", "99")
    ),
    "train_ref_pos_augment": str(
        os.getenv("TRAIN_REF_POS_AUGMENT", "True")
    ).lower() in {"1", "true", "yes"},
    "test_ref_pos_augment": str(
        os.getenv("TEST_REF_POS_AUGMENT", "True")
    ).lower() in {"1", "true", "yes"},
    "train_sets": ListValue([train_dataset_name]),
    # Optional explicit representation manifest. Example:
    # [{"name":"atlas_w500","dataset":"atlas","window_size":500,
    #   "dt_ps":10,"fourier_output_dir":".../atlas","sample_weight":0.5}]
    "train_variants": train_variants_config,
    # Mixed runs default to exposure-balanced sampling: window-count
    # compensation for a shared dt, physical-time compensation when cadence
    # differs. Set TRAIN_VARIANT_SAMPLING=package for package-level control;
    # physical_window remains an explicit historical duration-weighted control.
    "train_variant_sampling": os.getenv("TRAIN_VARIANT_SAMPLING", "auto"),
    # Stratified quotas make the realized mixed prior stable for short epochs.
    # ``multinomial`` remains available as a matched stochastic control.
    "train_variant_sampling_strategy": os.getenv(
        "TRAIN_VARIANT_SAMPLING_STRATEGY", "stratified_epoch"
    ),
    "group_split_by_trajectory": False,
    "train_sampler": {
        "train_sample_weights": ListValue([1.0]),
        "sampler_type": "weighted",
    },
    "test_sets": ListValue([test_dataset_name]),
    # use_sliding_window and unroll_frames — derived from fft_type above
    "use_sliding_window": bool(use_sliding_window or manifest_has_window or any("window" in t for t in manifest_fft_types)),
    "unroll_frames": bool(unroll_frames or manifest_has_window or any("window" in t for t in manifest_fft_types)),
    "fft_type": fft_type,
    "window_size": window_size,
    "atlas": {
        "base_info": {
            "bioassembly_dict_dir": os.path.join(MD_DATA_ROOT_DIR, "bioassembly", "atlas"),
            "fourier_output_dir": fourier_dataset_dir("atlas"),
            "traj_indices_fpath": os.path.join(
                MD_DATA_ROOT_DIR,
                "indices/atlas_traj_indices.csv",
            ),
            "pdb_indices_fpath": os.path.join(
                MD_DATA_ROOT_DIR,
                "indices/atlas_pdb_indices.csv",
            ),
            "pdb_list": os.getenv("ATLAS_PDB_LIST", ""),
            "random_sample_if_failed": True,
            "max_n_token": -1,  # can be used for removing data with too many tokens.
            "use_reference_chains_only": False,
            "exclusion": {  # do not sample the data based on ions.
                "mol_1_type": ListValue(["ions"]),
                "mol_2_type": ListValue(["ions"]),
            },
        },
        "sequence_fpath": os.path.join(MD_DATA_ROOT_DIR, "sequences", "pdb_labels_seq.csv"),
        "msa_dir": msa_dataset_dir("atlas"),
        "esm_embedding_dir": os.path.join(MD_DATA_ROOT_DIR, "esm_cache", "esm2_3b_embed"),
        "esm_fusion_rate": {
            "msa": 0.3,
            "esm": 0.3,
            "both": 0.3,
            "neither": 0.1,
        },
        "pairformer_repr_dir": os.path.join(PAIRFORMER_REPR_ROOT, "atlas"),
        # BiHiTo is not part of the current Atlas/SpectrumVAE contract.
        "bihito_repr_dir": os.path.join(MD_DATA_ROOT_DIR, "bihito_repr_cache", "atlas"),
        "use_bihito": False,
        "use_cache": True if str(os.getenv("PAIRFORMER_PRECOMPUTED", "False")).lower() == "true" else False,
        "cache_dir": feature_cache_dir("atlas"),
        "return_atom_token_array": True,
        **deepcopy(default_weighted_pdb_configs),
    },
    "DPF": {
        "base_info": {
            "bioassembly_dict_dir": os.path.join(MD_DATA_ROOT_DIR, "bioassembly", "DPF"),
            "fourier_output_dir": fourier_dataset_dir("DPF"),
            "traj_indices_fpath": os.path.join(
                MD_DATA_ROOT_DIR,
                "indices/DPF_traj_indices.csv",
            ),
            "pdb_indices_fpath": os.path.join(
                MD_DATA_ROOT_DIR,
                "indices/DPF_pdb_indices.csv",
            ),
            # One exact trajectory ID or parent PDB ID per line.
            "pdb_list": os.getenv("DPF_PDB_LIST", ""),
            "random_sample_if_failed": True,
            "max_n_token": -1,  # can be used for removing data with too many tokens.
            "use_reference_chains_only": False,
            "exclusion": {  # do not sample the data based on ions.
                "mol_1_type": ListValue(["ions"]),
                "mol_2_type": ListValue(["ions"]),
            },
        },
        "sequence_fpath": os.path.join(MD_DATA_ROOT_DIR, "sequences", "pdb_labels_seq.csv"),
        "msa_dir": msa_dataset_dir("DPF"),
        "esm_embedding_dir": os.path.join(MD_DATA_ROOT_DIR, "esm_cache", "esm2_3b_embed"),
        "esm_fusion_rate": {
            "msa": 0.3,
            "esm": 0.3,
            "both": 0.3,
            "neither": 0.1,
        },
        "pairformer_repr_dir": os.path.join(PAIRFORMER_REPR_ROOT, "DPF"),
        "bihito_repr_dir": os.path.join(MD_DATA_ROOT_DIR, "bihito_repr_cache", "DPF"),
        "use_bihito": False,
        "use_cache": True if str(os.getenv("PAIRFORMER_PRECOMPUTED", "False")).lower() == "true" else False,
        "cache_dir": feature_cache_dir("DPF"),
        "return_atom_token_array": True,
        **deepcopy(default_weighted_pdb_configs),
    },
    "chameleon": {
        "base_info": {
            "bioassembly_dict_dir": os.path.join(MD_DATA_ROOT_DIR, "bioassembly", "chameleon"),
            "fourier_output_dir": fourier_dataset_dir("chameleon"),
            "traj_indices_fpath": os.path.join(
                MD_DATA_ROOT_DIR,
                "indices/chameleon_traj_indices.csv",
            ),
            "pdb_indices_fpath": os.path.join(
                MD_DATA_ROOT_DIR,
                "indices/chameleon_pdb_indices.csv",
            ),
            # One exact trajectory ID or parent PDB ID per line.
            "pdb_list": os.getenv("CHAMELEON_PDB_LIST", ""),
            "random_sample_if_failed": True,
            "max_n_token": -1,  # can be used for removing data with too many tokens.
            "use_reference_chains_only": False,
            "exclusion": {  # do not sample the data based on ions.
                "mol_1_type": ListValue(["ions"]),
                "mol_2_type": ListValue(["ions"]),
            },
        },
        "sequence_fpath": os.path.join(MD_DATA_ROOT_DIR, "sequences", "pdb_labels_seq.csv"),
        "msa_dir": msa_dataset_dir("chameleon"),
        "esm_embedding_dir": os.path.join(MD_DATA_ROOT_DIR, "esm_cache", "esm2_3b_embed"),
        "esm_fusion_rate": {
            "msa": 0.3,
            "esm": 0.3,
            "both": 0.3,
            "neither": 0.1,
        },
        "pairformer_repr_dir": os.path.join(PAIRFORMER_REPR_ROOT, "chameleon"),
        "bihito_repr_dir": os.path.join(MD_DATA_ROOT_DIR, "bihito_repr_cache", "chameleon"),
        "use_bihito": False,
        "use_cache": True if str(os.getenv("PAIRFORMER_PRECOMPUTED", "False")).lower() == "true" else False,
        "cache_dir": feature_cache_dir("chameleon"),
        "return_atom_token_array": True,
        **deepcopy(default_weighted_pdb_configs),
    },
    "pareto": external_dynamics_dataset_config("pareto", "PARETO_PDB_LIST"),
    "mdcath": external_dynamics_dataset_config("mdcath", "MDCATH_PDB_LIST"),
    "dynamicpdb": external_dynamics_dataset_config(
        "dynamicpdb", "DYNAMICPDB_PDB_LIST"
    ),
    "mdrepo": external_dynamics_dataset_config("mdrepo", "MDREPO_PDB_LIST"),

    # "weightedPDB_before2109_wopb_nometalc_0925": {
    #     "base_info": {
    #         "mmcif_dir": os.path.join(DATA_ROOT_DIR, "mmcif"),
    #         "bioassembly_dict_dir": os.path.join(DATA_ROOT_DIR, "mmcif_bioassembly"),
    #         "indices_fpath": os.path.join(
    #             DATA_ROOT_DIR,
    #             "indices/weightedPDB_indices_before_2021-09-30_wo_posebusters_resolution_below_9.csv.gz",
    #         ),
    #         "pdb_list": "",
    #         "random_sample_if_failed": True,
    #         "max_n_token": -1,  # can be used for removing data with too many tokens.
    #         "use_reference_chains_only": False,
    #         "exclusion": {  # do not sample the data based on ions.
    #             "mol_1_type": ListValue(["ions"]),
    #             "mol_2_type": ListValue(["ions"]),
    #         },
    #     },
    #     "esm_fusion_rate": {
    #         "msa": 0.3,
    #         "esm": 0.3,
    #         "both": 0.3,
    #         "neither": 0.1,
    #     },
    #     **deepcopy(default_weighted_pdb_configs),
    # },
    # "recentPDB_1536_sample384_0925": {
    #     "base_info": {
    #         "mmcif_dir": os.path.join(DATA_ROOT_DIR, "mmcif"),
    #         "bioassembly_dict_dir": os.path.join(
    #             DATA_ROOT_DIR, "recentPDB_bioassembly"
    #         ),
    #         "indices_fpath": os.path.join(
    #             DATA_ROOT_DIR, "indices/recentPDB_low_homology_maxtoken1536.csv"
    #         ),
    #         "pdb_list": os.path.join(
    #             DATA_ROOT_DIR,
    #             "indices/recentPDB_low_homology_maxtoken1024_sample384_pdb_id.txt",
    #         ),
    #         "max_n_token": GlobalConfigValue("test_max_n_token"),  # filter data
    #         "sort_by_n_token": False,
    #         "group_by_pdb_id": True,
    #         "find_eval_chain_interface": True,
    #     },
    #     **deepcopy(default_test_configs),
    # },
    # "posebusters_0925": {
    #     "base_info": {
    #         "mmcif_dir": os.path.join(DATA_ROOT_DIR, "posebusters_mmcif"),
    #         "bioassembly_dict_dir": os.path.join(
    #             DATA_ROOT_DIR, "posebusters_bioassembly"
    #         ),
    #         "indices_fpath": os.path.join(
    #             DATA_ROOT_DIR, "indices/posebusters_indices_mainchain_interface.csv"
    #         ),
    #         "pdb_list": "",
    #         "find_pocket": True,
    #         "find_all_pockets": False,
    #         "max_n_token": GlobalConfigValue("test_max_n_token"),  # filter data
    #     },
    #     **deepcopy(default_test_configs),
    # },
    "msa": {
        "enable": env_config.config_dict["use_msa"],
        "enable_rna_msa": False,
        "prot": {
            "pairing_db": "uniref100",
            "non_pairing_db": "mmseqs_other",
            "pdb_mmseqs_dir": os.getenv(
                "PROTENIX_MSA_DIR", os.path.join(AF3_DATA_ROOT_DIR, "mmcif_msa")
            ),
            "seq_to_pdb_idx_path": os.getenv(
                "PROTENIX_MSA_SEQ_INDEX_PATH",
                os.path.join(AF3_DATA_ROOT_DIR, "seq_to_pdb_index.json"),
            ),
            "indexing_method": "sequence",
        },
        "rna": {
            "seq_to_pdb_idx_path": "",
            "rna_msa_dir": "",
            "indexing_method": "sequence",
        },
        "strategy": "random",
        "merge_method": "dense_max",
        "min_size": {
            "train": 1,
            "test": 1,
        },
        "max_size": {
            "train": 16384,
            "test": 16384,
        },
        "sample_cutoff": {
            "train": 16384,
            "test": 16384,
        },
    },
    "esm": {
        "enable": env_config.config_dict["use_esm"],
        "model_name": "esm2-3b",
        "embedding_dim": 2560,
        "embedding_dir": "esm_cache/esm2_3b",
        "sequence_fpath": "esm_cache/pdb_labels_seqs.csv",
    },
    "template": {
        "enable": False,
    },
    "ccd_components_file": CCD_COMPONENTS_FILE_PATH,
    "ccd_components_rdkit_mol_file": CCD_COMPONENTS_RDKIT_MOL_FILE_PATH,
    "pdb_cluster_file": PDB_CLUSTER_FILE_PATH,
    # Chameleon is the active default dataset. An exact override is available
    # for mixed-data runs; never silently load an unrelated DPF processor.
    "fourier_processor_fpath": os.environ.get(
        "FOURIER_PROCESSOR_FPATH",
        os.path.join(fourier_dataset_dir("chameleon"), "processor.pth"),
    ),
}
