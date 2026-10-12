# Evaluation

Model-independent metrics for BioDynaSpec. Nothing here imports the training
loop, so it can be run against any set of generated ensembles.

Two groups:

- `metrics/` — ensemble and reconstruction metrics, driven by `.pt`/mdtraj
  trajectory pairs.
- `analysis_kevin/` — the protocol used for the paper's Table/Figure numbers
  (frame-wise Cα RMSD, PCA-2D Pearson/Hausdorff). These were originally written
  against a fixed directory layout; they now take their paths as CLI arguments
  with the original locations as defaults.

Units: `metrics/trajectory.py` and `metrics/coordinates.py` are **Ångström**-native.
`metrics/analyze_ensembles.py` works in **nanometres** internally and multiplies
by 10 on the way out. `metrics/reconstruction_ensembles.py` converts Å → nm for you.

## metrics/

### `trajectory.py` — pure NumPy alignment primitives

No mdtraj needed; takes `(T, N_atom, 3)` arrays in Å.

| Function | Returns |
|---|---|
| `framewise_kabsch_rmsd_angstrom(prediction, reference)` | `(T,)` per-frame optimal-rigid-fit RMSD |
| `global_alignment_rmsd_angstrom(prediction, reference)` | one transform from frame 0 applied to the whole trajectory |
| `rmsf(coords)` | `(N_atom,)` per-atom fluctuation |
| `target_reference_rmsf_metrics(prediction, reference)` | RMSF MAE/RMSE/bias/Pearson vs reference |
| `summarize_rmsd(values)` | `{mean,median,p95,max}_angstrom`, `num_frames` |

### `spectrum.py` — Fourier-domain MSE

```python
spectrum_metrics(vae, recon_tokens, target_real, target_imag)
```

Returns `normalized_token_spectrum_mse` and `physical_positive_spectrum_mse`.
`vae` must expose `prepare_spectrum_input(...)` and `recover_spectrum_from_tokens(...)`.

### `coordinates.py` — coordinate-space MSE + RMSD

`coordinate_metrics(processor, ...)` returns processed/physical coordinate MSE
plus framewise Kabsch RMSD.

> **Needs porting.** This module imports `restore_physical_coords` and
> `safe_irfft_window` from `src.inference_spectrum_vae`, which does not exist in
> this repo. The import is done lazily inside the function so the package still
> imports, and nothing downstream currently calls `coordinate_metrics`.
>
> **Intended adaptation.** In this repo the pipeline is: `ProteinDynamicsFAR`
> infers the *full* spectrum, `irfft` decodes it into per-window trajectories,
> the windows are concatenated into the complete trajectory, and only then is
> `coordinate_metrics` computed on it. So this module does not need a
> drop-in `src.inference_spectrum_vae`; it should be re-pointed at the
> trajectory-assembly helpers already in `src/data/fourier/utils.py`
> (`TrajectoryFFTProcessor.irfft_full`, `invert_preprocessing`,
> `reconstruct_and_unpack`,
> `reconstruct_package_from_positive_half_real_imag`) and at the window
> concatenation done by `src/inferencer_far.py`
> (`_spectrum_to_full_fft_coords` + `torch.cat(window_coords, dim=-1)`).
> Its inputs are then the assembled `(T, N_res, 3)` trajectory in Angstroms,
> which matches what `metrics/trajectory.py` already consumes.

### `analyze_ensembles.py` — the ensemble metric core

`analyze_ensemble_trajectories(prediction, reference, *, sample_frames=256)` is
the reusable entry point; both arguments are `mdtraj.Trajectory` in nm with
matching atom order. It computes, after selecting heavy atoms and aligning to
the reference frame 0:

PCA W2 (`Root Mean W2`), PC1 cosine, variance ratio, pairwise RMSD, contact
probability, atomic EMD, RMSF, SASA exposure and sidechain mutual information.

Supporting functions: `get_pca`, `get_rmsds`, `get_wasserstein`,
`get_pca2_wasserstein`, `get_emd`, `get_atomic_emd`, `get_contact_probability`,
`get_sasa_metrics`, `condense_sidechain_sasas`, `sasa_mi`, `summarize_ensemble_errors`.

Legacy ATLAS CLI (writes `out.pkl` into `--pdbdir`):

```bash
python -m evaluation.metrics.analyze_ensembles \
  --atlas_dir /path/to/atlas --pdbdir /path/to/generated --num_workers 8
```

### `reconstruction_ensembles.py` — compact paper-facing table

```bash
python evaluation/metrics/reconstruction_ensembles.py \
  --control arms/control --protenix arms/protenix \
  --feature-cache /path/to/feature_cache --output results.json
```

Each arm directory holds `*_trajectory.pt` files (torch dicts with
`pred_coords` and `target_coords`, in Å); both arms must contain the same names.
`reconstruction_ensemble_metrics(...)` returns the compact dict (PCA W2, PC1
cosine, variance, pairwise RMSD, contact MAE, atomic EMD, RMSF, SASA
exposure/MI) and `analyze_ensembles_compat_metrics(...)` returns the full one.

> `--feature-cache` is read via `src.data.basic.data_pipeline.DataPipeline`, whose
> import is lazy so the module still loads if that path is unavailable.

## analysis_kevin/

All six scripts now take `--help`. Every flag defaults to the original hardcoded
location, so an invocation with no arguments reproduces the old behaviour on the
machine where those paths exist.

```bash
# 1. Fit reference PCA models (Cartesian, then pairwise-distance)
python evaluation/analysis_kevin/fit_pca.py \
  --root-dir /path/to/alphaflow/atlas \
  --output-dir /path/to/pca_models \
  --split-csv /path/to/atlas_test.csv

python evaluation/analysis_kevin/fit_pca_ca_pwd.py \
  --root-dir /path/to/alphaflow/atlas \
  --output-dir /path/to/pca_models_pwd --split-csv /path/to/atlas_test.csv

# 2. PCA-2D Pearson + bidirectional Hausdorff vs each generated ensemble
python evaluation/analysis_kevin/analysis_pca.py \
  --pca-models-dir /path/to/pca_models \
  --real-data-dir  /path/to/pca_models \
  --net mdgen=R1:/path/atlas_R1,R2:/path/atlas_R2,R3:/path/atlas_R3 \
  --net specpd=R1:/path/reconstructed_pdbs_atlas \
  --output-dir pca_analysis_results

python evaluation/analysis_kevin/analysis_pca_ca_pwd.py \
  --pca-models-dir /path/to/pca_models_pwd \
  --real-data-dir  /path/to/pca_models_pwd \
  --net mdgen=R1:/path/atlas_R1 \
  --output-dir pca_analysis_results_ca_pwd

# 3. KDE plots of the same projections
python evaluation/analysis_kevin/draw_pca_new.py \
  --pca-dir coord=/path/to/pca_models \
  --net mdgen=R1:/path/atlas_R1 \
  --output-dir pca_visualizations

# 4. Frame-wise Ca RMSD: R50 / R100 / R150 / R200 / R250
python evaluation/analysis_kevin/framewise_rmsd.py \
  --root-reference /path/to/MD_raw_data/ATLAS \
  --root-net1 /path/to/mdgen_results/atlas_R3 \
  --root-net2 /path/to/2025-07-21-p2dflow-R3 \
  --split-csv /path/to/atlas_test.csv \
  --output rmsd_results.txt
```

### Assumptions baked into `framewise_rmsd.py`

These come from the paper protocol and are **not** parameterised, because the
metrics `R_s` are defined over exactly these frame indices:

- generated ensembles have exactly **250** frames; the reference is subsampled
  to 250 at a **400 ps** interval (`net_time = arange(0, 250*400, 400)`), from
  reference trajectories spaced at 100 ps;
- `R_s = mean(rmsd[:s])` for `s ∈ {50, 100, 150, 200, 250}`;
- the reference is `{name}/{name}.pdb` + `{name}_R3.xtc`; generated ensembles are
  a single `{name}.pdb`.

`analysis_pca*.py` look for generated files named
`protein_{name}_R1_reconstructed.pdb` (specpd) or `{name}_time_3.pdb`
(alphafolding) inside each per-replica directory.

## Dependencies

`mdtraj` (added to `requirements.txt`), plus `numpy`, `scipy`, `scikit-learn`,
`pandas`, `joblib`, `matplotlib`, `seaborn`, `tqdm` — all already present.
