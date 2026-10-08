# Language Models Read the Word Before They Weigh the Stake

Code, stimuli, pre-registrations, and summary results for the paper
"Language Models Read the Word Before They Weigh the Stake: Valence Shifts from Lexical Polarity to Goal-Relative Appraisal".

## Layout

| Folder | Content |
|---|---|
| `data/` | Stimuli: valence stimuli (`goal_stimuli.jsonl`), non-affective controls (`control*_stimuli.jsonl` and their stake-free source sentences), goal-relevant key controls (`goalneutral*_stimuli.jsonl`), the naturalistic set (`natural_raw.jsonl`, `natural_stimuli.jsonl`), and the GoEmotions valence sentences used for the robustness directions |
| `src/` | Stimulus generation, activation extraction, behaviour, patching, and all analyses |
| `figures/` | Figures, tables, and LaTeX number macros of the paper, computed from `results/` |
| `results/` | Summary results (JSON) from which every number in the paper is generated |
| `preregistration/` | Pre-registration v1 and amendments v2–v11, the kill-switch record, and the list of post hoc analyses |

## Setup

```bash
pip install -r requirements.txt
export GOALVAL_ROOT=$(pwd)        # data/ and results/ are read from and written to this folder
```

All experiments ran on a single NVIDIA H100 NVL GPU (95 GB) with forward passes only, in about 6 GPU-hours in total.

## Reproducing the figures and tables (no GPU needed)

```bash
python figures/make_figures.py          # figures, tables, and number macros
python figures/band_sensitivity.py      # robustness: band edges
python figures/threshold_sensitivity.py # robustness: thresholds and model families
python figures/robustness_tables.py     # robustness tables, placebo figure, macros
python figures/make_fig_overview.py     # Figure 1
```

Outputs are written to `outputs/`.

## Reproducing the experiments

Models are given as `tag=HuggingFaceRepo`:

| Tag | Model |
|---|---|
| `qwen25_7b_it`, `qwen25_14b_it`, `qwen25_32b_it` | Qwen/Qwen2.5-7B/14B/32B-Instruct |
| `mistral_7b_it` | mistralai/Mistral-7B-Instruct-v0.3 |
| `falcon3_7b_it` | tiiuae/Falcon3-7B-Instruct |
| `granite31_8b_it` | ibm-granite/granite-3.1-8b-instruct |
| `olmo2_7b_it` | allenai/OLMo-2-1124-7B-Instruct |
| `phi4` | microsoft/phi-4 |
| `qwen25_7b_base`, `olmo2_7b_base` | Qwen/Qwen2.5-7B, allenai/OLMo-2-1124-7B |

1. Stimuli (already included in `data/`): `make_stimuli.py`, `make_control_stimuli.py`, `make_control2_stimuli.py`, `make_goalneutral_stimuli.py`, `make_goalneutral2_stimuli.py`; the naturalistic set with `gen_natural.py` (requires `OPENAI_API_KEY`) and `natural_prep.py` (NLI validation).
2. Extraction and the confirmatory analysis (H1–H5): `python src/orchestrate.py qwen25_7b_it=Qwen/Qwen2.5-7B-Instruct ...`, which downloads each model, runs `extract.py` and `analyze.py`, and removes the weights; then `summarize.py`, `h5.py`, `band_boot.py`, `ablate.py` (via `run_ablate.py`), and `summarize_ablate.py`.
3. Later registered analyses, one model load per sweep: `run_sweep2.py` (controls, prompt formats, patching, naturalistic set; H6–H10), `run_control2.py` (H11), `run_informat.py` (H12), `run_sweep3.py` (H13–H17), `run_sweep4.py` (H18–H19), `run_sweep5.py` (H22), and `run_sweep6.py` (robustness R2 and R6). `run_sweep2.py`, `run_control2.py`, and `run_informat.py` take `tag=repo`; `run_sweep3.py` to `run_sweep6.py` take `tag=repo=parts`, for example `python src/run_sweep4.py qwen25_7b_it=Qwen/Qwen2.5-7B-Instruct=taskgate,gn2`. The 32B model was run with `run_32b_light.py`, which stores only projections.
4. Analyses of the saved outputs: `analyze_sweep2.py` to `analyze_sweep5.py`, `analyze_informat.py`, `decompose.py` (H20), `leak_audit.py` (H21), `domain_robust.py`, `natural_relabel.py`, `prior_regression.py`, `clf_baseline.py`, `cross_source.py`, `sst_validity.py`, and `domain_band.py`.

Each analysis writes a summary JSON to `results/`; `figures/` then turns these files into the numbers reported in the paper.

## Pre-registration

`preregistration/` contains the registrations in the order in which they were made. Each file states what had been seen when it was written. Analyses that were not registered are listed in `posthoc_after_review.md` and marked as post hoc in the paper. The timestamped version-control history of these files will be released with the de-anonymised version of this repository.

## Data and licenses

GoEmotions (Apache-2.0) and SST-2 (loaded with `datasets` from `stanfordnlp/sst2`) are used as stake-free valence sources. The naturalistic sentences were generated with GPT-4.1-mini (version 2025-04-14) and contain fictional names only. Model weights are not redistributed.
