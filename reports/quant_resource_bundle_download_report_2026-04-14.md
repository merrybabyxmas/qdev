# Quant Resource Bundle Download Report

Date: 2026-04-14

## What Was Done

- Normalized the pasted quant resource list into a deduplicated inventory.
- Converted the inventory into an executable bash downloader:
  - `scripts/download_quant_resource_bundle.sh`
- Added a generator for the manifest and cleaned markdown:
  - `scripts/generate_quant_resource_bundle.py`
- Generated the cleaned inventory and manifest:
  - `docs/references/quant_resource_bundle_clean.md`
  - `docs/references/quant_resource_bundle_manifest.json`
- Prioritized resources by intent:
  - Priority 1: core data, modeling, backtest, and trading infrastructure
  - Priority 2: broader research references and supporting libraries
  - Priority 3: lower-priority leftovers

## Deduplication Result

The clean manifest currently records:

- Total unique resources: `159`
- Papers: `58`
- Repos: `51`
- Packages: `50`

Duplicate entries removed from the source bundle:

- `microsoft/qlib.git` duplicate repo
- `1708.07469` duplicate paper
- `2304.07619` duplicate paper

## Download Execution

Downloads were executed into:

- `artifacts/quant_resource_bundle/`
- Canonical paper store: `docs/references/papers/`

The executable script supports:

- `PRIORITY_MAX`
- `KIND_FILTER`
- `MODE`
- `FORCE`
- `PYTHON_BIN`

### Final Download State

After the last run and cleanup, the artifact tree contains:

- Papers downloaded: `57` (consolidated into `docs/references/papers/`)
- Repos downloaded: `44`
- Packages downloaded: `49`

### Explicitly Excluded by Request

- `tensorflow` was downloaded during the package batch and then removed from the artifact tree at user request.
- The working direction for implementation is torch-first.

### Directory Unification

- The paper directory is being unified so that `artifacts/quant_resource_bundle/papers` points at `docs/references/papers`.
- Existing paper PDFs from the bundle are being consolidated into the canonical docs directory rather than living in two separate physical locations.

## Remaining Missing Items

The manifest still has a small set of unresolved items:

- `awesome-portfolio-management` repo
- `DeepPortfolio` repo
- `shrimpy-python` repo
- `algo-trading-python` repo
- `Attention Networks for Financial Data` paper
- `Stochastic-Calculus` repo
- `quant-ecosystem` repo
- `awesome-quant-finance` repo
- `tensorflow` package artifact is intentionally absent

## Notable Download Outcomes

### Succeeded

- Priority 1 papers: bulk download completed
- Priority 1 repos: most core repos completed
- Priority 1 packages: most core packages completed
- Priority 2 repos: major references such as `QuantLib`, `dx`, `Financial-Models-Numerical-Methods`, `Kalman-and-Bayesian-Filters-in-Python`, `FinRL`, `FinGPT`, `ElegantRL`, `gluon-ts`, `darts`, `handson-ml3`, `fastbook`, `d2l-ko`, and `pytorch-beginner`
- Priority 2 packages: `QuantLib-Python`, `sympy`, `vollib`, `py_vollib`, `option-price`, `stochastic`, `numba`, `jax`, `dask`, `ray`, `gekko`, `filterpy`, `torch`, `torchvision`, `transformers`, `darts`, `sktime`, `tslearn`, `statsforecast`, `neuralforecast`, `tsai`, `scikit-learn`

### Failed or Blocked

- Some repository URLs are no longer valid or no longer exist upstream.
- One paper URL appears invalid:
  - `https://arxiv.org/pdf/1909.00000.pdf`
- Some older package names from the source list are not compatible with Python 3.12 or are not available on PyPI in the expected form.

## Validation Notes

- The download script was validated with `bash -n`.
- The generator script was validated with `python3 -m py_compile`.
- Download runs were executed end-to-end and produced actual artifacts.

## Current Recommendation

- Keep the implementation stack torch-first.
- Leave TensorFlow out of future runtime paths unless there is a strong, explicit reason to add it back.
- Use the manifest and bash script as the canonical inventory/download path for the bundle.
