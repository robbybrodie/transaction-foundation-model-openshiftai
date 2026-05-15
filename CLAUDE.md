# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an NVIDIA developer example for building a **financial transaction foundation model** end-to-end on NVIDIA GPUs. It pretrains a decoder-only Llama-like model (~29M parameters) on financial transaction sequences using causal language modeling, then demonstrates fraud detection using extracted embeddings.

The workflow is entirely notebook-driven (5 sequential notebooks) and requires NVIDIA GPUs and the NeMo Framework container.

## Environment Setup

All development runs inside the NeMo Framework container. There is no local pip install workflow — each notebook installs its own dependencies inline.

```bash
# Pull and launch NeMo container
docker run --gpus all --rm -it \
  -v $(pwd):/workspace \
  --shm-size=8g \
  -p 8888:8888 \
  --ulimit memlock=-1 \
  nvcr.io/nvidia/nemo:25.09.01

# Inside container: install Git LFS and fetch the pretrained checkpoint
git config --global --add safe.directory /workspace
apt-get update && apt-get install -y git-lfs
git lfs install
git lfs pull

# Start Jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

The `git config safe.directory` line is required because bind-mounting causes a Git ownership mismatch inside the container.

## Running Training

```bash
# Single GPU (testing, ~30 steps demo)
python scripts/train_decoder_model.py \
    -c configs/pretrain_financial_decoder.yaml \
    --dataset.data_path data/decoder_corpus/train_corpus.txt

# Multi-GPU (use torchrun directly — do NOT use the automodel CLI for multi-GPU,
# as it misinterprets --nproc-per-node as a config override)
torchrun --nproc-per-node=8 scripts/train_decoder_model.py \
    -c configs/pretrain_financial_decoder.yaml \
    --dataset.data_path data/decoder_corpus/train_corpus.txt \
    --validation_dataset.data_path data/decoder_corpus/val_corpus.txt
```

## CI Pipeline

CI runs notebooks sequentially via `papermill` inside the NeMo container on `arc-runners-org-nvidia-ai-bp-2-gpu` (self-hosted GPU runners). Requires the `NGC_API_KEY` secret. The workflow is defined in `.github/workflows/ci.yml`.

The CI pipeline strips kernel-shutdown cells before execution (using `ci_strip_kernel_shutdown.py`) and installs RAPIDS pip wheels (`cudf-cu12`, `cuml-cu12`) when conda/mamba is not available in the container.

## Architecture

### Notebook Workflow (must be run sequentially)

| Notebook | Purpose |
|----------|---------|
| `01_dataset_baseline.ipynb` | Load TabFormer dataset, temporal train/val/test splits, XGBoost baseline |
| `02_seq_preproc_tokenization.ipynb` | Build GPU-accelerated tokenizer pipeline |
| `03_foundation_model_training.ipynb` | Pretrain decoder model (30-step demo to `models/decoder-demo/`) |
| `04_inference_embedding_extraction.ipynb` | Extract 512-d embeddings via last-token pooling, UMAP visualization |
| `05_xgboost_fraud_detection.ipynb` | Compare XGBoost with raw features vs. embeddings |

**Notebooks 04–05 require the pretrained checkpoint from Git LFS** (`models/decoder-foundation-model/`), NOT the 30-step demo checkpoint from notebook 03.

### Source Code (`src/`)

- **`src/tokenizer/pipeline.py`** — `TokenizerPipeline`: orchestrates multiple `BaseTokenizer` steps with a global vocabulary with per-step offsets. Supports GPU CUDA streams for parallel fitting when ≥5 steps. Transforms transaction DataFrames into token strings, then assembles corpus lines in format `<bos> txn1 <sep> txn2 ... <eos>`.

- **`src/tokenizer/financial_tokenizer.py`** — `FinancialTabularTokenizer`: wraps `FinancialTokenizerPipeline` in an `encode()`/`decode()` API compatible with `FinancialCLMDataset`. Vocabulary is ~6,251 domain-specific tokens built from fixed-vocab tokenizer components (no data needed at construction time).

- **`src/tokenizer/financial_pipeline.py`** — `FinancialTokenizerPipeline`: concrete pipeline for financial data, wiring together amount, merchant, category, and temporal tokenizer steps.

- **`src/clm_data.py`** — `FinancialCLMDataset`: PyTorch `Dataset` for causal LM. Reads pre-generated text corpus lines (space-separated token strings) and produces `{input_ids, labels}` dicts. Labels use -100 for pad positions (HuggingFace convention). Entry point `build_financial_clm_dataset()` is called by NeMo AutoModel via YAML `_target_` resolution.

- **`src/decoder_inference.py`** — Inference and embedding extraction utilities used by notebook 04.

- **`src/tokenizer/base.py`** — `BaseTokenizer` abstract class for all tokenizer components.

- **`src/tokenizer/numerical.py`**, **`categorical_hash.py`**, **`fixed_vocab.py`**, **`mapping.py`**, **`timedelta.py`** — Individual tokenizer step implementations.

### Model Configuration (`configs/pretrain_financial_decoder.yaml`)

NeMo AutoModel YAML config. Key parameters:

- Architecture: `transformers.LlamaConfig` (architecture-agnostic — swap `_target_` to use other HuggingFace decoder models)
- Dataset `_target_` uses file-path syntax: `src/clm_data.py:build_financial_clm_dataset`
- Vocab size must match `FinancialTabularTokenizer` output (~6,251)
- Context window: 4,096 tokens (~315 transactions per sequence at 12 tokens/txn)
- Default `max_steps: 30` (demo) — increase for real training

### NeMo AutoModel Integration

The YAML config uses `_target_` to reference Python classes/functions by dotted path or `file:function` syntax. NeMo AutoModel resolves these at runtime, instantiates the dataset, and wraps it in a `StatefulDataLoader`. The training script adds `BLUEPRINT_ROOT` to `sys.path` so NeMo can resolve `src/clm_data.py` relative imports.

## OpenShift AI Deployment (`openshift/`)

The `openshift/` directory contains RHOAI 3.3 operator Custom Resources (not raw Kubernetes manifests). All resources are applied with `oc`, not `kubectl`, to an OpenShift **project** (`oc new-project`).

### CR inventory

| File | Kind / apiVersion | Purpose |
|------|-------------------|---------|
| `notebook-image/imagestream.yaml` | `ImageStream` (image.openshift.io/v1) | Registers custom NeMo workbench image with RHOAI dashboard via `opendatahub.io/notebook-image: "true"` label |
| `notebook-image/buildconfig.yaml` | `BuildConfig` (build.openshift.io/v1) | Builds workbench image inside the cluster (Docker strategy — NeMo base is not S2I-compatible) |
| `workbench/notebook.yaml` | `Notebook` (kubeflow.org/v1) | RHOAI Workbench; controller injects StatefulSet, Route, OAuth proxy sidecar — do NOT set `serviceAccountName` |
| `training/pytorchjob.yaml` | `PyTorchJob` (kubeflow.org/v1) | Distributed training; must use `serviceAccountName: nemo-tfm-training` for `anyuid` SCC |
| `pipeline/dspa.yaml` | `DataSciencePipelinesApplication` (v1alpha1) | Provisions KFP 2.5 pipeline server stack |
| `serving/serving-runtime.yaml` | `ServingRuntime` (serving.kserve.io/v1alpha1) | Model server container template |
| `serving/inference-service.yaml` | `InferenceService` (serving.kserve.io/v1beta1) | KServe RawDeployment model endpoint |
| `rbac/service-accounts.yaml` | SA + ClusterRole + RoleBinding | Grants `anyuid` SCC to `nemo-tfm-training` SA (NeMo runs as root; `restricted-v2` default SCC blocks this) |
| `templates/nemo-tfm-template.yaml` | `Template` (template.openshift.io/v1) | Parameterised deploy via `oc process` |
| `scripts/bootstrap-project.sh` | — | One-shot bootstrap for all of the above |

### Key constraints

- Do NOT set `nvidia.com/gpu` in both `ServingRuntime` and `InferenceService` predictor — the operator merges them and conflicting types cause errors.
- Do NOT add `serviceAccountName` to the `Notebook` CR — the Kubeflow Notebook Controller injects it.
- Use `PyTorchJob` (`kubeflow.org/v1`), not the KFTO v2 `TrainJob` — the latter is not supported in RHOAI 3.3.
- Secrets files (`*.env`, populated `*.yaml`) are gitignored. Copy from `*.template.*` files and fill in values.

### Apply order (or use bootstrap script)

```bash
source openshift/secrets/cluster-credentials.env
oc new-project nemo-tfm
oc apply -f openshift/secrets/registry-credentials.yaml -n nemo-tfm
oc secrets link default registry-pull-secret --for=pull -n nemo-tfm
oc secrets link builder registry-pull-secret -n nemo-tfm
oc apply -f openshift/secrets/workbench-secret.yaml -n nemo-tfm
oc apply -f openshift/rbac/service-accounts.yaml -n nemo-tfm
oc apply -f openshift/notebook-image/imagestream.yaml -n nemo-tfm
oc apply -f openshift/notebook-image/buildconfig.yaml -n nemo-tfm && oc start-build nemo-tfm-workbench -n nemo-tfm --follow
oc apply -f openshift/pipeline/dspa.yaml -n nemo-tfm
oc apply -f openshift/workbench/notebook.yaml -n nemo-tfm

# Or parameterised via Template:
oc process -f openshift/templates/nemo-tfm-template.yaml -p NAMESPACE=nemo-tfm ... | oc apply -f -
```

## Key Dependencies

- **NVIDIA NeMo AutoModel** — Training orchestration, distributed training, checkpointing
- **NVIDIA RAPIDS (cuDF, cuML, CuPy)** — GPU-accelerated tokenization and data processing
- **PyTorch + HuggingFace Transformers** — Model definitions and checkpointing
- **XGBoost + scikit-learn** — Downstream fraud detection evaluation
- **torchdata** — `StatefulDataLoader` for training
- **papermill** — Notebook execution in CI

## Hardware Requirements

- GPU: 1× NVIDIA A100 (80 GB) or H100 (minimum)
- System RAM: 32 GB
- CUDA 12 (provided via the NeMo container)
