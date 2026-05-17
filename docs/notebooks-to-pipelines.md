# From Notebooks to Pipelines: A Trade-off Analysis

Notebooks are the right starting point for ML experimentation; KFP decorated components
are the right destination for production. The question is not *whether* to migrate but
*how far* along the ladder to go and *when*.

---

## The Five Options

Every ML team that has working notebooks faces five distinct paths when they want to
operationalise. They are ordered by the amount of refactoring required and the
operational capability they unlock.

### Option 1 — Run Notebooks Manually

Run each notebook in order from the workbench UI.

**What you get:** Full interactive control, zero infrastructure overhead.
**What you don't get:** Reproducibility, scheduling, audit trail, failure recovery.
**When it's right:** During active R&D, when the experiment is changing daily.

The five notebooks in this project (`01_` through `05_`) were built here first.
This is where they should have been.

---

### Option 2 — Papermill in a KFP Component (this repo, `nemo_tfm_pipeline.py`)

Each KFP step calls `papermill <notebook> <output>` inside a container that already
has all dependencies installed. The notebook itself does not change. The pipeline
definition is a thin wrapper.

**What you get:**
- Pipeline visible in the RHOAI dashboard
- Every run produces an executed `.ipynb` output (full cell-level log)
- Caching disabled; every step re-executes on each run
- Shared PVC mounts the same data across all steps
- Node selector pins all steps to the GPU node holding the PVC

**What you don't get:**
- Typed I/O contracts between steps
- KFP metadata store entries for intermediate artefacts
- Fine-grained resource allocation per logical operation
- Independent component testability (can't unit-test a papermill wrapper)

**When it's right:** When you want pipeline orchestration without touching the notebooks.
This is the production-proven path for this project. The accuracy numbers in the
RHOAI dashboard come from this pipeline.

**Honest cost:** The `prev` parameter threading (Option 2's way of forcing step order)
is a workaround for the absence of typed outputs. It works and is explicit, but it
is not what KFP was designed for.

---

### Option 3 — Hybrid: Papermill for GPU, Decorated for Evaluation

Split the pipeline: keep GPU-intensive notebook steps under papermill (zero refactoring
risk on the expensive steps), and add a true `@component` for the final CPU-only
evaluation step to get structured metrics in the KFP metadata store.

**What you get:** Structured metrics (AUC values, lift) queryable across runs, visible
in the dashboard, without touching the GPU notebooks.
**What you don't get:** Typed I/O for the GPU steps.
**When it's right:** When you want dashboard metrics now but can't risk touching the
training or embedding code.

This is a pragmatic intermediate step many teams land on.

---

### Option 4 — True Decorated Components (this repo, `components_decorated.py`)

Each step is a `@component`-decorated function that calls `src/` directly. No notebooks
involved at runtime. Typed inputs and outputs. The KFP metadata store records every
artefact path, every hyperparameter, every metric.

**Components (demo: s0 → s1 → s3 → s4; train: s0 → s1 → s2 → s3 → s4):**

| Step | Component | What it does | demo | train |
|------|-----------|--------------|:----:|:-----:|
| s0 | `prepare_dataset` | Downloads TabFormer CSV; creates 80/10/10 temporal splits + 100K eval subsets; CPU only | ✓ | ✓ |
| s1 | `tokenize_transactions` | GPU cuDF tokeniser; returns `corpus_dir` typed output | ✓ | ✓ |
| s2 | `train_foundation_model` | NeMo pretraining via `torchrun`; records `max_steps` in metadata store | — | ✓ |
| s3 | `extract_embeddings` | 512-d last-token embeddings from LFS checkpoint | ✓ | ✓ |
| s4 | `evaluate_fraud_detection` | PCA 64d + XGBoost (3 models); logs 5 structured metrics: `auc_raw_features`, `auc_embeddings`, `lift_pct`, `auc_combined`, `lift_combined_pct` | ✓ | ✓ |

**What you get:**
- Self-contained on a fresh PVC — `prepare_dataset` (s0) downloads the TabFormer dataset and
  creates all required parquet splits; no prior papermill run needed
- Typed I/O contracts (e.g. `corpus_dir: str` flows from tokenise → train, recorded in lineage)
- Five structured metrics visible in the dashboard Metrics tab — queryable across versions
- `max_steps` is a pipeline parameter recorded per-run, not buried in a notebook cell
- Components can be unit-tested in isolation without running the full pipeline
- Fine-grained GPU allocation (s0 and s4: CPU only; s1 and s3: 1 GPU each)

**What you don't get:**
- Free: cell-by-cell log output that papermill provides (you get container logs instead)
- Zero risk: this path requires the `src/` modules to be importable from the component
  container, which means `sys.path` management and PYTHONPATH discipline

**When it's right:** When regulatory or operational requirements demand a full lineage
record, or when you need to run HPO across pipeline configurations and need metrics
to be comparable across runs.

**Honest cost:** The refactoring from notebooks to `src/` calls is real work. In this
project that work was done upfront (`src/tokenizer/`, `src/decoder_inference.py`),
which is why Option 4 was straightforward to implement. If your code lives entirely
in notebook cells, Option 4 requires extracting functions first.

---

### Option 5 — Ray, Argo Workflows, or Custom Orchestrators

Replace KFP entirely with a different orchestration layer.

**When it's right:** When you have very large distributed training jobs (thousands of
GPUs), complex fan-out/fan-in patterns, or existing infrastructure on a different
orchestrator.
**When it's not:** For most financial ML workflows at team scale, the complexity cost
of a custom orchestrator outweighs the benefit. KFP + RHOAI is the supported path on
OpenShift AI.

---

## The Maturity Ladder

```
Option 1   Notebooks manually                No infrastructure needed
    │
Option 2   Papermill in KFP                  Dashboard + logs, zero refactoring
    │
Option 3   Hybrid (papermill + 1 component)  Structured metrics, minimal risk
    │
Option 4   True decorated components         Full lineage, typed I/O, testable
    │
Option 5   Alternative orchestrators         Large-scale distributed training
```

Each rung adds operational capability at the cost of refactoring effort.
The right answer depends on where you are in the product lifecycle, not on
which option sounds most sophisticated.

**This project implements Option 2 and Option 4 in parallel.** Option 2
(`nemo_tfm_pipeline.py`) is the production pipeline with a proven track record.
Option 4 (`foundation_model_pipeline_decorated.py`) demonstrates what the
same workflow looks like one rung higher on the ladder. Both produce the same
accuracy result; they differ in what the KFP metadata store knows about the run.

---

## Why Decorated Components for Regulatory Defensibility

APRA CPG 220 (Operational Resilience, Section 52) requires that material models
have documented governance, including version control, parameter records, and
reproducibility evidence.

The papermill pipeline (Option 2) satisfies the *execution* requirement: every run
produces a timestamped `.ipynb` file showing exactly which cells ran and what output
they produced. That is a strong audit artefact.

The decorated pipeline (Option 4) adds three things the papermill version cannot:

**1. Structured parameter records.**
`max_steps=30` is a KFP pipeline parameter recorded in the metadata store at run time.
You can query the API and get back every run of `nemo-tfm-foundation-model-decorated`
with the exact value of `max_steps` that was used. In the papermill version, that value
is in a notebook cell — readable, but not queryable.

**2. Structured metrics queryable across versions.**
Five metrics are logged via `Output[Metrics]` and appear in the RHOAI dashboard Metrics tab:
`auc_raw_features` (raw 13d features baseline), `auc_embeddings` (embeddings only, PCA 64d),
`lift_pct` (embeddings-only lift — expected to be negative; embeddings alone do not beat
well-engineered tabular features), `auc_combined` (raw + embeddings, 77d), and
`lift_combined_pct` (the headline number: combined model lift over the baseline, ~+1.82%).
You can compare all five across 10 runs in a single view without opening a notebook.
When a model governance board asks "did AUC change between the June and July versions?",
the answer is a dashboard screenshot, not a notebook cell output.

The negative `lift_pct` is deliberately exposed rather than hidden. Recording it makes the
positive `lift_combined_pct` credible: the embeddings carry signal that raw features do not
have, and the combined model captures it. Showing only the combined result without the
embeddings-only comparison would be incomplete governance evidence.

**3. Explicit data lineage.**
The typed `corpus_dir` output of `tokenize_transactions` is the typed input of
`train_foundation_model`. The KFP metadata store records this as a lineage edge.
In the papermill version, that relationship exists only in the `prev` parameter
threading — it enforces ordering but does not record what was passed.

None of this makes the papermill pipeline wrong. Both pipelines are correct.
The decorated pipeline makes the governance evidence machine-readable rather than
human-readable. That distinction matters when APRA asks to see it, not when you
are building it.

---

## Concrete Example: Notebook 02 Before and After

### Before (papermill wrapper in `nemo_tfm_pipeline.py`)

```python
@component(base_image=IMAGE)
def seq_preproc_tokenization(work_dir: str, prev: str = "") -> str:
    """GPU-accelerated sequence preprocessing and tokenisation (notebook 02)."""
    import subprocess, sys, os, json, tempfile

    notebook_name = "02_seq_preproc_tokenization.ipynb"
    # ... strip shutdown cells ...
    subprocess.run(
        ["papermill", stripped_nb, output_nb, "--kernel", "python3"],
        check=True, cwd=work_dir,
    )
    return output_nb   # ← returns the path of the executed notebook
```

The return value is the executed notebook path. The downstream step receives it only
to establish ordering — it reads from the PVC directly, not from the return value.
There is no typed contract for "where did the corpus end up?".

### After (true decorated component in `components_decorated.py`)

```python
@component(base_image=IMAGE)
def tokenize_transactions(work_dir: str) -> str:
    """Run the GPU-accelerated financial tokeniser and write text corpora."""
    import sys, os
    sys.path.insert(0, work_dir)

    from src.tokenizer import FinancialTokenizerPipeline
    import cudf

    corpus_dir = os.path.join(work_dir, "data", "decoder_corpus")
    # ... preprocess → fit → transform → to_corpus_lines ...
    return corpus_dir   # ← returns the actual artefact directory
```

The return value is `data/decoder_corpus/` — the actual output of the step.
The downstream `train_foundation_model` component declares `corpus_dir: str` as an
explicit typed input. KFP records this path in the metadata store. If the corpus
directory ever changes (e.g. a data versioning scheme), the change is visible in the
run's lineage without reading the notebook source.

The difference is small when things work. It is significant when a regulator, an
auditor, or a post-incident review asks "what data did training run X use?".

---

## GitOps Platform vs Pipeline: A Necessary Distinction

A common source of confusion: ArgoCD and KFP solve different problems.

**ArgoCD (GitOps)** manages *infrastructure*:
- Creates the namespace, RBAC, secrets, image streams
- Provisions the workbench, the pipeline server (DSPA), the serving runtime
- Keeps cluster state in sync with `openshift/gitops/` in the repo
- Does not know anything about ML runs or training jobs

**KFP (Kubeflow Pipelines)** manages *ML workflow execution*:
- Runs the data preparation, tokenisation, training, embedding, and evaluation steps
- Records run parameters, metrics, and artefact lineage
- Does not provision infrastructure — it assumes the cluster is already set up

The interaction between them is intentionally one-directional:
1. ArgoCD syncs `openshift/gitops/` and creates the DSPA (pipeline server)
2. A data scientist registers a pipeline via `00_register_pipeline.ipynb`
3. KFP executes the pipeline, creating pods that mount the PVC ArgoCD provisioned

ArgoCD does not manage KFP pipelines or pipeline runs. KFP does not interact with
ArgoCD. They share a namespace and a PVC; that is the extent of their coupling.

**Why this matters for the demo:** The GitOps demonstration (ArgoCD Synced/Healthy,
automated cluster provisioning) is independent of the ML accuracy demonstration
(lift_combined_pct in the dashboard). You can run the GitOps demo without running any ML.
You can run the ML demo on a manually-provisioned cluster without ArgoCD.
Both demonstrations are real; neither depends on the other.

---

## Summary Recommendation

| If you want… | Use… |
|---|---|
| Fast demo, proven accuracy | `nemo_tfm_pipeline.py` (Option 2) |
| Dashboard metrics queryable across runs | `foundation_model_pipeline_decorated.py` (Option 4) |
| To show both at once | Register both pipelines; they coexist |
| To add a new experiment step | Add a notebook first (Option 1), then wrap it in papermill (Option 2) |
| To satisfy a governance query about a specific run | Option 4 metrics tab — or the `pipeline-outputs/` notebook on the PVC |

The maturity ladder is not a one-way ratchet. A team that needs to move fast
should stay on Option 2. A team preparing for a model risk review should be on
Option 4 for at least the evaluation component. Both are implemented here so
the choice does not require rewriting anything — it requires registering a
different pipeline YAML.
