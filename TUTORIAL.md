# NeMo Transaction Foundation Model — Platform Tutorial

> **Who this is for:** Data scientists who want to explore, run, and extend
> this project on the Red Hat OpenShift AI platform. No Kubernetes or
> infrastructure knowledge required.

---

## What this project does

Financial transaction data is one of the richest signals a bank has, but most
fraud models still rely on hand-crafted features that someone had to think of
in advance — "ratio of night-time purchases", "velocity in the last 7 days",
and so on. Foundation models take a different approach: instead of features,
the model reads raw transaction sequences and learns the patterns of normal
behaviour by itself, the same way a language model reads text. Once trained,
it produces a compact numerical "fingerprint" of each customer's behaviour
that captures structure no hand-crafted feature could.

This project does that end-to-end:

```
Raw transactions
      |
      v
  GPU tokeniser  (RAPIDS / cuDF — same schema as your production data)
      |
      v
  NeMo decoder   (29M-parameter Llama-style model, trained with causal LM)
      |
      v
  512-d embeddings  (one per customer, learned from behaviour)
      |
      v
  XGBoost fraud detector  (compare: raw features vs. embeddings vs. both)
```

The result is a concrete answer to "do foundation model embeddings actually
help with fraud detection on real financial data?" — with numbers, charts,
and reproducible code.

---

## What the platform gives you

Running this on Red Hat OpenShift AI means you get infrastructure you would
normally spend days setting up, ready immediately:

| What you get | What that means for you |
|---|---|
| **GPU-backed JupyterLab** | A100/H100 GPUs available to your notebooks without any setup |
| **NeMo + RAPIDS pre-installed** | The 37 GB NVIDIA container is already there — no `pip install` hell |
| **Pipeline tracking** | Every notebook run is logged: inputs, outputs, timing, and cell-by-cell results |
| **Reproducible runs** | Re-run any historical experiment exactly as it was |
| **Shared persistent storage** | Your data, checkpoints, and results persist across sessions and are visible to all team members |
| **Scales to multi-GPU** | Switch from 1 GPU to 8 by changing one number — no code changes |

You write notebooks the same way you always have. The platform handles
everything else.

---

## Getting access

The platform is already running. Ask your Red Hat contact for:

1. The **dashboard URL** — something like
   `https://rhods-dashboard-redhat-ods-applications.apps.<cluster>.opentlc.com`
2. Your **login credentials** — you log in with your OpenShift username and
   password, the same as any other Red Hat managed service

That's it. No VPN client to install, no SSH keys, no Docker.

---

## Opening the workbench

1. Open the dashboard URL in your browser and log in
2. In the left sidebar, click **Data Science Projects**
3. Click on **NeMo Transaction Foundation Model**
4. Click the **Workbenches** tab
5. Click **Launch** next to `nemo-tfm-workbench`

JupyterLab opens in a new tab. The environment has everything installed:
NeMo, RAPIDS (cuDF, cuML), PyTorch, XGBoost, and all project dependencies.

The file browser on the left shows the project files at `/nemo-tfm/`. The five
project notebooks are at the top level.

---

## The five notebooks

Run them in order. Each one saves its outputs so the next one can pick up
where it left off.

### `01_dataset_baseline.ipynb` — Dataset & XGBoost Baseline

Downloads the [TabFormer](https://github.com/IBM/TabFormer) financial
transaction dataset and sets up a temporal train/validation/test split — the
only split that reflects real-world deployment, where the model always predicts
into the future.

Then it trains an XGBoost model on hand-crafted features. This is your
baseline: the best you can do without a foundation model. Keep the AUC and
F1 numbers — you'll compare them at the end.

**What to look at:** The class imbalance chart. Real fraud data is heavily
skewed (often < 0.5% fraud). The notebook shows you how the split is
constructed and why the temporal ordering matters.

---

### `02_seq_preproc_tokenization.ipynb` — Sequence Tokenisation

This is where the tabular transaction records are converted into token
sequences that the NeMo decoder can read. The tokeniser runs on GPU using
RAPIDS cuDF, so it handles the full dataset in seconds rather than minutes.

The tokeniser is modular: each field (merchant category, amount band, time
delta, day of week) gets its own component with its own vocabulary. The
resulting sequences look like this:

```
<bos> MCC:retail AMOUNT:low DOW:mon DELTA:3h MCC:grocery AMOUNT:mid DOW:mon DELTA:2d ... <eos>
```

This structure preserves the sequential nature of a customer's transaction
history — something that flat feature vectors throw away.

**What to look at:** The vocabulary size breakdown by field. You can see
exactly what information the model has access to, and where you might want
to add or change fields for your own transaction schema.

---

### `03_foundation_model_training.ipynb` — Pre-training

Trains the NeMo decoder on the tokenised sequences using causal language
modelling — the same objective used to train GPT-style models, applied to
transaction tokens instead of text.

The notebook runs a 30-step demo so you can see the training loop work end
to end in a few minutes. Checkpoints are saved to `models/decoder-demo/`. For
a properly trained model, the pre-trained checkpoint from Git LFS
(`models/decoder-foundation-model/`) was trained for ~3,000 steps on 8 A100s
— that's what notebooks 04 and 05 use.

**What to look at:** The loss curve. You're watching the model learn the
grammar of transaction sequences. When loss drops, the model is getting better
at predicting what a customer will do next — which means it's learning what
"normal" looks like.

---

### `04_inference_embedding_extraction.ipynb` — Embedding Extraction

Loads the pre-trained model and runs inference over the test set. For each
transaction sequence, it extracts the final hidden state (last-token pooling)
as a 512-dimensional embedding vector — a learned fingerprint of that
customer's behaviour.

Then it uses UMAP to project those 512 dimensions down to 3 so you can see
them in a scatter plot. Fraudulent transactions should cluster away from
legitimate ones if the model has learned anything useful.

**What to look at:** The UMAP plot. Does fraud separate from non-fraud in
embedding space? If yes, the model has learned that fraud looks different
at a sequence level — without ever being told what fraud is during training.
This is the core promise of self-supervised pre-training.

---

### `05_xgboost_fraud_detection.ipynb` — Fraud Detection Comparison

The payoff. Trains three XGBoost models and compares them:

| Model | Features used |
|---|---|
| Baseline | Raw tabular features only (same as notebook 01) |
| Embeddings | 512-d NeMo embeddings only |
| Combined | Raw features + NeMo embeddings |

Reports AUC, F1, precision/recall, and a calibrated probability plot for each.

**What to look at:** The lift from embeddings. In the pre-trained model
results, the embedding model typically outperforms the raw-feature baseline
on AUC and, more importantly, on precision at high recall thresholds — which
is what matters in production fraud detection where you can't afford to miss
cases.

---

## Running the full pipeline (recommended)

Instead of running notebooks one by one, you can run them all as a tracked
pipeline from the RHOAI dashboard. Every run is recorded — you can inspect
each notebook's cell outputs, compare runs, and reproduce any historical result
exactly.

### Step 1 — Go to the Pipelines tab

From the project view in the RHOAI dashboard, click the **Pipelines** tab.
You should see a pipeline named **nemo-transaction-foundation-model**.

### Step 2 — Create a run

Click the pipeline name, then click **Create run** (top right).

Give your run a name — something like `baseline-run-<date>` — and leave all
other settings at their defaults. Click **Create**.

### Step 3 — Watch the progress

The pipeline view shows each notebook step as a node in a graph:

```
[1 - Dataset & XGBoost Baseline]
          |
[2 - Sequence Tokenisation]
          |
[3 - Foundation Model Pre-training]
          |
[4 - Embedding Extraction]
          |
[5 - XGBoost Fraud Detection]
```

Click any step to see its logs in real time. When a step completes, the
executed notebook (with all cell outputs) is saved to `pipeline-outputs/` on
the shared storage — you can open it in JupyterLab to see exactly what
happened.

### Step 4 — Review results

When the pipeline finishes, open `pipeline-outputs/05_xgboost_fraud_detection.ipynb`
in JupyterLab. All the charts and metrics tables are there, exactly as they
were when the pipeline ran.

---

## Customising the notebooks

The notebooks are plain Jupyter notebooks — you can edit them directly in the
workbench. Common things to try:

**Change the model size** — In `configs/pretrain_financial_decoder.yaml`,
increase `hidden_size`, `num_layers`, and `num_attention_heads`. Larger models
generally produce better embeddings but take longer to train.

**Extend the tokeniser** — In `src/tokenizer/financial_pipeline.py`, add
a new tokeniser step for a field that matters to your transaction schema
(location, channel, device type, etc.). Each step is a self-contained class.

**Swap the downstream classifier** — In notebook 05, replace XGBoost with
any sklearn-compatible classifier. The embeddings are standard numpy arrays.

**Train for longer** — In `configs/pretrain_financial_decoder.yaml`, increase
`max_steps` from 30 (demo) to 3000+ for a production-quality model.

After editing a notebook, either run it interactively from the workbench, or
re-trigger the pipeline from the dashboard to get a tracked, reproducible run.

---

## Understanding what OpenShift AI is doing

For those curious about the infrastructure — you don't need to know this to
use the platform, but it helps explain why things work the way they do.

```
Your browser
    |
    v
OpenShift AI Dashboard  (the UI you log into)
    |
    +-- Workbenches tab  --> your JupyterLab pod, with GPU attached
    |
    +-- Pipelines tab    --> Kubeflow Pipelines (KFP) v2
    |                        each step runs as its own pod
    |                        logs and outputs stored in S3
    |
    +-- Models tab       --> KServe model serving
                             OpenAI-compatible REST endpoint
```

When you click "Create run" in the pipeline UI, OpenShift AI schedules each
of the five notebook steps as an independent GPU pod in sequence. The pods
share a persistent volume (the same one your workbench uses), so files written
by step 2 are immediately available to step 3.

The key thing: **you didn't have to configure any of that.** The platform team
set it up once. From your side it's just: open a URL, run notebooks, see results.

---

## Getting help

| Question | Who to ask |
|---|---|
| Platform access, workbench not loading, pipeline errors | Your Red Hat contact |
| Model architecture, tokeniser design, training hyperparameters | This repository's issues page |
| Adapting to a different transaction schema | Open an issue or pull request — the tokeniser is designed for this |

---

## A note on the dataset

This project uses [TabFormer](https://github.com/IBM/TabFormer) — a synthetic
transaction dataset created by IBM Research for exactly this kind of
research. It has the same structure as real transaction data (merchant
categories, amounts, timestamps, fraud labels) but contains no real customer
information.

When adapting this to real Westpac transaction data, the main changes are:
1. Update the data loader in `01_dataset_baseline.ipynb` to point at your data source
2. Update the tokeniser field list in `src/tokenizer/financial_pipeline.py`
   to match your schema
3. The model architecture and training loop are unchanged

The platform handles the rest.
