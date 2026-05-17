# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""True decorated KFP v2 components for the NeMo Transaction Foundation Model.

Each component calls src/ directly instead of running a notebook via papermill.
Contrast with nemo_tfm_pipeline.py, which wraps each notebook in papermill.

Why both pipelines exist:
  nemo_tfm_pipeline.py              -- production-proven papermill pipeline.
                                      Notebooks ARE the pipeline steps.
                                      Zero refactoring required.

  components_decorated.py +         -- demonstrates the next evolution step:
  foundation_model_pipeline_         typed I/O contracts, fine-grained GPU
  decorated.py                       allocation, KFP metadata store audit trail,
                                      and independent component testability.

See docs/notebooks-to-pipelines.md for the full trade-off analysis,
the maturity ladder, and the regulatory-defensibility argument.
"""

from kfp.dsl import component, Output, Metrics

# Internal registry only -- no DockerHub dependency.
# This image is built by the BuildConfig in openshift/gitops/notebook-image/
# and pushed to the cluster registry after ArgoCD syncs wave 2.
IMAGE = "image-registry.openshift-image-registry.svc:5000/nemo-tfm/nemo-tfm-workbench:latest"


# ---------------------------------------------------------------------------
# Component 0: prepare_dataset
# Replaces: data preparation portion of 01_dataset_baseline.ipynb
# (does NOT run the XGBoost baseline -- that lives in evaluate_fraud_detection)
# ---------------------------------------------------------------------------

@component(base_image=IMAGE)
def prepare_dataset(
    work_dir:    str,
    force_rerun: bool = False,
) -> str:
    """Download TabFormer dataset and create temporal train/val/test splits.

    True decorated component -- CPU only, no GPU required.
    Replaces the data preparation portion of notebook 01
    (01_dataset_baseline.ipynb). Does NOT run the XGBoost baseline --
    that is computed inside evaluate_fraud_detection alongside the
    embedding model for a fair comparison.

    What this does:
    - Downloads transactions.tgz from IBM Box if not present
    - Extracts card_transaction.v1.csv to data/TabFormer/raw/
    - Creates 80/10/10 temporal splits by cumulative row count
      (same logic as notebook 01 -- cutoff dates derived from
      find_cutoff_date at 0.8 and 0.9 cumulative ratios)
    - Writes train.parquet, val.parquet, test.parquet
    - Writes val_eval.parquet, test_eval.parquet
      (100K stratified subsets, random_state=42, used by
      extract_embeddings and evaluate_fraud_detection)

    All parquets are written in raw format (Amount still has $,
    Time is still a string) -- identical to what notebook 01 saves.

    Idempotent: if train.parquet exists and force_rerun=False,
    returns immediately without re-downloading or re-splitting.

    Better than papermill: the split directory path is a typed return
    value recorded in KFP metadata store. The notebook version writes
    wherever the cell says with no typed contract visible to downstream
    steps.

    Parameters
    ----------
    work_dir    : str  -- repo root on the shared PVC
    force_rerun : bool -- False: skip if splits already exist
                         True: delete and regenerate all splits

    Returns
    -------
    str
        Absolute path to data/TabFormer/temporal_split/
    """
    import os
    import sys
    import tarfile
    import time
    from urllib.request import urlretrieve

    sys.path.insert(0, work_dir)
    os.environ["HOME"] = "/tmp"

    split_dir     = os.path.join(work_dir, "data", "TabFormer", "temporal_split")
    train_parquet = os.path.join(split_dir, "train.parquet")

    if force_rerun and os.path.exists(split_dir):
        import shutil
        shutil.rmtree(split_dir)
        print("force_rerun=True -- cleared existing splits")

    if os.path.exists(train_parquet):
        import pandas as pd
        n = len(pd.read_parquet(train_parquet))
        print(f"Splits already exist: {n:,} train rows -- skipping")
        return split_dir

    # ------------------------------------------------------------------
    # Download and extract
    # ------------------------------------------------------------------
    raw_dir  = os.path.join(work_dir, "data", "TabFormer", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    tgz_path = os.path.join(work_dir, "data", "TabFormer", "transactions.tgz")
    csv_path = os.path.join(raw_dir, "card_transaction.v1.csv")

    DOWNLOAD_URL = (
        "https://ibm.ent.box.com/index.php"
        "?rm=box_download_shared_file"
        "&shared_name=mhrtz6xiknblqznoi9h4f390scoqustt"
        "&file_id=f_770766751708"
    )

    if not os.path.exists(tgz_path):
        print("Downloading transactions.tgz from IBM Box (~2.2 GB)...")
        t0 = time.time()
        urlretrieve(DOWNLOAD_URL, tgz_path)
        print(f"  Downloaded in {time.time()-t0:.1f}s")

    if not os.path.exists(csv_path):
        print("Extracting transactions.tgz...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=raw_dir)
        print(f"  Extracted: {csv_path}")

    print(f"Dataset ready: {csv_path}")

    # ------------------------------------------------------------------
    # Load (pandas -- CPU component, no cuDF dependency)
    # ------------------------------------------------------------------
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    TRAIN_RATIO  = 0.8
    VAL_RATIO    = 0.1
    EVAL_SAMPLES = 100_000
    RANDOM_STATE = 42

    print("Loading card_transaction.v1.csv with pandas...")
    t0 = time.time()
    raw_df = pd.read_csv(csv_path)
    raw_df.columns = [c.strip() for c in raw_df.columns]   # match notebook 01 cell-12
    print(f"  Loaded {len(raw_df):,} rows x {raw_df.shape[1]} cols in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Temporal split -- direct port of notebook 01 find_cutoff_date logic.
    # 80/10/10 by cumulative transaction count, not by fixed date.
    # ------------------------------------------------------------------
    print("Building temporal splits (80/10/10 by cumulative row count)...")
    t0 = time.time()

    year_s  = raw_df['Year'].astype(str)
    month_s = raw_df['Month'].astype(str).str.zfill(2)
    day_s   = raw_df['Day'].astype(str).str.zfill(2)
    raw_df['date'] = pd.to_datetime(
        year_s + '-' + month_s + '-' + day_s, format='%Y-%m-%d'
    )

    def find_cutoff_date(df, target_ratio):
        """First date where cumulative row count >= target_ratio * total.
        Direct port of notebook 01 find_cutoff_date."""
        daily = (df.groupby('date')
                   .size()
                   .reset_index(name='count')
                   .sort_values('date'))
        daily['cumulative'] = daily['count'].cumsum()
        total  = int(daily['cumulative'].iloc[-1])
        target = total * target_ratio
        return daily.loc[daily['cumulative'] >= target, 'date'].iloc[0]

    train_cutoff = find_cutoff_date(raw_df, TRAIN_RATIO)
    test_cutoff  = find_cutoff_date(raw_df, TRAIN_RATIO + VAL_RATIO)
    print(f"  Train/Val cutoff: {train_cutoff.strftime('%Y-%m-%d')}")
    print(f"  Val/Test  cutoff: {test_cutoff.strftime('%Y-%m-%d')}")

    train_mask = raw_df['date'] < train_cutoff
    val_mask   = (raw_df['date'] >= train_cutoff) & (raw_df['date'] < test_cutoff)
    test_mask  = raw_df['date'] >= test_cutoff

    train_df = raw_df[train_mask].drop(columns=['date']).reset_index(drop=True)
    val_df   = raw_df[val_mask].drop(columns=['date']).reset_index(drop=True)
    test_df  = raw_df[test_mask].drop(columns=['date']).reset_index(drop=True)
    del raw_df
    print(f"  Split time: {time.time()-t0:.1f}s")

    total = len(train_df) + len(val_df) + len(test_df)
    for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        n     = len(df)
        fraud = int(((df['Is Fraud?'] == 'Yes') | (df['Is Fraud?'] == '1')).sum())
        print(f"  {name:<6} {n:>12,}  ({n/total*100:.1f}%)  fraud {fraud:,}  ({fraud/n:.4%})")

    # ------------------------------------------------------------------
    # Save train / val / test parquets (raw format, no feature engineering)
    # ------------------------------------------------------------------
    os.makedirs(split_dir, exist_ok=True)
    t0 = time.time()
    for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        path = os.path.join(split_dir, f'{name}.parquet')
        df.to_parquet(path, index=False)
        print(f"  Saved {name}.parquet ({len(df):,} rows)")
    print(f"  Write time: {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # val_eval and test_eval -- 100K stratified subsets.
    # Mirrors notebook 01 section 5: stratified_subsample uses
    # train_test_split(..., stratify=_target, test_size=100_000,
    # random_state=42) then saves the raw rows (before feature
    # engineering) at the sampled positional indices.
    # val_eval and test_eval are used by extract_embeddings and
    # evaluate_fraud_detection; the full val/test parquets are used
    # by tokenize_transactions for corpus generation.
    # ------------------------------------------------------------------
    print(f"\nCreating {EVAL_SAMPLES:,}-row stratified eval subsets...")

    def stratified_subsample_idx(df, n_samples, random_state):
        """Return positional indices of a stratified sample.
        Stratify on binary fraud flag (mirrors notebook 01 _target logic)."""
        target = ((df['Is Fraud?'] == 'Yes') | (df['Is Fraud?'] == '1')).astype(int)
        if n_samples >= len(df):
            return df.index.to_numpy()
        _, sampled_idx = train_test_split(
            df.index,
            test_size=n_samples,
            stratify=target,
            random_state=random_state,
        )
        return sampled_idx

    for split_name, full_df in [('val_eval', val_df), ('test_eval', test_df)]:
        idx    = stratified_subsample_idx(full_df, EVAL_SAMPLES, RANDOM_STATE)
        subset = full_df.iloc[idx].reset_index(drop=True)
        out    = os.path.join(split_dir, f'{split_name}.parquet')
        subset.to_parquet(out, index=False)
        fraud  = int(((subset['Is Fraud?'] == 'Yes') | (subset['Is Fraud?'] == '1')).sum())
        print(f"  Saved {split_name}.parquet: {len(subset):,} rows "
              f"(fraud {fraud:,}, {fraud/len(subset):.4%})")

    print(f"\nAll splits saved to {split_dir}")
    return split_dir


# ---------------------------------------------------------------------------
# Component 1: tokenize_transactions
# Replaces: papermill on 02_seq_preproc_tokenization.ipynb
# ---------------------------------------------------------------------------

@component(base_image=IMAGE)
def tokenize_transactions(work_dir: str, force_rerun: bool = False) -> str:
    """Run the GPU-accelerated financial tokeniser and write text corpora.

    True decorated component -- calls src/ directly.
    Contrast with nemo_tfm_pipeline.py which uses papermill to run the
    equivalent notebook (02_seq_preproc_tokenization.ipynb).

    Reads the temporal splits from data/TabFormer/temporal_split/ (produced
    by notebook 01), runs FinancialTokenizerPipeline.preprocess() to derive
    the 12 pipeline columns, then calls fit_transform() and to_corpus_lines()
    to produce text sequences in the format expected by NeMo AutoModel:

        <bos> AMT_3 MERCH_1498 CAT_RETAIL ... CUST_42 <sep> AMT_1 ... <eos>

    Each line is ~315 transactions (~4096 tokens at 12 tok/txn + separators),
    matching the model's context window. Three corpus files are written:
    train_corpus.txt, val_corpus.txt, test_corpus.txt.

    Better than papermill: the corpus directory path is a typed return value
    recorded in KFP's metadata store. The notebook version writes wherever
    the cell says, with no typed contract visible to downstream steps.

    Returns
    -------
    str
        Absolute path to data/decoder_corpus/ on the shared PVC.
    """
    import os
    import sys
    import time

    # True decorated component -- calls src/ directly.
    sys.path.insert(0, work_dir)
    os.environ["HOME"] = "/tmp"

    import cudf
    from src.tokenizer import FinancialTokenizerPipeline

    MERCHANT_HASH_SIZE = 2000
    CHUNK_SIZE = 315  # ~4096 tokens at 12 tokens/txn + separators

    temporal_dir = os.path.join(work_dir, "data", "TabFormer", "temporal_split")
    corpus_dir   = os.path.join(work_dir, "data", "decoder_corpus")

    if force_rerun and os.path.exists(corpus_dir):
        import shutil
        shutil.rmtree(corpus_dir)
        print("force_rerun=True -- cleared existing corpus")

    os.makedirs(corpus_dir, exist_ok=True)

    splits = [
        ("train", os.path.join(temporal_dir, "train.parquet"),
                  os.path.join(corpus_dir, "train_corpus.txt")),
        ("val",   os.path.join(temporal_dir, "val.parquet"),
                  os.path.join(corpus_dir, "val_corpus.txt")),
        ("test",  os.path.join(temporal_dir, "test.parquet"),
                  os.path.join(corpus_dir, "test_corpus.txt")),
    ]

    for split_name, parquet_path, corpus_path in splits:
        if os.path.exists(corpus_path):
            n_lines = sum(1 for _ in open(corpus_path))
            print(f"[{split_name}] Corpus already exists: {n_lines:,} sequences -- skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Generating corpus for {split_name}")
        print(f"{'='*60}")

        t0 = time.time()
        gdf = cudf.read_parquet(parquet_path)
        print(f"  Loaded {len(gdf):,} rows in {time.time()-t0:.1f}s")

        pip = FinancialTokenizerPipeline(merchant_hash_size=MERCHANT_HASH_SIZE)
        gdf_proc = pip.preprocess(gdf)
        pip.fit(gdf_proc)
        token_df = pip.transform(gdf_proc)

        # Identify group columns for chunking by (user, card)
        group_cols = []
        for col_name in ["user", "User", "cust"]:
            if col_name in gdf_proc.columns:
                group_cols.append(col_name)
                break
        for col_name in ["card", "Card", "card_id"]:
            if col_name in gdf_proc.columns:
                group_cols.append(col_name)
                break
        if not group_cols:
            group_cols = [gdf_proc.columns[0]]

        corpus_lines = pip.to_corpus_lines(
            token_df, gdf_proc, group_cols, chunk_size=CHUNK_SIZE
        )

        with open(corpus_path, "w") as f:
            for line in corpus_lines:
                f.write(line + "\n")

        elapsed = time.time() - t0
        print(f"  Generated {len(corpus_lines):,} sequences in {elapsed:.1f}s")
        print(f"  Saved -> {corpus_path}")

    return corpus_dir


# ---------------------------------------------------------------------------
# Component 2: train_foundation_model
# Replaces: papermill on 03_foundation_model_training.ipynb
# ---------------------------------------------------------------------------

@component(base_image=IMAGE)
def train_foundation_model(
    work_dir:    str,
    corpus_dir:  str,
    config_path: str  = "configs/pretrain_financial_decoder.yaml",
    max_steps:   int  = 30,
    force_rerun: bool = False,
) -> str:
    """Run NeMo decoder pretraining via torchrun.

    True decorated component -- calls src/ directly.
    Contrast with nemo_tfm_pipeline.py which uses papermill to run the
    equivalent notebook (03_foundation_model_training.ipynb).

    Wraps torchrun --nproc-per-node=1 around scripts/train_decoder_model.py,
    passing the supplied YAML config and max_steps override. Single-GPU inside
    the KFP component container. For multi-node distributed training, use
    openshift/training/pytorchjob.yaml (KFTO PyTorchJob) instead -- the KFP
    pipeline can submit a PyTorchJob and poll for completion via the
    Kubernetes client (documented as a future enhancement).

    corpus_dir is an explicit typed input derived from tokenize_transactions.
    The data dependency ensures this component waits for tokenization to
    complete before launching training -- the corpus files must exist before
    the NeMo dataloader opens them.

    Better than papermill: max_steps is a typed pipeline parameter recorded in
    the KFP metadata store. You can compare runs and see exactly how many steps
    each training run executed, not just "notebook 03 ran."

    Parameters
    ----------
    work_dir    : str  -- repo root on the shared PVC
    corpus_dir  : str  -- path returned by tokenize_transactions (data dependency)
    config_path : str  -- path to NeMo YAML config, relative to work_dir
    max_steps   : int  -- training steps (30 = demo, increase for real training)

    Returns
    -------
    str
        Absolute path to models/decoder-demo/checkpoints/ on the shared PVC.
    """
    import os
    import sys
    import subprocess

    # True decorated component -- calls src/ directly.
    sys.path.insert(0, work_dir)
    os.environ["HOME"] = "/tmp"

    if force_rerun:
        import shutil
        checkpoint_dir_to_clear = os.path.join(work_dir, "models", "decoder-demo")
        if os.path.exists(checkpoint_dir_to_clear):
            shutil.rmtree(checkpoint_dir_to_clear)
            print("force_rerun=True -- cleared existing decoder-demo checkpoint")

    abs_config   = os.path.join(work_dir, config_path)
    train_script = os.path.join(work_dir, "scripts", "train_decoder_model.py")
    train_corpus = os.path.join(corpus_dir, "train_corpus.txt")
    val_corpus   = os.path.join(corpus_dir, "val_corpus.txt")

    print(f"NeMo pretraining: max_steps={max_steps}")
    print(f"  Config:         {abs_config}")
    print(f"  Train corpus:   {train_corpus}")
    print(f"  Val corpus:     {val_corpus}")

    env = os.environ.copy()
    env["PYTHONPATH"] = work_dir + ":" + env.get("PYTHONPATH", "")

    cmd = [
        "torchrun", "--nproc-per-node=1",
        train_script,
        "-c", abs_config,
        f"--dataset.data_path={train_corpus}",
        f"--validation_dataset.data_path={val_corpus}",
        f"--step_scheduler.max_steps={max_steps}",
    ]

    subprocess.run(cmd, check=True, cwd=work_dir, env=env)

    checkpoint_dir = os.path.join(work_dir, "models", "decoder-demo", "checkpoints")
    print(f"Training complete. Checkpoint dir: {checkpoint_dir}")
    return checkpoint_dir


# ---------------------------------------------------------------------------
# Component 3: extract_embeddings
# Replaces: papermill on 04_inference_embedding_extraction.ipynb
# ---------------------------------------------------------------------------

@component(base_image=IMAGE)
def extract_embeddings(
    work_dir:    str,
    model_path:  str  = "models/decoder-foundation-model",
    force_rerun: bool = False,
) -> str:
    """Extract 512-d last-token embeddings from the decoder model.

    True decorated component -- calls src/ directly.
    Contrast with nemo_tfm_pipeline.py which uses papermill to run the
    equivalent notebook (04_inference_embedding_extraction.ipynb).

    Runs GPU-accelerated cuDF tokenisation + HuggingFaceDecoderInference
    batch inference for all three splits:
      - train:  balanced ~1M sample (10% fraud, seed=42, matching notebook 05)
      - val:    100K stratified subset from val_eval.parquet
      - test:   100K stratified subset from test_eval.parquet

    Row IDs are preserved before preprocess() sorts by user/card/time, so
    raw feature alignment in evaluate_fraud_detection is exact.

    LFS smudge logic: if models/decoder-foundation-model/ contains pointer
    files (133-byte git-lfs stubs rather than real safetensors), the real
    blobs are fetched from GitHub LFS automatically before inference. This
    logic is copied verbatim from inference_embedding_extraction in
    nemo_tfm_pipeline.py so behaviour is identical in both pipelines.

    Better than papermill: model_path is an explicit typed input -- if you
    evaluate a different checkpoint the decision is recorded in the run's
    lineage. The papermill version buries the model path in a notebook cell.

    Parameters
    ----------
    work_dir   : str -- repo root on the shared PVC
    model_path : str -- relative or absolute path to the model checkpoint.
                       Default: models/decoder-foundation-model/ (Git LFS,
                       3000-step NVIDIA checkpoint). In both demo and train
                       modes this default is used -- training output is separate.

    Returns
    -------
    str
        Absolute path to data/embeddings/ on the shared PVC.
    """
    import glob
    import json
    import os
    import sys
    import subprocess
    import time

    import numpy as np

    # True decorated component -- calls src/ directly.
    sys.path.insert(0, work_dir)
    os.environ["HOME"] = "/tmp"

    # Resolve model_path relative to work_dir if not absolute
    if not os.path.isabs(model_path):
        model_path = os.path.join(work_dir, model_path)

    if force_rerun:
        import shutil
        embed_dir_to_clear = os.path.join(work_dir, "data", "embeddings")
        if os.path.exists(embed_dir_to_clear):
            shutil.rmtree(embed_dir_to_clear)
            os.makedirs(embed_dir_to_clear, exist_ok=True)
            print("force_rerun=True -- cleared existing embeddings")

    # ------------------------------------------------------------------
    # LFS smudge: detect pointer files and download real safetensors blobs.
    # Copied verbatim from inference_embedding_extraction in nemo_tfm_pipeline.py
    # -- behaviour is identical in both pipelines. Do not modify independently.
    # ------------------------------------------------------------------
    LFS_REPO = (
        "https://github.com/robbybrodie/"
        "transaction-foundation-model-openshiftai.git"
    )
    for model_file in glob.glob(os.path.join(model_path, "*.safetensors")):
        with open(model_file, "rb") as _f:
            header = _f.read(50)
        if not header.startswith(b"version https://git-lfs"):
            continue
        with open(model_file, encoding="utf-8") as _f:
            pointer_text = _f.read()
        oid = next(
            line.split("sha256:")[1].strip()
            for line in pointer_text.splitlines()
            if line.startswith("oid ")
        )
        size = int(next(
            line.split()[1]
            for line in pointer_text.splitlines()
            if line.startswith("size ")
        ))
        print(
            f"LFS pointer detected: {os.path.basename(model_file)} "
            f"({size:,} bytes). Fetching real blob..."
        )
        batch_body = json.dumps({
            "operation": "download",
            "transfers": ["basic"],
            "objects": [{"oid": oid, "size": size}],
        })
        batch_result = subprocess.run(
            [
                "curl", "-s", "-X", "POST",
                "-H", "Content-Type: application/vnd.git-lfs+json",
                "-H", "Accept: application/vnd.git-lfs+json",
                f"{LFS_REPO}/info/lfs/objects/batch",
                "-d", batch_body,
            ],
            capture_output=True, text=True, check=True,
        )
        batch_resp = json.loads(batch_result.stdout)
        download_url = batch_resp["objects"][0]["actions"]["download"]["href"]
        print(f"Downloading {os.path.basename(model_file)}...")
        subprocess.run(["curl", "-L", "-o", model_file, download_url], check=True)
        print(f"Smudged: {os.path.basename(model_file)} = {os.path.getsize(model_file):,} bytes")

    # ------------------------------------------------------------------
    # Initialise tokeniser and model
    # ------------------------------------------------------------------
    import cudf
    from src.tokenizer import FinancialTokenizerPipeline, FinancialTabularTokenizer
    from src.decoder_inference import HuggingFaceDecoderInference

    BATCH_SIZE          = 1024
    MAX_LENGTH          = 128
    MERCHANT_HASH_SIZE  = 2000
    BALANCED_TRAIN_SIZE = 1_000_000

    tokenizer = FinancialTabularTokenizer(
        merchant_hash_size=MERCHANT_HASH_SIZE,
        category_hierarchy=True,
        temporal_encoding=True,
    )
    inference = HuggingFaceDecoderInference(
        model_path=model_path,
        tokenizer=tokenizer,
        pooling="last_token",
    )
    print(f"Model: {inference.device}, embed_dim={inference.embedding_dim}")

    temporal_dir = os.path.join(work_dir, "data", "TabFormer", "temporal_split")
    embed_dir    = os.path.join(work_dir, "data", "embeddings")
    os.makedirs(embed_dir, exist_ok=True)

    split_to_parquet = {
        "train": "train.parquet",
        "val":   "val_eval.parquet",
        "test":  "test_eval.parquet",
    }

    all_embeddings = []
    all_labels     = []
    split_sizes    = {}

    for split in ("train", "val", "test"):
        embed_path  = os.path.join(embed_dir, f"{split}_embeddings.npy")
        label_path  = os.path.join(embed_dir, f"{split}_labels.npy")
        row_id_path = os.path.join(embed_dir, f"{split}_row_ids.npy")

        if os.path.exists(embed_path) and os.path.exists(label_path):
            emb = np.load(embed_path)
            lbl = np.load(label_path)
            print(f"[{split}] Already extracted: {emb.shape}, {lbl.sum():,} fraud -- skipping")
            all_embeddings.append(emb)
            all_labels.append(lbl)
            split_sizes[split] = len(emb)
            continue

        parquet_path = os.path.join(temporal_dir, split_to_parquet[split])
        print(f"\n{'='*60}\nExtracting {split} embeddings\n{'='*60}")

        t0  = time.time()
        gdf = cudf.read_parquet(parquet_path)

        # Extract labels before preprocess() renames columns
        labels = None
        for col in ["Is Fraud?", "is_fraud", "Is_Fraud", "label", "fraud"]:
            if col in gdf.columns:
                lbl_series = gdf[col].to_pandas()
                if lbl_series.dtype == object:
                    labels = ((lbl_series == "Yes") | (lbl_series == "1")).astype(int).values
                else:
                    labels = lbl_series.astype(int).values
                print(f"  Labels from '{col}': {labels.sum():,} fraud / {len(labels):,}")
                break

        # Balanced sampling for train -- must match notebook 05 (deterministic seed)
        if split == "train" and labels is not None:
            fraud_idx  = np.where(labels == 1)[0]
            normal_idx = np.where(labels == 0)[0]
            np.random.seed(42)
            n_fraud  = min(len(fraud_idx),  int(BALANCED_TRAIN_SIZE * 0.1))
            n_normal = min(len(normal_idx), BALANCED_TRAIN_SIZE - n_fraud)
            sampled  = np.concatenate([
                np.random.choice(fraud_idx,  n_fraud,  replace=False),
                np.random.choice(normal_idx, n_normal, replace=False),
            ])
            np.random.shuffle(sampled)
            gdf    = gdf.iloc[sampled].reset_index(drop=True)
            labels = labels[sampled]
            print(f"  Balanced: {len(gdf):,} rows, {labels.sum():,} fraud ({labels.mean():.1%})")

        # Preserve row IDs so raw features can be aligned after preprocess() sorts rows
        gdf["__row_id__"] = np.arange(len(gdf), dtype=np.int64)

        pip     = FinancialTokenizerPipeline(merchant_hash_size=MERCHANT_HASH_SIZE)
        gdf     = pip.preprocess(gdf)
        row_ids = gdf["__row_id__"].to_pandas().to_numpy(dtype=np.int64)
        if labels is not None:
            labels = labels[row_ids]
        pip.fit(gdf)
        token_df   = pip.transform(gdf)
        padded_ids = pip.encode(token_df, max_length=MAX_LENGTH)
        print(f"  Tokenized {len(padded_ids):,} rows in {time.time()-t0:.1f}s")

        t0  = time.time()
        emb = inference.extract_embeddings_batched(
            padded_ids, batch_size=BATCH_SIZE, show_progress=True
        )
        print(f"  Extracted {emb.shape} in {time.time()-t0:.1f}s")

        np.save(embed_path, emb)
        if labels is not None:
            np.save(label_path, labels)
        np.save(row_id_path, row_ids)

        all_embeddings.append(emb)
        all_labels.append(labels if labels is not None else np.zeros(len(emb), dtype=np.int8))
        split_sizes[split] = len(emb)

    # ------------------------------------------------------------------
    # Save concatenated embeddings + metadata.json
    # ------------------------------------------------------------------
    embeddings_all = np.concatenate(all_embeddings)
    labels_all     = np.concatenate(all_labels)
    np.save(os.path.join(embed_dir, "embeddings.npy"), embeddings_all)
    np.save(os.path.join(embed_dir, "labels.npy"),     labels_all)

    metadata = {
        "backend":           "huggingface_decoder",
        "pooling":           "last_token",
        "model_path":        str(model_path),
        "n_samples":         len(embeddings_all),
        "embedding_dim":     int(embeddings_all.shape[1]),
        "batch_size":        BATCH_SIZE,
        "max_length":        MAX_LENGTH,
        "splits":            ["train", "val", "test"],
        "n_train":           split_sizes.get("train", 0),
        "n_val":             split_sizes.get("val",   0),
        "n_test":            split_sizes.get("test",  0),
        "row_id_alignment":  "explicit_split_row_ids",
    }
    with open(os.path.join(embed_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll embeddings saved to {embed_dir}")
    for k, v in split_sizes.items():
        print(f"  {k}: {v:,}")

    return embed_dir


# ---------------------------------------------------------------------------
# Component 4: evaluate_fraud_detection
# Replaces: papermill on 05_xgboost_fraud_detection.ipynb
# CPU only -- no GPU accelerator set in the pipeline definition
# ---------------------------------------------------------------------------

@component(base_image=IMAGE)
def evaluate_fraud_detection(
    work_dir:        str,
    embeddings_path: str,
    metrics:         Output[Metrics],
) -> float:
    """XGBoost fraud detection: raw features vs. foundation model embeddings.

    True decorated component -- calls src/ directly. CPU only (no GPU).
    Contrast with nemo_tfm_pipeline.py which uses papermill to run the
    equivalent notebook (05_xgboost_fraud_detection.ipynb).

    Loads pre-extracted embeddings and raw tabular features, reduces 512d
    embeddings to 64d via PCA, then trains two HPO-optimized XGBoost models:
      1. Baseline: raw tabular features only  (13 fields, OrdinalEncoded)
      2. Embeddings: 64d PCA from 512d decoder embeddings

    Logs three structured metrics to the KFP metadata store:
      - auc_raw_features: test ROC-AUC for the raw feature baseline
      - auc_embeddings:   test ROC-AUC for the embedding model
      - lift_pct:         relative improvement (the headline demo number)

    These metrics appear in the OpenShift AI pipeline dashboard metrics tab --
    that is the screenshot that closes Demonstration 3 ("The model accuracy
    improvement is real"). Pointing at lift_pct and saying "this is the
    improvement" is only credible because the KFP metadata store records
    exactly what ran, on what data, with what parameters, at what time.
    APRA CPG 220 requires the latter.

    Better than papermill: both AUC values are structured metrics queryable
    across runs, comparable across versions, and visible in the dashboard
    without opening the notebook. The papermill version prints to a cell.

    Parameters
    ----------
    work_dir        : str            -- repo root on the shared PVC
    embeddings_path : str            -- path returned by extract_embeddings
    metrics         : Output[Metrics] -- KFP-injected metrics sink

    Returns
    -------
    float
        Test ROC-AUC for the embedding model (the headline accuracy number).
    """
    import os
    import sys
    import json

    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.compose import make_column_transformer, make_column_selector
    import xgboost as xgb
    import cudf

    # True decorated component -- calls src/ directly.
    sys.path.insert(0, work_dir)
    os.environ["HOME"] = "/tmp"

    embed_dir    = embeddings_path if os.path.isabs(embeddings_path) else os.path.join(work_dir, embeddings_path)
    temporal_dir = os.path.join(work_dir, "data", "TabFormer", "temporal_split")

    # ------------------------------------------------------------------
    # Load embeddings + labels + row IDs
    # ------------------------------------------------------------------
    print("Loading embeddings...")
    X_train_embed_raw = np.load(os.path.join(embed_dir, "train_embeddings.npy"))
    y_train           = np.load(os.path.join(embed_dir, "train_labels.npy"))
    train_row_ids     = np.load(os.path.join(embed_dir, "train_row_ids.npy"))
    X_val_embed_raw   = np.load(os.path.join(embed_dir, "val_embeddings.npy"))
    y_val             = np.load(os.path.join(embed_dir, "val_labels.npy"))
    val_row_ids       = np.load(os.path.join(embed_dir, "val_row_ids.npy"))
    X_test_embed_raw  = np.load(os.path.join(embed_dir, "test_embeddings.npy"))
    y_test            = np.load(os.path.join(embed_dir, "test_labels.npy"))
    test_row_ids      = np.load(os.path.join(embed_dir, "test_row_ids.npy"))

    n_train       = len(X_train_embed_raw)
    embed_dim_orig = X_train_embed_raw.shape[1]
    print(f"Embeddings: train={n_train:,}, val={len(X_val_embed_raw):,}, "
          f"test={len(X_test_embed_raw):,}, dim={embed_dim_orig}")

    # ------------------------------------------------------------------
    # PCA: 512d -> 64d
    # ------------------------------------------------------------------
    PCA_DIM = 64
    print(f"PCA: {embed_dim_orig}d -> {PCA_DIM}d")
    pca = PCA(n_components=PCA_DIM, random_state=42)
    X_train_embed_pca = pca.fit_transform(X_train_embed_raw)
    X_val_embed_pca   = pca.transform(X_val_embed_raw)
    X_test_embed_pca  = pca.transform(X_test_embed_raw)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    # ------------------------------------------------------------------
    # Load and align raw tabular features (GPU-accelerated via cuDF)
    # ------------------------------------------------------------------
    FEATURE_COLS   = [
        "User", "Card", "Year", "Month", "Day", "Hour", "Amount",
        "Use Chip", "Merchant Name", "Merchant City", "Merchant State",
        "Zip", "MCC",
    ]
    FRAUD_COL      = "Is Fraud?"
    BALANCED_TOTAL = n_train

    print("Loading raw features (cuDF)...")
    train_gdf = cudf.read_parquet(os.path.join(temporal_dir, "train.parquet"))
    val_gdf   = cudf.read_parquet(os.path.join(temporal_dir, "val_eval.parquet"))
    test_gdf  = cudf.read_parquet(os.path.join(temporal_dir, "test_eval.parquet"))

    for gdf in (train_gdf, val_gdf, test_gdf):
        gdf["Hour"]   = gdf["Time"].str.split(":", n=1, expand=True)[0].astype(int)
        gdf["Amount"] = (
            gdf["Amount"].str.replace("$", "", regex=False)
                          .str.replace(",", "").astype(float)
        )

    train_pdf = train_gdf.to_pandas()
    val_pdf   = val_gdf.to_pandas()
    test_pdf  = test_gdf.to_pandas()
    del train_gdf, val_gdf, test_gdf

    # Recreate balanced sample indices -- must match extract_embeddings (seed=42)
    fraud_mask    = (train_pdf[FRAUD_COL] == "Yes") | (train_pdf[FRAUD_COL] == "1")
    fraud_idx     = train_pdf.index[fraud_mask].tolist()
    normal_idx    = train_pdf.index[~fraud_mask].tolist()
    np.random.seed(42)
    n_fraud_tgt   = min(len(fraud_idx),  int(BALANCED_TOTAL * 0.1))
    n_normal_tgt  = min(len(normal_idx), BALANCED_TOTAL - n_fraud_tgt)
    sampled_fraud  = np.random.choice(fraud_idx,  n_fraud_tgt,  replace=False)
    sampled_normal = np.random.choice(normal_idx, n_normal_tgt, replace=False)
    balanced_idx   = np.concatenate([sampled_fraud, sampled_normal])
    np.random.shuffle(balanced_idx)

    X_train_raw = (
        train_pdf.loc[balanced_idx, FEATURE_COLS]
        .reset_index(drop=True).iloc[train_row_ids].reset_index(drop=True)
    )
    X_val_raw  = val_pdf.iloc[val_row_ids][FEATURE_COLS].reset_index(drop=True)
    X_test_raw = test_pdf.iloc[test_row_ids][FEATURE_COLS].reset_index(drop=True)

    preprocessor = make_column_transformer(
        (
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            make_column_selector(dtype_include=["object", "category"]),
        ),
        remainder="passthrough",
    )
    X_train_enc = preprocessor.fit_transform(X_train_raw)
    X_val_enc   = preprocessor.transform(X_val_raw)
    X_test_enc  = preprocessor.transform(X_test_raw)
    print(f"Raw features encoded: {X_train_enc.shape[1]}d")

    # ------------------------------------------------------------------
    # Train XGBoost models (HPO-optimized params from notebook 05)
    # ------------------------------------------------------------------
    import torch
    XGB_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Hyperparameters from Optuna tuning in notebook 05
    XGB_PARAMS_RAW = {
        "n_estimators": 400, "max_depth": 8, "learning_rate": 0.0023,
        "colsample_bytree": 0.95, "min_child_weight": 12, "subsample": 0.673,
        "reg_alpha": 0.01, "reg_lambda": 0.001, "random_state": 42,
    }
    XGB_PARAMS_EMBED = {
        "n_estimators": 435, "max_depth": 12, "learning_rate": 0.03774,
        "colsample_bytree": 0.587, "min_child_weight": 2.61, "subsample": 0.569,
        "reg_alpha": 0.01364, "reg_lambda": 9.7e-05, "gamma": 1.7, "random_state": 42,
    }

    def _train_xgb(X_tr, y_tr, X_v, y_v, X_te, y_te, params, name):
        clf = xgb.XGBClassifier(
            **params,
            tree_method="hist",
            device=XGB_DEVICE,
            early_stopping_rounds=20,
            eval_metric="auc",
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
        test_auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
        print(f"  {name}: test ROC-AUC = {test_auc:.4f}")
        return float(test_auc)

    print("\nTraining baseline (raw features)...")
    baseline_auc = _train_xgb(
        X_train_enc, y_train, X_val_enc, y_val, X_test_enc, y_test,
        XGB_PARAMS_RAW, f"Raw features ({X_train_enc.shape[1]}d)",
    )

    print("Training embeddings model (PCA 64d)...")
    embedding_auc = _train_xgb(
        X_train_embed_pca, y_train, X_val_embed_pca, y_val, X_test_embed_pca, y_test,
        XGB_PARAMS_EMBED, "Embeddings (PCA 64d)",
    )

    lift_pct = (embedding_auc / baseline_auc - 1) * 100

    # ------------------------------------------------------------------
    # Log structured metrics to KFP metadata store.
    # These appear in the RHOAI pipeline dashboard metrics tab.
    # lift_pct is the headline number for Demonstration 3.
    # ------------------------------------------------------------------
    metrics.log_metric("auc_raw_features", round(baseline_auc, 4))
    metrics.log_metric("auc_embeddings",   round(embedding_auc, 4))
    metrics.log_metric("lift_pct",         round(lift_pct, 2))

    print(f"\n{'='*50}")
    print(f"RESULTS (logged to KFP metadata store)")
    print(f"{'='*50}")
    print(f"  auc_raw_features : {baseline_auc:.4f}")
    print(f"  auc_embeddings   : {embedding_auc:.4f}")
    print(f"  lift_pct         : {lift_pct:+.2f}%")
    print(f"\nThese metrics appear in the RHOAI dashboard -> Runs -> Metrics tab.")

    return embedding_auc
