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

"""Decorated KFP v2 pipeline for the NeMo Transaction Foundation Model.

Uses true decorated components from components_decorated.py instead of
papermill-wrapped notebook calls.  Contrast with nemo_tfm_pipeline.py.

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

Pipeline modes
--------------
mode="demo"  (default)
    s1 -> s3d -> s4d
    Tokenise, extract embeddings from the Git LFS checkpoint, evaluate.
    Training is skipped. Completes in ~20-30 min on an L4 GPU.

mode="train"
    s1 -> s2 -> s3t -> s4t
    Tokenise, then train a 30-step demo model before extracting embeddings
    and evaluating.  s3t uses the Git LFS checkpoint, not the 30-step output.

Compile to YAML by running this file directly:

    python pipeline/foundation_model_pipeline_decorated.py
    # -> pipeline/foundation_model_pipeline_decorated.yaml
"""

from kfp import dsl, compiler
from kfp import kubernetes

from components_decorated import (
    tokenize_transactions,
    train_foundation_model,
    extract_embeddings,
    evaluate_fraud_detection,
)

# ---------------------------------------------------------------------------
# Image and storage -- must match what's deployed in OpenShift
# ---------------------------------------------------------------------------
IMAGE    = "image-registry.openshift-image-registry.svc:5000/nemo-tfm/nemo-tfm-workbench:latest"
PVC_NAME = "nemo-tfm-workbench-data"
WORK_DIR = "/opt/app-root/src/nemo-tfm"

PIPELINE_NAME        = "nemo-tfm-foundation-model-decorated"
PIPELINE_DESCRIPTION = (
    "Decorated KFP v2 pipeline -- typed I/O components calling src/ directly. "
    "Contrast with nemo-transaction-foundation-model (papermill version)."
)


# ---------------------------------------------------------------------------
# Helper: attach the shared PVC and set required pod environment variables.
# Identical to _configure_task() in nemo_tfm_pipeline.py.
# ---------------------------------------------------------------------------

def _configure_task(task):
    """Mount PVC, set HOME, disable caching, pin to GPU node."""
    # Mount the PVC at /opt/app-root/src -- the same mount point used by the
    # workbench pod.  The repo is cloned to nemo-tfm/ within the PVC, so
    # notebooks resolve correctly at WORK_DIR = /opt/app-root/src/nemo-tfm.
    kubernetes.mount_pvc(task, pvc_name=PVC_NAME, mount_path="/opt/app-root/src")
    # Setting HOME=/tmp redirects pip's user-scheme to /tmp/.local, which is
    # always writable, fixing both the launcher bootstrap and any %pip install
    # cells inside the notebooks.
    task.set_env_variable("HOME", "/tmp")
    # Disable KFP caching -- components write results to the shared PVC so every
    # run should execute all steps rather than silently reusing stale outputs.
    task.set_caching_options(enable_caching=False)
    # Pin every pipeline step to the same GPU node as the workbench.
    # The shared PVC is EBS ReadWriteOnce -- it can only attach to one node at
    # a time.  The workbench's nodeSelector uses the same label, so both land
    # on the same node regardless of cluster topology.
    kubernetes.add_node_selector(
        task,
        label_key="nvidia.com/gpu.present",
        label_value="true",
    )
    return task


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------

@dsl.pipeline(name=PIPELINE_NAME, description=PIPELINE_DESCRIPTION)
def foundation_model_pipeline_decorated(
    work_dir:    str  = WORK_DIR,
    mode:        str  = "demo",
    model_path:  str  = "models/decoder-foundation-model",
    max_steps:   int  = 30,
    force_rerun: bool = False,
):
    """Switchable end-to-end pipeline using true decorated components.

    Parameters
    ----------
    work_dir    : str  -- absolute path to the cloned repo on the shared PVC.
    mode        : str  -- "demo" (default, skip training, ~20-30 min on L4)
                         or "train" (all steps including NeMo pretraining).
    model_path  : str  -- path to model checkpoint for embedding extraction.
                         Default: models/decoder-foundation-model/ (Git LFS,
                         3000-step NVIDIA checkpoint). Relative paths are
                         resolved from work_dir inside extract_embeddings.
    max_steps   : int  -- training steps; only used when mode="train".
                         30   = demo capability proof (fast).
                         500+ = meaningful training.
                         3000 = full NVIDIA equivalent (use H200s).
    force_rerun : bool -- False (default): reuse existing corpus/embeddings
                         if present -- makes repeated demo runs fast.
                         True: clear and regenerate all intermediate outputs.
                         Always set True for train mode to avoid stale demo
                         artifacts contaminating results.
    """

    # ------------------------------------------------------------------
    # Step 1: tokenize_transactions -- always runs in both modes.
    # Replaces: papermill on 02_seq_preproc_tokenization.ipynb
    # Returns: corpus_dir (data/decoder_corpus/)
    # ------------------------------------------------------------------
    s1 = tokenize_transactions(work_dir=work_dir, force_rerun=force_rerun)
    s1.set_cpu_request("4").set_memory_request("32G")
    s1.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1)
    _configure_task(s1)

    # ------------------------------------------------------------------
    # TRAIN MODE: s1 -> s2 -> s3t -> s4t
    # Wrap the full tail so GPU steps remain strictly sequential inside
    # the branch -- critical for the EBS ReadWriteOnce PVC constraint.
    # ------------------------------------------------------------------
    with dsl.If(mode == "train", name="train-mode"):

        # Step 2t: train_foundation_model
        # corpus_dir is an explicit typed input (data dependency on s1).
        # The KFP metadata store records max_steps for every run -- you can
        # compare runs and see exactly how many steps each executed.
        # Replaces: papermill on 03_foundation_model_training.ipynb
        s2t = train_foundation_model(
            work_dir=work_dir,
            corpus_dir=s1.output,
            max_steps=max_steps,
            force_rerun=force_rerun,
        )
        s2t.set_cpu_request("8").set_memory_request("64G")
        s2t.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1)
        _configure_task(s2t)

        # Step 3t: extract_embeddings
        # model_path defaults to the Git LFS 3000-step checkpoint -- NOT
        # the 30-step output -- so the accuracy story is consistent.
        # .after(s2t) is redundant here (KFP infers from s2t.output) but
        # is explicit to document the intended GPU serialisation.
        # Replaces: papermill on 04_inference_embedding_extraction.ipynb
        s3t = extract_embeddings(
            work_dir=work_dir,
            model_path=model_path,
            force_rerun=force_rerun,
        )
        s3t.after(s2t)
        s3t.set_cpu_request("4").set_memory_request("32G")
        s3t.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1)
        _configure_task(s3t)

        # Step 4t: evaluate_fraud_detection -- CPU only (no set_accelerator_type).
        # metrics: Output[Metrics] is injected by KFP -- do not pass it.
        # lift_pct is the headline number for Demonstration 3.
        # Replaces: papermill on 05_xgboost_fraud_detection.ipynb
        s4t = evaluate_fraud_detection(
            work_dir=work_dir,
            embeddings_path=s3t.output,
        )
        s4t.set_cpu_request("4").set_memory_request("16G")
        _configure_task(s4t)

    # ------------------------------------------------------------------
    # DEMO MODE: s1 -> s3d -> s4d  (training skipped)
    # extract_embeddings uses the Git LFS checkpoint directly.
    # .after(s1) enforces GPU serialisation -- s3d must wait for s1's
    # GPU to be released before requesting its own, even though s3d
    # does not consume s1's typed output.
    # ------------------------------------------------------------------
    with dsl.Else(name="demo-mode"):

        # Step 3d: extract_embeddings -- uses Git LFS checkpoint
        # Replaces: papermill on 04_inference_embedding_extraction.ipynb
        s3d = extract_embeddings(
            work_dir=work_dir,
            model_path=model_path,
            force_rerun=force_rerun,
        )
        s3d.after(s1)
        s3d.set_cpu_request("4").set_memory_request("32G")
        s3d.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1)
        _configure_task(s3d)

        # Step 4d: evaluate_fraud_detection -- CPU only.
        # Replaces: papermill on 05_xgboost_fraud_detection.ipynb
        s4d = evaluate_fraud_detection(
            work_dir=work_dir,
            embeddings_path=s3d.output,
        )
        s4d.set_cpu_request("4").set_memory_request("16G")
        _configure_task(s4d)


# ---------------------------------------------------------------------------
# Compile when run as a script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    output = os.path.join(
        os.path.dirname(__file__),
        "foundation_model_pipeline_decorated.yaml",
    )
    compiler.Compiler().compile(foundation_model_pipeline_decorated, output)
    print(f"Compiled -> {output}")
