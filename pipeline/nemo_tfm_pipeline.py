"""NeMo Transaction Foundation Model — KFP v2 pipeline definition.

Five sequential steps, each running one of the project's Jupyter notebooks
via papermill.  All steps share the workbench PVC so data, checkpoints,
and model artefacts are visible across the whole run.

The pipeline can be compiled to YAML by running this file directly:

    python pipeline/nemo_tfm_pipeline.py
    # → pipeline/nemo_tfm_pipeline.yaml

Registration is done via 00_register_pipeline.ipynb in this directory.
"""

from kfp import dsl, compiler
from kfp.dsl import component
from kfp import kubernetes

# ---------------------------------------------------------------------------
# Image and storage — must match what's deployed in OpenShift
# ---------------------------------------------------------------------------
IMAGE = "image-registry.openshift-image-registry.svc:5000/nemo-tfm/nemo-tfm-workbench:latest"
PVC_NAME = "nemo-tfm-workbench-data"
# Repo is cloned to this path on the workbench PVC
WORK_DIR = "/opt/app-root/src/nemo-tfm"

PIPELINE_NAME = "nemo-transaction-foundation-model"
PIPELINE_DESCRIPTION = (
    "End-to-end NeMo decoder foundation model: "
    "data prep -> tokenisation -> pre-training -> embedding extraction -> fraud detection"
)


# ---------------------------------------------------------------------------
# Shared component: execute any notebook via papermill
# ---------------------------------------------------------------------------

@component(base_image=IMAGE)
def run_notebook(notebook_name: str, work_dir: str, prev: str = "") -> str:
    """Run a Jupyter notebook via papermill and return the output path.

    ``prev`` is intentionally unused — it exists only to wire a data-dependency
    between steps so KFP executes them sequentially rather than in parallel.
    Executed notebooks are saved to ``<work_dir>/pipeline-outputs/``.

    Cells containing kernel-shutdown calls (``do_shutdown(True)`` or
    ``IPython.Application.instance()``) are stripped before execution —
    identical to the CI pipeline's ci_strip_kernel_shutdown.py logic.
    These cells are meant for interactive workbench use and cause papermill
    to receive a DeadKernelError mid-run when not stripped.
    """
    import json
    import os
    import subprocess
    import tempfile

    # --- strip kernel-shutdown cells (same markers as CI ci_strip_kernel_shutdown.py) ---
    SHUTDOWN_MARKERS = ("do_shutdown(True)", "IPython.Application.instance()")

    input_nb = os.path.join(work_dir, notebook_name)
    output_nb = os.path.join(work_dir, "pipeline-outputs", notebook_name)
    os.makedirs(os.path.dirname(output_nb), exist_ok=True)

    with open(input_nb, encoding="utf-8") as f:
        nb = json.load(f)

    original_count = len(nb["cells"])
    nb["cells"] = [
        cell for cell in nb["cells"]
        if not any(marker in "".join(cell.get("source", [])) for marker in SHUTDOWN_MARKERS)
    ]
    removed = original_count - len(nb["cells"])
    if removed:
        print(f"Stripped {removed} kernel-shutdown cell(s) from {notebook_name}")

    # Write the stripped notebook to a temp file so the original is unchanged
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ipynb", dir="/tmp", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(nb, tmp, ensure_ascii=False)
        stripped_nb = tmp.name

    # Pipeline step pods run with a restricted SCC that cannot write to
    # /opt/app-root/src/.local (the default pip --user install target).
    # HOME is set to /tmp at the pod level (see _configure_task) but we
    # also copy it here for clarity and to ensure the subprocess inherits it.
    env = os.environ.copy()
    env["HOME"] = "/tmp"

    subprocess.run(
        [
            "papermill",
            stripped_nb,
            output_nb,
            "--kernel", "python3",
            "--no-progress-bar",
        ],
        check=True,
        cwd=work_dir,
        env=env,
    )
    return output_nb


# ---------------------------------------------------------------------------
# Helper: attach the shared PVC and set required pod environment variables
# ---------------------------------------------------------------------------

def _configure_task(task):
    # Mount the PVC at /opt/app-root/src — the same mount point used by the
    # workbench pod.  The repo is cloned to nemo-tfm/ within the PVC, so
    # notebooks resolve correctly at WORK_DIR = /opt/app-root/src/nemo-tfm.
    # (Mounting at WORK_DIR instead would put the PVC root one level too high,
    # making notebooks unreachable at the path papermill expects.)
    kubernetes.mount_pvc(task, pvc_name=PVC_NAME, mount_path="/opt/app-root/src")
    # The KFP launcher bootstraps itself by running 'pip install kfp' before
    # executing the component code.  Pipeline step pods run with a restricted
    # SCC (non-root, read-only system dirs) so pip defaults to --user install
    # at ~/.local, which resolves to /opt/app-root/src/.local — not writable.
    # Setting HOME=/tmp redirects pip's user-scheme to /tmp/.local, which is
    # always writable, fixing both the launcher bootstrap and any %pip install
    # cells inside the notebooks.
    task.set_env_variable("HOME", "/tmp")
    return task


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------

@dsl.pipeline(name=PIPELINE_NAME, description=PIPELINE_DESCRIPTION)
def nemo_tfm_pipeline(work_dir: str = WORK_DIR):
    """Five-step end-to-end pipeline for the NeMo transaction foundation model."""

    # Step 1 — load the TabFormer dataset and run the XGBoost baseline
    s1 = run_notebook(
        notebook_name="01_dataset_baseline.ipynb",
        work_dir=work_dir,
    )
    s1.set_display_name("1 - Dataset & XGBoost Baseline")
    s1.set_cpu_request("2").set_memory_request("8G")
    _configure_task(s1)

    # Step 2 — GPU-accelerated tokenisation pipeline (cuDF / cuML)
    s2 = run_notebook(
        notebook_name="02_seq_preproc_tokenization.ipynb",
        work_dir=work_dir,
        prev=s1.output,
    )
    s2.set_display_name("2 - Sequence Tokenisation")
    s2.set_cpu_request("4").set_memory_request("32G")
    s2.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1)
    _configure_task(s2)

    # Step 3 — pre-train the NeMo decoder foundation model
    s3 = run_notebook(
        notebook_name="03_foundation_model_training.ipynb",
        work_dir=work_dir,
        prev=s2.output,
    )
    s3.set_display_name("3 - Foundation Model Pre-training")
    s3.set_cpu_request("8").set_memory_request("64G")
    s3.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1)
    _configure_task(s3)

    # Step 4 — extract 512-d embeddings from the trained model
    s4 = run_notebook(
        notebook_name="04_inference_embedding_extraction.ipynb",
        work_dir=work_dir,
        prev=s3.output,
    )
    s4.set_display_name("4 - Embedding Extraction")
    s4.set_cpu_request("4").set_memory_request("32G")
    s4.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1)
    _configure_task(s4)

    # Step 5 — compare XGBoost with raw features vs. NeMo embeddings
    s5 = run_notebook(
        notebook_name="05_xgboost_fraud_detection.ipynb",
        work_dir=work_dir,
        prev=s4.output,
    )
    s5.set_display_name("5 - XGBoost Fraud Detection")
    s5.set_cpu_request("4").set_memory_request("16G")
    _configure_task(s5)


# ---------------------------------------------------------------------------
# Compile when run as a script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    output = os.path.join(os.path.dirname(__file__), "nemo_tfm_pipeline.yaml")
    compiler.Compiler().compile(nemo_tfm_pipeline, output)
    print(f"Compiled → {output}")
