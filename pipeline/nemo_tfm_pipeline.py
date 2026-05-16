"""NeMo Transaction Foundation Model — KFP v2 pipeline definition.

Five sequential steps, each running one of the project's Jupyter notebooks
via papermill.  All steps share the workbench PVC so data, checkpoints,
and model artefacts are visible across the whole run.

Each step's name in the RHOAI Dashboard matches the source notebook filename
so data scientists can map steps to notebooks at a glance.

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
# Shared papermill body — inlined in each @component because KFP requires
# component functions to be fully self-contained.  notebook_name is the
# only thing that differs between steps.
# ---------------------------------------------------------------------------

def _papermill_body(notebook_name: str, work_dir: str) -> str:
    """Template — not called at runtime; see inline copies in each component."""
    ...


# ---------------------------------------------------------------------------
# Step 1: dataset_baseline  →  01_dataset_baseline.ipynb
# ---------------------------------------------------------------------------

@component(base_image=IMAGE)
def dataset_baseline(work_dir: str, prev: str = "") -> str:
    """Load the TabFormer dataset and run the XGBoost baseline (notebook 01)."""
    import json
    import os
    import subprocess
    import sys
    import tempfile

    notebook_name = "01_dataset_baseline.ipynb"
    env = os.environ.copy()
    env["HOME"] = "/tmp"

    subprocess.run(
        [sys.executable, "-m", "ipykernel", "install",
         "--user", "--name", "python3", "--display-name", "Python 3"],
        check=True, env=env,
    )

    SHUTDOWN_MARKERS = ("do_shutdown(True)", "IPython.Application.instance()")
    input_nb = os.path.join(work_dir, notebook_name)
    output_nb = os.path.join(work_dir, "pipeline-outputs", notebook_name)
    os.makedirs(os.path.dirname(output_nb), exist_ok=True)

    with open(input_nb, encoding="utf-8") as f:
        nb = json.load(f)

    kept, removed = [], 0
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if all(m in src for m in SHUTDOWN_MARKERS):
                removed += 1
                continue
        kept.append(cell)
    nb["cells"] = kept
    if removed:
        print(f"Stripped {removed} kernel-shutdown cell(s) from {notebook_name}")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ipynb", dir="/tmp", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(nb, tmp, ensure_ascii=False)
        tmp.write("\n")
        stripped_nb = tmp.name

    subprocess.run(
        ["papermill", stripped_nb, output_nb,
         "--kernel", "python3", "--no-progress-bar"],
        check=True, cwd=work_dir, env=env,
    )
    return output_nb


# ---------------------------------------------------------------------------
# Step 2: seq_preproc_tokenization  →  02_seq_preproc_tokenization.ipynb
# ---------------------------------------------------------------------------

@component(base_image=IMAGE)
def seq_preproc_tokenization(work_dir: str, prev: str = "") -> str:
    """GPU-accelerated sequence preprocessing and tokenisation (notebook 02)."""
    import json
    import os
    import subprocess
    import sys
    import tempfile

    notebook_name = "02_seq_preproc_tokenization.ipynb"
    env = os.environ.copy()
    env["HOME"] = "/tmp"

    subprocess.run(
        [sys.executable, "-m", "ipykernel", "install",
         "--user", "--name", "python3", "--display-name", "Python 3"],
        check=True, env=env,
    )

    SHUTDOWN_MARKERS = ("do_shutdown(True)", "IPython.Application.instance()")
    input_nb = os.path.join(work_dir, notebook_name)
    output_nb = os.path.join(work_dir, "pipeline-outputs", notebook_name)
    os.makedirs(os.path.dirname(output_nb), exist_ok=True)

    with open(input_nb, encoding="utf-8") as f:
        nb = json.load(f)

    kept, removed = [], 0
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if all(m in src for m in SHUTDOWN_MARKERS):
                removed += 1
                continue
        kept.append(cell)
    nb["cells"] = kept
    if removed:
        print(f"Stripped {removed} kernel-shutdown cell(s) from {notebook_name}")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ipynb", dir="/tmp", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(nb, tmp, ensure_ascii=False)
        tmp.write("\n")
        stripped_nb = tmp.name

    subprocess.run(
        ["papermill", stripped_nb, output_nb,
         "--kernel", "python3", "--no-progress-bar"],
        check=True, cwd=work_dir, env=env,
    )
    return output_nb


# ---------------------------------------------------------------------------
# Step 3: foundation_model_training  →  03_foundation_model_training.ipynb
# ---------------------------------------------------------------------------

@component(base_image=IMAGE)
def foundation_model_training(work_dir: str, prev: str = "") -> str:
    """Pre-train the NeMo decoder foundation model (notebook 03)."""
    import json
    import os
    import subprocess
    import sys
    import tempfile

    notebook_name = "03_foundation_model_training.ipynb"
    env = os.environ.copy()
    env["HOME"] = "/tmp"

    subprocess.run(
        [sys.executable, "-m", "ipykernel", "install",
         "--user", "--name", "python3", "--display-name", "Python 3"],
        check=True, env=env,
    )

    SHUTDOWN_MARKERS = ("do_shutdown(True)", "IPython.Application.instance()")
    input_nb = os.path.join(work_dir, notebook_name)
    output_nb = os.path.join(work_dir, "pipeline-outputs", notebook_name)
    os.makedirs(os.path.dirname(output_nb), exist_ok=True)

    with open(input_nb, encoding="utf-8") as f:
        nb = json.load(f)

    kept, removed = [], 0
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if all(m in src for m in SHUTDOWN_MARKERS):
                removed += 1
                continue
        kept.append(cell)
    nb["cells"] = kept
    if removed:
        print(f"Stripped {removed} kernel-shutdown cell(s) from {notebook_name}")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ipynb", dir="/tmp", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(nb, tmp, ensure_ascii=False)
        tmp.write("\n")
        stripped_nb = tmp.name

    subprocess.run(
        ["papermill", stripped_nb, output_nb,
         "--kernel", "python3", "--no-progress-bar"],
        check=True, cwd=work_dir, env=env,
    )
    return output_nb


# ---------------------------------------------------------------------------
# Step 4: inference_embedding_extraction  →  04_inference_embedding_extraction.ipynb
# ---------------------------------------------------------------------------

@component(base_image=IMAGE)
def inference_embedding_extraction(work_dir: str, prev: str = "") -> str:
    """Extract 512-d embeddings from the trained decoder model (notebook 04)."""
    import glob
    import json
    import os
    import subprocess
    import sys
    import tempfile

    notebook_name = "04_inference_embedding_extraction.ipynb"
    env = os.environ.copy()
    env["HOME"] = "/tmp"

    # ---------------------------------------------------------------------------
    # Smudge Git LFS pointer files in models/decoder-foundation-model/
    # The repo clone on the PVC may only have LFS pointer files (133-byte ASCII)
    # if `git lfs pull` was never run interactively.  This block detects any
    # pointer files and downloads the real blobs from GitHub LFS automatically.
    # ---------------------------------------------------------------------------
    LFS_REPO = (
        "https://github.com/robbybrodie/"
        "transaction-foundation-model-openshiftai.git"
    )
    models_dir = os.path.join(work_dir, "models", "decoder-foundation-model")
    for model_path in glob.glob(os.path.join(models_dir, "*.safetensors")):
        with open(model_path, "rb") as _f:
            header = _f.read(50)
        if not header.startswith(b"version https://git-lfs"):
            continue
        # It's a pointer — parse OID and size
        with open(model_path, encoding="utf-8") as _f:
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
            f"LFS pointer detected: {os.path.basename(model_path)} "
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
        print(f"Downloading {os.path.basename(model_path)}...")
        subprocess.run(
            ["curl", "-L", "-o", model_path, download_url],
            check=True, env=env,
        )
        print(f"Smudged: {os.path.basename(model_path)} = {os.path.getsize(model_path):,} bytes")

    subprocess.run(
        [sys.executable, "-m", "ipykernel", "install",
         "--user", "--name", "python3", "--display-name", "Python 3"],
        check=True, env=env,
    )

    SHUTDOWN_MARKERS = ("do_shutdown(True)", "IPython.Application.instance()")
    input_nb = os.path.join(work_dir, notebook_name)
    output_nb = os.path.join(work_dir, "pipeline-outputs", notebook_name)
    os.makedirs(os.path.dirname(output_nb), exist_ok=True)

    with open(input_nb, encoding="utf-8") as f:
        nb = json.load(f)

    kept, removed = [], 0
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if all(m in src for m in SHUTDOWN_MARKERS):
                removed += 1
                continue
        kept.append(cell)
    nb["cells"] = kept
    if removed:
        print(f"Stripped {removed} kernel-shutdown cell(s) from {notebook_name}")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ipynb", dir="/tmp", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(nb, tmp, ensure_ascii=False)
        tmp.write("\n")
        stripped_nb = tmp.name

    subprocess.run(
        ["papermill", stripped_nb, output_nb,
         "--kernel", "python3", "--no-progress-bar"],
        check=True, cwd=work_dir, env=env,
    )
    return output_nb


# ---------------------------------------------------------------------------
# Step 5: xgboost_fraud_detection  →  05_xgboost_fraud_detection.ipynb
# ---------------------------------------------------------------------------

@component(base_image=IMAGE)
def xgboost_fraud_detection(work_dir: str, prev: str = "") -> str:
    """Compare XGBoost with raw features vs. NeMo embeddings (notebook 05)."""
    import json
    import os
    import subprocess
    import sys
    import tempfile

    notebook_name = "05_xgboost_fraud_detection.ipynb"
    env = os.environ.copy()
    env["HOME"] = "/tmp"

    subprocess.run(
        [sys.executable, "-m", "ipykernel", "install",
         "--user", "--name", "python3", "--display-name", "Python 3"],
        check=True, env=env,
    )

    SHUTDOWN_MARKERS = ("do_shutdown(True)", "IPython.Application.instance()")
    input_nb = os.path.join(work_dir, notebook_name)
    output_nb = os.path.join(work_dir, "pipeline-outputs", notebook_name)
    os.makedirs(os.path.dirname(output_nb), exist_ok=True)

    with open(input_nb, encoding="utf-8") as f:
        nb = json.load(f)

    kept, removed = [], 0
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if all(m in src for m in SHUTDOWN_MARKERS):
                removed += 1
                continue
        kept.append(cell)
    nb["cells"] = kept
    if removed:
        print(f"Stripped {removed} kernel-shutdown cell(s) from {notebook_name}")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ipynb", dir="/tmp", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(nb, tmp, ensure_ascii=False)
        tmp.write("\n")
        stripped_nb = tmp.name

    subprocess.run(
        ["papermill", stripped_nb, output_nb,
         "--kernel", "python3", "--no-progress-bar"],
        check=True, cwd=work_dir, env=env,
    )
    return output_nb


# ---------------------------------------------------------------------------
# Helper: attach the shared PVC and set required pod environment variables
# ---------------------------------------------------------------------------

def _configure_task(task):
    # Mount the PVC at /opt/app-root/src — the same mount point used by the
    # workbench pod.  The repo is cloned to nemo-tfm/ within the PVC, so
    # notebooks resolve correctly at WORK_DIR = /opt/app-root/src/nemo-tfm.
    kubernetes.mount_pvc(task, pvc_name=PVC_NAME, mount_path="/opt/app-root/src")
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
    s1 = dataset_baseline(work_dir=work_dir)
    s1.set_cpu_request("2").set_memory_request("8G")
    _configure_task(s1)

    # Step 2 — GPU-accelerated tokenisation pipeline (cuDF / cuML)
    s2 = seq_preproc_tokenization(work_dir=work_dir, prev=s1.output)
    s2.set_cpu_request("4").set_memory_request("32G")
    s2.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1)
    _configure_task(s2)

    # Step 3 — pre-train the NeMo decoder foundation model
    s3 = foundation_model_training(work_dir=work_dir, prev=s2.output)
    s3.set_cpu_request("8").set_memory_request("64G")
    s3.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1)
    _configure_task(s3)

    # Step 4 — extract 512-d embeddings from the trained model
    s4 = inference_embedding_extraction(work_dir=work_dir, prev=s3.output)
    s4.set_cpu_request("4").set_memory_request("32G")
    s4.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1)
    _configure_task(s4)

    # Step 5 — compare XGBoost with raw features vs. NeMo embeddings
    s5 = xgboost_fraud_detection(work_dir=work_dir, prev=s4.output)
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
