# Workbench image for Transaction Foundation Model
#
# Based on the NVIDIA NeMo Framework container.
# The Kubeflow Notebook Controller (RHOAI) injects:
#   - USER / runAsUser (from namespace SCC)
#   - OAuth proxy sidecar container
#   - Jupyter port and probe configuration
#   - Service and Route
#
# Do NOT add USER, EXPOSE, or security context directives here.

FROM nvcr.io/nvidia/nemo:25.09.01

# Copy and install this repo's additional Python requirements.
# NeMo, PyTorch, RAPIDS (cuDF/cuML), and CUDA are already in the base image.
COPY requirements.txt /opt/app-root/src/requirements.txt
RUN pip install --no-cache-dir -r /opt/app-root/src/requirements.txt

# OpenShift runs containers as a random non-root UID. The NeMo base image
# sets HOME=/root, which is not writable under the restricted-v2 SCC.
# Point HOME at the PVC-backed mount so Jupyter can create its runtime dir.
ENV HOME=/opt/app-root/src

# The Kubeflow Notebook Controller (RHOAI) expects the container to start
# JupyterLab on port 8888. The NeMo base image defaults to /bin/bash, which
# exits immediately when run without a TTY in a pod. Override that here.
# NB_PREFIX is injected by the Notebook Controller (e.g. /notebook/nemo-tfm/nemo-tfm-workbench).
# Using sh -c so the variable is expanded at runtime.
CMD ["sh", "-c", "exec jupyter lab --port=8888 --ip=0.0.0.0 --no-browser --allow-root --ServerApp.allow_remote_access=True --ServerApp.token= --ServerApp.password= --ServerApp.base_url=${NB_PREFIX:-/} --ServerApp.root_dir=/opt/app-root/src"]
