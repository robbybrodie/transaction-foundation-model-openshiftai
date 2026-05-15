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

# The Kubeflow Notebook Controller (RHOAI) expects the container to start
# JupyterLab on port 8888. The NeMo base image defaults to /bin/bash, which
# exits immediately when run without a TTY in a pod. Override that here.
# The controller injects the OAuth proxy sidecar and adds --NotebookApp.base_url;
# ServerApp settings are additive so setting them here is safe.
CMD ["jupyter", "lab", \
     "--port=8888", \
     "--ip=0.0.0.0", \
     "--no-browser", \
     "--allow-root", \
     "--ServerApp.allow_remote_access=True", \
     "--ServerApp.token=''", \
     "--ServerApp.password=''"]
