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
