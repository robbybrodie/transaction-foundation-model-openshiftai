#!/usr/bin/env bash
# bootstrap-project.sh
# --------------------
# One-shot OpenShift project bootstrap for the NeMo Transaction Foundation
# Model deployment. Applies all operator CRs in dependency order using oc.
#
# Prerequisites:
#   - oc CLI logged in: oc login --server=<url> --token=<token>
#   - Credentials populated in openshift/secrets/
#     (copy from *.template.env / *.template.yaml and fill values)
#   - Cluster-admin access (required for ClusterRole and SCC RoleBinding)
#
# Usage:
#   # From repo root:
#   source openshift/secrets/cluster-credentials.env
#   bash openshift/scripts/bootstrap-project.sh
#
#   # Or pass namespace directly:
#   NAMESPACE=my-project bash openshift/scripts/bootstrap-project.sh
#
# What this script does (in order):
#   1.  Creates the OpenShift project (oc new-project)
#   2.  Applies pull secret for nvcr.io
#   3.  Links pull secret to default and pipeline service accounts
#   4.  Applies runtime env secret (workbench-runtime-secret)
#   5.  Applies RBAC — ServiceAccount + SCC ClusterRole + RoleBinding
#   6.  Applies ImageStream (registers workbench image with RHOAI dashboard)
#   7.  Applies BuildConfig and triggers the first image build
#   8.  Waits for the build to complete and ImageStream tag to appear
#   9.  Applies DataSciencePipelinesApplication (pipeline server)
#   10. Applies Notebook workbench
#   11. Prints routes and next steps

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — override any of these via environment variables
# ---------------------------------------------------------------------------
NAMESPACE="${NAMESPACE:-${OPENSHIFT_NAMESPACE:-}}"
SECRETS_DIR="${SECRETS_DIR:-openshift/secrets}"
MANIFESTS_DIR="${MANIFESTS_DIR:-openshift}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { echo "[INFO]  $*"; }
warn()  { echo "[WARN]  $*" >&2; }
die()   { echo "[ERROR] $*" >&2; exit 1; }

require_cmd() {
    command -v "$1" &>/dev/null || die "Required command not found: $1"
}

require_file() {
    [[ -f "$1" ]] || die "Required file not found: $1 — copy the matching .template.* and populate it"
}

oc_apply() {
    local file="$1"
    info "Applying $file"
    oc apply -f "$file" -n "$NAMESPACE"
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
require_cmd oc
require_cmd sed

[[ -n "$NAMESPACE" ]] || die "NAMESPACE is not set. Source openshift/secrets/cluster-credentials.env or set NAMESPACE=<your-project>"

info "Target namespace: $NAMESPACE"

# Verify oc is logged in
oc whoami &>/dev/null || die "oc is not logged in. Run: oc login --server=<url> --token=<token>"

# Verify required populated secret files exist
require_file "$SECRETS_DIR/registry-credentials.yaml"
require_file "$SECRETS_DIR/workbench-secret.yaml"

# ---------------------------------------------------------------------------
# Step 1: Create the OpenShift project
# ---------------------------------------------------------------------------
info "Creating project $NAMESPACE (oc new-project)..."
if oc get project "$NAMESPACE" &>/dev/null; then
    info "Project $NAMESPACE already exists — skipping creation"
    oc project "$NAMESPACE"
else
    oc new-project "$NAMESPACE" \
        --display-name="NeMo Transaction Foundation Model" \
        --description="Westpac NeMo decoder foundation model — training, evaluation, and serving"
fi

# Label the namespace so it appears as a managed project in the RHOAI dashboard.
# Without this label the Pipelines, Workbenches, and Models tabs are not shown.
info "Labelling namespace for RHOAI dashboard visibility..."
oc label namespace "$NAMESPACE" opendatahub.io/dashboard=true --overwrite

# ---------------------------------------------------------------------------
# Step 2: Pull secret for nvcr.io
# ---------------------------------------------------------------------------
info "Applying registry pull secret..."
oc apply -f "$SECRETS_DIR/registry-credentials.yaml" -n "$NAMESPACE"

# Step 3: Link pull secret to SAs so pods can pull from nvcr.io without
# explicitly adding imagePullSecrets to every pod spec.
info "Linking pull secret to default and builder service accounts..."
oc secrets link default registry-pull-secret --for=pull -n "$NAMESPACE" || true
oc secrets link builder registry-pull-secret -n "$NAMESPACE" || true

# ---------------------------------------------------------------------------
# Step 4: Runtime env secret (referenced by Notebook and PyTorchJob envFrom)
# ---------------------------------------------------------------------------
info "Applying workbench runtime secret..."
oc apply -f "$SECRETS_DIR/workbench-secret.yaml" -n "$NAMESPACE"

# ---------------------------------------------------------------------------
# Step 5: RBAC — ServiceAccount + SCC grant (requires cluster-admin)
# ---------------------------------------------------------------------------
info "Applying RBAC (ClusterRole, RoleBinding, ServiceAccount)..."
info "Note: ClusterRole requires cluster-admin — contact your cluster admin if this fails"
oc apply -f "$MANIFESTS_DIR/rbac/service-accounts.yaml" -n "$NAMESPACE"

# ---------------------------------------------------------------------------
# Step 6: ImageStream — registers the workbench image with RHOAI dashboard
# ---------------------------------------------------------------------------
info "Applying ImageStream..."
oc apply -f "$MANIFESTS_DIR/notebook-image/imagestream.yaml" -n "$NAMESPACE"

# ---------------------------------------------------------------------------
# Step 7: BuildConfig + first build
# ---------------------------------------------------------------------------
info "Applying BuildConfig..."
oc apply -f "$MANIFESTS_DIR/notebook-image/buildconfig.yaml" -n "$NAMESPACE"

info "Triggering workbench image build..."
oc start-build nemo-tfm-workbench -n "$NAMESPACE" --follow || \
    warn "Build trigger failed or already running — check: oc get builds -n $NAMESPACE"

# ---------------------------------------------------------------------------
# Step 8: Wait for ImageStreamTag :latest to appear
# ---------------------------------------------------------------------------
info "Waiting for ImageStream tag 'latest' to be populated (timeout: 15 min)..."
BUILD_TIMEOUT=900  # seconds
ELAPSED=0
until oc get istag "nemo-tfm-workbench:latest" -n "$NAMESPACE" &>/dev/null; do
    if (( ELAPSED >= BUILD_TIMEOUT )); then
        die "Timed out waiting for ImageStreamTag nemo-tfm-workbench:latest. Check build logs: oc logs -f bc/nemo-tfm-workbench -n $NAMESPACE"
    fi
    sleep 15
    (( ELAPSED += 15 ))
    info "  ...still waiting (${ELAPSED}s elapsed)"
done
info "ImageStreamTag nemo-tfm-workbench:latest is ready"

# ---------------------------------------------------------------------------
# Step 9: DataSciencePipelinesApplication (pipeline server)
# ---------------------------------------------------------------------------
info "Verifying DSP operator CRD is installed..."
oc get crd datasciencepipelinesapplications.opendatahub.io &>/dev/null || \
    die "CRD datasciencepipelinesapplications.opendatahub.io not found — is the RHOAI operator installed and the DSP component enabled?"

info "Applying DataSciencePipelinesApplication..."
oc apply -f "$MANIFESTS_DIR/pipeline/dspa.yaml" -n "$NAMESPACE"

# ---------------------------------------------------------------------------
# Step 10: Notebook workbench
# ---------------------------------------------------------------------------
info "Verifying Notebook CRD is installed..."
oc get crd notebooks.kubeflow.org &>/dev/null || \
    die "CRD notebooks.kubeflow.org not found — is the RHOAI workbenches component enabled?"

info "Applying Notebook workbench..."
oc apply -f "$MANIFESTS_DIR/workbench/notebook.yaml" -n "$NAMESPACE"

# ---------------------------------------------------------------------------
# Done — print status and next steps
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================="
echo " Bootstrap complete for project: $NAMESPACE"
echo "======================================================================="
echo ""
echo "Check workbench status:"
echo "  oc get notebook nemo-tfm-workbench -n $NAMESPACE"
echo ""
echo "Get workbench URL:"
echo "  oc get route -l app=nemo-tfm-workbench -n $NAMESPACE"
echo ""
echo "Run a training job (after filling in pytorchjob.yaml placeholders):"
echo "  oc apply -f openshift/training/pytorchjob.yaml -n $NAMESPACE"
echo "  oc get pytorchjob nemo-tfm-pretrain -n $NAMESPACE"
echo ""
echo "Deploy model serving (after training is complete):"
echo "  oc apply -f openshift/serving/serving-runtime.yaml -n $NAMESPACE"
echo "  oc apply -f openshift/serving/inference-service.yaml -n $NAMESPACE"
echo "  oc get inferenceservice nemo-tfm -n $NAMESPACE"
echo ""
echo "Get inference service route:"
echo "  oc get route -l serving.kserve.io/inferenceservice=nemo-tfm -n $NAMESPACE"
echo "======================================================================="
