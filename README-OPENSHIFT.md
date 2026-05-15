# Deploying on Red Hat OpenShift AI

This guide covers the full deployment of the NeMo Transaction Foundation Model
on **Red Hat OpenShift AI (RHOAI) 3.3** using OpenShift-native primitives:
`oc` CLI, `BuildConfig`, `ImageStream`, RHOAI operator CRs, and OpenShift
`Template`.

All commands use `oc`, not `kubectl`. All resources live in an OpenShift
**project** (not a generic Kubernetes namespace).

---

## Architecture overview

```
Git repo
  └── openshift/notebook-image/Dockerfile
        │
        ▼
   BuildConfig (build.openshift.io/v1)
        │  Docker strategy — pulls nvcr.io base, installs requirements.txt
        ▼
   ImageStream (image.openshift.io/v1)
        │  nemo-tfm-workbench:latest — stored in OpenShift internal registry
        │  RHOAI dashboard watches ImageStreams with opendatahub.io/notebook-image="true"
        ▼
   Notebook CR (kubeflow.org/v1)            ← Kubeflow Notebook Controller
        │  StatefulSet + Route + OAuth proxy sidecar (injected by operator)
        ▼
   Workbench pod (JupyterLab on NeMo)
        │
        ├── PyTorchJob CR (kubeflow.org/v1)  ← KFTO training operator
        │     Master + Worker pods, GPU-backed, nemo-tfm-training SA
        │
        ├── DataSciencePipelinesApplication   ← DSPO (KFP 2.5)
        │     KFP API server + UI + MariaDB + S3 artifact storage
        │
        └── InferenceService CR (serving.kserve.io/v1beta1)
              KServe RawDeployment + ServingRuntime (vLLM GPU)
```

---

## Why OpenShift, not generic Kubernetes

| Concern | Generic Kubernetes approach | OpenShift approach used here |
|---------|----------------------------|------------------------------|
| **Image build** | External CI/CD (GitHub Actions, Jenkins) | `BuildConfig` — build runs inside the cluster, uses internal registry |
| **Image registry** | DockerHub or external registry, pull secrets on every pod | `ImageStream` — internal registry, automatic pull rights for project SAs |
| **Custom notebook images** | No standard mechanism | `ImageStream` with `opendatahub.io/notebook-image: "true"` label — RHOAI dashboard picks it up automatically |
| **Security context** | Hand-set `runAsUser`, `fsGroup`, `securityContext` on every pod spec | Operator CRs — Notebook controller injects appropriate SCC; KFTO training pods use dedicated SA with `anyuid` SCC grant |
| **Ingress / TLS** | `Ingress` + cert-manager + manual TLS | `Route` — created and managed by the RHOAI operator; TLS terminated by the OpenShift router |
| **Parameterised deploy** | Helm chart or Kustomize | `Template` (`template.openshift.io/v1`) — native to `oc`, no extra tooling |
| **Auth for workbenches** | Manual OAuth proxy setup | `notebooks.opendatahub.io/inject-oauth: 'true'` annotation — controller injects the OAuth proxy sidecar |
| **RBAC for SCC** | Not applicable | `ClusterRole` + `RoleBinding` with `use` verb on SCC — cluster-admin required once; developer workflow is project-scoped |

---

## Prerequisites

### Cluster-level (cluster-admin, one-time)

Full cluster setup — OpenShift install, operator subscriptions (NVIDIA GPU
Operator, NFD, OSSM, Serverless, RHOAI), GPU `ClusterPolicy`, and
`DataScienceCluster` configuration — is documented and scripted in the
companion **[aws_ocp](../aws_ocp)** repository. Complete that setup first.

The cluster must have the following in place before continuing here:

- RHOAI `DataScienceCluster` with `workbenches`, `kserve`,
  `datasciencepipelines`, and `trainingoperator` all `Managed`
- NVIDIA GPU Operator with NFD running and `ClusterPolicy` applied
- At least one GPU node with `nvidia.com/gpu` allocatable

Verify before proceeding:
```bash
# All four CRDs must be present
oc get crd notebooks.kubeflow.org
oc get crd pytorchjobs.kubeflow.org
oc get crd inferenceservices.serving.kserve.io
oc get crd | grep datasciencepipelinesapplications

# GPU must be allocatable
oc get node -l nvidia.com/gpu.present=true \
  -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}'
# Expected: 1 (or more)
```

### Tools

```bash
# OpenShift CLI — https://mirror.openshift.com/pub/openshift-v4/clients/ocp/latest/
oc version        # must be 4.19 or later

# Git LFS (for pulling the pretrained checkpoint)
git lfs install
git lfs pull
```

---

## Step 1 — Log in and create the project

```bash
# Log in (token from RHOAI dashboard → Copy login command)
oc login --server=https://<your-api-server>:6443 --token=<your-token>

# Create the project
oc new-project nemo-tfm \
  --display-name="NeMo Transaction Foundation Model" \
  --description="Decoder foundation model — training, evaluation, and serving"

# Confirm
oc project
```

OpenShift **projects** are namespaces with additional metadata and
default RBAC. Use `oc new-project` not `kubectl create namespace`.

---

## Step 2 — Populate credentials

Copy the template files and fill in real values:

```bash
cp openshift/secrets/cluster-credentials.template.env openshift/secrets/cluster-credentials.env
cp openshift/secrets/ai-platform.template.env       openshift/secrets/ai-platform.env
cp openshift/secrets/registry-credentials.template.yaml openshift/secrets/registry-credentials.yaml
cp openshift/secrets/workbench-secret.template.yaml    openshift/secrets/workbench-secret.yaml
```

Edit each file. The `.env` files and the populated `.yaml` files are
gitignored — they will never be committed. See `openshift/secrets/README.md`
for field-by-field instructions.

Apply the Kubernetes Secrets:

```bash
source openshift/secrets/cluster-credentials.env

oc apply -f openshift/secrets/registry-credentials.yaml -n nemo-tfm
oc apply -f openshift/secrets/workbench-secret.yaml     -n nemo-tfm

# Link the registry pull secret to the default SA (enables pod pulls without
# per-pod imagePullSecrets)
oc secrets link default registry-pull-secret --for=pull -n nemo-tfm
oc secrets link builder registry-pull-secret -n nemo-tfm
```

---

## Step 3 — Apply RBAC (requires cluster-admin)

The NVIDIA NeMo container runs as root (uid 0). OpenShift's default
`restricted-v2` SCC does not allow root. The RBAC file creates a dedicated
ServiceAccount for training pods and grants it the `anyuid` SCC
within the `nemo-tfm` project only.

```bash
oc apply -f openshift/rbac/service-accounts.yaml -n nemo-tfm
```

This creates:
- `ServiceAccount/nemo-tfm-training`
- `ClusterRole/nemo-tfm-anyuid-scc` — grants `use` on the `anyuid` SCC
- `RoleBinding/nemo-tfm-training-anyuid` — binds the ClusterRole to the SA in this project only

Verify:
```bash
oc adm policy who-can use scc anyuid -n nemo-tfm
```

> **Note on the workbench SA:** The Kubeflow Notebook Controller manages its
> own ServiceAccount for workbench pods. The `Notebook` CR spec must NOT
> include `serviceAccountName` — the operator injects it from platform defaults.

---

## Step 4 — Build the workbench image

OpenShift's `BuildConfig` builds the workbench image inside the cluster,
using the `Dockerfile` in this repo, and pushes the result to the integrated
registry tagged in the `nemo-tfm-workbench` ImageStream.

Edit `openshift/notebook-image/buildconfig.yaml` and set `spec.source.git.uri`
to your fork or mirror URL, then:

```bash
# Register the ImageStream (makes the image selectable in the RHOAI dashboard)
oc apply -f openshift/notebook-image/imagestream.yaml -n nemo-tfm

# Apply the BuildConfig and trigger the first build
oc apply -f openshift/notebook-image/buildconfig.yaml -n nemo-tfm
oc start-build nemo-tfm-workbench -n nemo-tfm --follow

# Verify the build succeeded and the tag was pushed
oc get builds -l buildconfig=nemo-tfm-workbench -n nemo-tfm
oc get istag nemo-tfm-workbench:latest -n nemo-tfm
```

Future builds trigger automatically on:
- `ConfigChange` — whenever the BuildConfig definition changes
- `ImageChange` — whenever the upstream image in the registry is updated

Manual rebuild:
```bash
oc start-build nemo-tfm-workbench -n nemo-tfm --follow
```

---

## Step 5 — Deploy via Template (recommended) or individual CRs

### Option A — OpenShift Template (recommended for repeatable deploys)

The `Template` wraps all deployment objects with named parameters.
`oc process` substitutes values and produces plain YAML for `oc apply`.

```bash
oc process -f openshift/templates/nemo-tfm-template.yaml \
  -p NAMESPACE=nemo-tfm \
  -p GIT_REPO_URL=https://github.com/<your-org>/<your-repo>.git \
  -p GIT_REF=main \
  -p OPENSHIFT_USERNAME=<your-oc-username> \
  -p DASHBOARD_URL=https://rhods-dashboard-redhat-ods-applications.apps.<cluster> \
  -p S3_ENDPOINT_HOST=s3.amazonaws.com \
  -p PIPELINE_BUCKET=my-kfp-bucket \
| oc apply -f -
```

Dry-run first to review what will be created:
```bash
oc process -f openshift/templates/nemo-tfm-template.yaml \
  -p NAMESPACE=nemo-tfm \
  ... \
| oc apply -f - --dry-run=client
```

### Option B — Individual CRs

```bash
# 1. Pipeline server (KFP 2.x)
oc apply -f openshift/pipeline/dspa.yaml -n nemo-tfm

# 2. Workbench
oc apply -f openshift/workbench/notebook.yaml -n nemo-tfm

# Verify the Notebook controller reconciled:
oc get notebook nemo-tfm-workbench -n nemo-tfm
oc get route -l app=nemo-tfm-workbench -n nemo-tfm
```

---

## Step 6 — Run a training job

Edit `openshift/training/pytorchjob.yaml`:
- Replace `<your-project-namespace>` with `nemo-tfm`
- Adjust `replicas` for Worker if using multi-node training
  (set to `total_nodes - 1`; `0` for single-node multi-GPU)

```bash
# Verify the Training Operator CRD is present
oc get crd pytorchjobs.kubeflow.org

# Apply the training job
oc apply -f openshift/training/pytorchjob.yaml -n nemo-tfm

# Monitor
oc get pytorchjob nemo-tfm-pretrain -n nemo-tfm
oc logs -f -l job-name=nemo-tfm-pretrain,replica-type=master -n nemo-tfm

# Delete when done (PyTorchJob does not self-delete)
oc delete pytorchjob nemo-tfm-pretrain -n nemo-tfm
```

The training job uses `serviceAccountName: nemo-tfm-training` (created in
Step 3) to satisfy the `anyuid` SCC requirement for root containers.

---

## Step 7 — Deploy model serving

After training completes, the checkpoint is in the `nemo-tfm-model-output` PVC.

```bash
# Custom ServingRuntime (vLLM GPU)
oc apply -f openshift/serving/serving-runtime.yaml -n nemo-tfm

# InferenceService (KServe RawDeployment)
oc apply -f openshift/serving/inference-service.yaml -n nemo-tfm

# Watch until READY=True
oc get inferenceservice nemo-tfm -n nemo-tfm -w

# Get the Route URL
oc get route -l serving.kserve.io/inferenceservice=nemo-tfm -n nemo-tfm
```

Test the OpenAI-compatible endpoint:
```bash
INFER_URL=$(oc get route -l serving.kserve.io/inferenceservice=nemo-tfm \
  -n nemo-tfm -o jsonpath='{.items[0].spec.host}')

curl -s "https://${INFER_URL}/v1/models" | python3 -m json.tool
```

---

## Automated bootstrap

The bootstrap script performs all of the above in a single run:

```bash
source openshift/secrets/cluster-credentials.env
NAMESPACE=nemo-tfm bash openshift/scripts/bootstrap-project.sh
```

The script will:
1. Create the project with `oc new-project`
2. Apply pull secret and link it to service accounts
3. Apply runtime secret
4. Apply RBAC
5. Apply ImageStream
6. Apply BuildConfig and trigger a build
7. Wait for `ImageStreamTag nemo-tfm-workbench:latest` to be ready
8. Apply DataSciencePipelinesApplication
9. Apply the Notebook workbench CR
10. Print routes and next steps

---

## Useful `oc` commands

```bash
# Project context
oc project nemo-tfm
oc status -n nemo-tfm

# Workbench
oc get notebook nemo-tfm-workbench -n nemo-tfm -o yaml
oc describe notebook nemo-tfm-workbench -n nemo-tfm
oc get route -l app=nemo-tfm-workbench -n nemo-tfm

# Build
oc get builds -l buildconfig=nemo-tfm-workbench -n nemo-tfm
oc logs -f bc/nemo-tfm-workbench -n nemo-tfm
oc get istag nemo-tfm-workbench:latest -n nemo-tfm

# Training
oc get pytorchjob nemo-tfm-pretrain -n nemo-tfm -o yaml
oc logs -f -l job-name=nemo-tfm-pretrain,replica-type=master -n nemo-tfm

# Pipeline server
oc get datasciencepipelinesapplication pipelines-definition -n nemo-tfm
oc get pods -l app=ds-pipeline -n nemo-tfm

# Serving
oc get inferenceservice nemo-tfm -n nemo-tfm
oc get servingruntimes -n nemo-tfm
oc get route -l serving.kserve.io/inferenceservice=nemo-tfm -n nemo-tfm

# RBAC / SCC diagnostics
oc adm policy who-can use scc anyuid -n nemo-tfm
oc get rolebindings -l app=nemo-tfm -n nemo-tfm
oc auth can-i use scc/anyuid --as=system:serviceaccount:nemo-tfm:nemo-tfm-training -n nemo-tfm

# Events (first stop for debugging)
oc get events -n nemo-tfm --sort-by=.lastTimestamp
```

---

## Directory reference

```
openshift/
├── implementation-notes.md        ← architectural decision record
├── notebook-image/
│   ├── Dockerfile                 ← FROM nvcr.io/nvidia/nemo:25.09.01
│   ├── imagestream.yaml           ← ImageStream (image.openshift.io/v1)
│   └── buildconfig.yaml           ← BuildConfig (build.openshift.io/v1)
├── rbac/
│   └── service-accounts.yaml      ← SA + ClusterRole + RoleBinding
├── workbench/
│   └── notebook.yaml              ← Notebook CR (kubeflow.org/v1)
├── training/
│   └── pytorchjob.yaml            ← PyTorchJob CR (kubeflow.org/v1)
├── pipeline/
│   └── dspa.yaml                  ← DataSciencePipelinesApplication
├── serving/
│   ├── serving-runtime.yaml       ← ServingRuntime (serving.kserve.io/v1alpha1)
│   └── inference-service.yaml     ← InferenceService (serving.kserve.io/v1beta1)
├── templates/
│   └── nemo-tfm-template.yaml     ← OpenShift Template (template.openshift.io/v1)
├── scripts/
│   └── bootstrap-project.sh       ← one-shot bootstrap using oc
└── secrets/
    ├── README.md
    ├── cluster-credentials.template.env
    ├── ai-platform.template.env
    ├── registry-credentials.template.yaml
    └── workbench-secret.template.yaml
```

---

## Troubleshooting

### Build fails with "ImagePullBackOff" on FROM nvcr.io/...
The `dockerStrategy.pullSecret` in the `BuildConfig` references
`registry-pull-secret`. Verify it exists and contains valid NGC credentials:
```bash
oc get secret registry-pull-secret -n nemo-tfm
oc secrets link builder registry-pull-secret -n nemo-tfm
```

### PyTorchJob pods stuck in "Pending" with SCC error
Verify the training SA has the anyuid SCC grant:
```bash
oc auth can-i use scc/anyuid \
  --as=system:serviceaccount:nemo-tfm:nemo-tfm-training -n nemo-tfm
```
If it returns `no`, re-apply `openshift/rbac/service-accounts.yaml` with
cluster-admin credentials.

### Notebook pod shows "0/1 nodes available: insufficient nvidia.com/gpu"
The cluster has no schedulable GPU nodes. Check:
```bash
oc get nodes -l nvidia.com/gpu.present=true
oc describe node <gpu-node> | grep -A5 "Allocatable"
```

### InferenceService stuck in "Not Ready"
```bash
oc describe inferenceservice nemo-tfm -n nemo-tfm
oc get pods -l serving.kserve.io/inferenceservice=nemo-tfm -n nemo-tfm
oc logs -l serving.kserve.io/inferenceservice=nemo-tfm -c kserve-container -n nemo-tfm
```
Ensure `nvidia.com/gpu` is declared in the `ServingRuntime` and NOT repeated
in the `InferenceService` predictor spec.
