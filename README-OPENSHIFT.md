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

> **You do not need the `aws_ocp` repository.**
> That repo documents how this project's demo cluster was provisioned on AWS.
> If you already have an OpenShift cluster with RHOAI and a GPU node, start
> here. If you need to build a cluster from scratch on AWS, see
> [aws_ocp](https://github.com/robbybrodie/aws_ocp) for reference.

---

### What you need

#### OpenShift cluster

| Requirement | Minimum | Tested on |
|---|---|---|
| OpenShift Container Platform | 4.19 | 4.21.15 |
| Infrastructure | Any (AWS, GCP, Azure, bare-metal, on-prem) | AWS `g6.8xlarge` (NVIDIA L4) |
| `oc` CLI | Matching cluster version | — |

The cluster can run anywhere OpenShift runs. Cloud, on-prem, or bare-metal
are all fine as long as the GPU and operator requirements below are met.

#### GPU node

| Requirement | Detail |
|---|---|
| GPU | **≥ 2 GPUs on a single node** — the workbench holds 1 GPU; sequential pipeline GPU steps each need 1 GPU; both must coexist on the same node because the shared PVC is EBS ReadWriteOnce (one node only). Recommended: `g6.12xlarge` (4 × L4 24 GB) for demo; `p4d.24xlarge` (8 × A100 40 GB) or `p5.48xlarge` (8 × H100 80 GB) for production |
| CUDA | 12.x (provided by the NeMo container — no host CUDA install needed) |
| Root volume | **300 GB minimum** — the NeMo base image is ~25 GB; the default 120 GB disk causes eviction during the BuildConfig image build |
| Count | 1 GPU node (multi-GPU). Multi-node is not required — the pipeline steps are sequential |

> **Why a single multi-GPU node?**
> The workbench PVC uses `accessModes: ReadWriteOnce` (EBS), which attaches to
> exactly one EC2 instance. The workbench and all pipeline steps declare
> `nodeSelector: nvidia.com/gpu.present: "true"`, so they all land on the same
> GPU node. With ≥ 2 GPUs on that node, the workbench (1 GPU) and each
> sequential pipeline GPU step (1 GPU) can coexist without contention.
>
> **MIG alternative (A100 / H100 only):**
> On A100 or H100 nodes, you can use MIG partitioning instead of a multi-GPU
> node. Enable MIG mode via the GPU Operator, configure a profile (e.g.
> `3g.40gb` on an A100 40 GB — gives 2 independent slices), then substitute the
> resource name in `notebook.yaml` and the pipeline accelerator calls:
> ```yaml
> # notebook.yaml resources — replace nvidia.com/gpu with MIG profile:
> nvidia.com/mig-3g.40gb: "1"
> # pipeline/nemo_tfm_pipeline.py accelerator calls — e.g.:
> s2.set_accelerator_type("nvidia.com/mig-3g.40gb").set_accelerator_limit(1)
> ```
> The `nodeSelector: nvidia.com/gpu.present: "true"` remains unchanged —
> the GPU Operator sets this label on any GPU node, MIG or otherwise.

#### RHOAI and operators (cluster-admin, one-time)

| Component | Version | Notes |
|---|---|---|
| Red Hat OpenShift AI | 3.x (`stable` channel) | Tested on operator 2.25.6 / RHOAI 3.3 |
| NVIDIA GPU Operator | v24.9+ | Channel `v24.9` |
| Node Feature Discovery (NFD) | Stable | Labels GPU nodes; GPU Operator depends on it |
| OpenShift Service Mesh (OSSM) | 2.x | Required by KServe (model serving only — skip if not serving) |
| OpenShift Serverless | 1.x | Required by KServe (model serving only — skip if not serving) |

The `DataScienceCluster` CR must have these components set to `Managed`:

```yaml
# Required for this project:
workbenches:          Managed   # JupyterLab workbench
datasciencepipelines: Managed   # KFP 2.x pipeline server
trainingoperator:     Managed   # PyTorchJob distributed training

# Required only for model serving (Step 7):
kserve:               Managed
serving:              Managed
```

#### S3-compatible object storage

The KFP pipeline server (`DataSciencePipelinesApplication`) stores pipeline
run artifacts in S3. You need:

- An S3-compatible bucket (AWS S3, MinIO, or OpenShift Data Foundation all work)
- An access key / secret key with read-write access to that bucket
- The bucket endpoint, region, and name — these go into `openshift/secrets/workbench-secret.yaml`

The bucket does **not** need to be on AWS. MinIO running anywhere works.

#### NGC API key

Required to pull `nvcr.io/nvidia/nemo:25.09.01` from NVIDIA's container
registry. Obtain one from [ngc.nvidia.com](https://ngc.nvidia.com) (free
account). This goes into `openshift/secrets/registry-credentials.yaml`.

---

### Credentials summary

Before running any steps, collect all of the following. The table shows
where each value comes from and where it is used.

| Credential | Where to get it | Used in |
|---|---|---|
| **OpenShift API URL** | Cluster console → top-right menu → "Copy login command" | `oc login`, `cluster-credentials.env` |
| **OpenShift token** | Same "Copy login command" page (or `oc whoami --show-token` after login) | `oc login`, `cluster-credentials.env` |
| **Cluster-admin role** | Required for Step 3 (RBAC/SCC) only — not needed for day-to-day use | `oc apply -f openshift/rbac/` |
| **RHOAI dashboard login** | Same OpenShift username and password — RHOAI uses OpenShift OAuth, no separate account | RHOAI dashboard UI, workbench launch |
| **NGC API key** | [ngc.nvidia.com](https://ngc.nvidia.com) → free account → API Keys | `openshift/secrets/registry-credentials.yaml` |
| **S3 access key** | Your AWS IAM console, MinIO admin, or ODF admin | `openshift/secrets/workbench-secret.yaml` |
| **S3 secret key** | Same as above | `openshift/secrets/workbench-secret.yaml` |
| **S3 bucket name** | Create a bucket in your S3-compatible store; note the name | `openshift/secrets/workbench-secret.yaml`, `openshift/pipeline/dspa.yaml` |
| **S3 endpoint** | e.g. `s3.us-west-2.amazonaws.com` (AWS) or your MinIO URL | `openshift/pipeline/dspa.yaml` |

**Getting your OpenShift login token:**

```bash
# Option A — from the console UI
# 1. Open the OpenShift console in your browser
# 2. Click your username (top right) → "Copy login command"
# 3. Click "Display Token" — copy the oc login ... command

# Option B — if already logged in
oc whoami --show-token

# Tokens expire (typically 24h). Refresh by repeating the above.
```

**Generating the NGC pull secret** (required for `registry-credentials.yaml`):

```bash
NGC_API_KEY=<your-key>
AUTH=$(echo -n "\$oauthtoken:${NGC_API_KEY}" | base64)
echo "{\"auths\":{\"nvcr.io\":{\"auth\":\"${AUTH}\"}}}" | base64
# Paste the output as the .dockerconfigjson value in registry-credentials.yaml
```

> **Note:** You do NOT need a separate RHOAI account or API key.
> RHOAI inherits OpenShift's identity provider — log into the dashboard
> with the same credentials you use for `oc login`.

---

### Verify your cluster is ready

Run these checks before applying any manifests:

```bash
# RHOAI operator installed
oc get csv -n redhat-ods-operator | grep rhods-operator

# All required CRDs present
oc get crd notebooks.kubeflow.org
oc get crd pytorchjobs.kubeflow.org
oc get crd datasciencepipelinesapplications.opendatahub.io

# GPU operator running and GPU nodes labelled
oc get nodes -l nvidia.com/gpu.present=true
oc get node -l nvidia.com/gpu.present=true \
  -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}'
# Expected: 1 (or more per node)

# DataScienceCluster components ready
oc get datasciencecluster -o jsonpath='{range .items[0].status.conditions[*]}{.type}{"\t"}{.status}{"\n"}{end}'
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

## GitOps deploy (preferred)

If OpenShift GitOps (ArgoCD) is available on your cluster, a single `oc apply`
deploys and reconciles the full project continuously from Git.

### Prerequisites

```bash
# Install OpenShift GitOps operator (cluster-admin, once per cluster)
oc apply -f openshift/gitops-install/gitops-subscription.yaml

# Wait for ArgoCD pods to be Running (~3 min)
oc get pods -n openshift-gitops

# Get the ArgoCD UI URL
oc get route openshift-gitops-server -n openshift-gitops -o jsonpath='{.spec.host}'
```

### Deploy

```bash
# Single command deploys everything in sync-wave order
oc apply -f openshift/argocd/application.yaml

# Monitor sync status
oc get application nemo-tfm -n openshift-gitops
```

ArgoCD applies resources in this order (sync-wave):

| Wave | Resource | Notes |
|------|----------|-------|
| -1 | `namespace.yaml` | Creates the `nemo-tfm` project |
| 0 | `secrets/*.sealed.yaml`, `rbac/service-accounts.yaml`, `rbac/sa-pull-secrets.yaml` | Secrets decrypted by Sealed Secrets controller; RBAC wired up |
| 1 | `notebook-image/imagestream.yaml` | Registers image with RHOAI dashboard |
| 2 | `notebook-image/buildconfig.yaml` | ConfigChange trigger fires the first build automatically |
| 3 | `pipeline/dspa.yaml` | KFP 2.x pipeline server |
| 4 | `workbench/notebook.yaml` + PVC | Workbench (retries until build completes) |
| 5 | `serving/serving-runtime.yaml` | vLLM ServingRuntime |
| 6 | `serving/inference-service.yaml` | KServe model endpoint |

### Re-sealing secrets

Secrets are encrypted with the cluster's Sealed Secrets public key. If you
rotate credentials or deploy to a new cluster, re-seal from the local plain
secrets (which are gitignored and never committed):

```bash
# Re-seal both secrets for the nemo-tfm namespace
kubeseal --scope namespace-wide --namespace nemo-tfm --format yaml \
  < openshift/secrets/registry-credentials.yaml \
  > openshift/gitops/secrets/registry-pull-secret.sealed.yaml

kubeseal --scope namespace-wide --namespace nemo-tfm --format yaml \
  < openshift/secrets/workbench-secret.yaml \
  > openshift/gitops/secrets/workbench-runtime-secret.sealed.yaml

# The sealed files are safe to commit — they are encrypted with the cluster
# public key and can only be decrypted inside the nemo-tfm namespace.
git add openshift/gitops/secrets/*.sealed.yaml
git commit -m "Rotate sealed secrets"
git push
# ArgoCD picks up the new sealed secrets and reconciles within ~30s
```

> **Production note:** For multi-cluster or key-rotation scenarios, consider
> External Secrets Operator (ESO) backed by HashiCorp Vault or AWS Secrets
> Manager instead of Sealed Secrets. The GitOps structure here is compatible
> with ESO: replace `*.sealed.yaml` with `ExternalSecret` CRs pointing to your
> secrets store.

---

## Automated bootstrap

The bootstrap script performs all of the above imperatively — use it when
GitOps is not available or for local development.

> **Prefer GitOps.** See the section above for the recommended deploy path.

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
├── gitops/                          ← ArgoCD-managed manifests (canonical)
│   ├── namespace.yaml               ← Namespace CR (wave -1)
│   ├── secrets/
│   │   ├── registry-pull-secret.sealed.yaml    ← SealedSecret (wave 0)
│   │   └── workbench-runtime-secret.sealed.yaml ← SealedSecret (wave 0)
│   ├── rbac/
│   │   ├── service-accounts.yaml    ← SA + ClusterRole + RoleBinding (wave 0)
│   │   └── sa-pull-secrets.yaml     ← SA imagePullSecrets patches (wave 0)
│   ├── notebook-image/
│   │   ├── imagestream.yaml         ← ImageStream (wave 1)
│   │   └── buildconfig.yaml         ← BuildConfig (wave 2)
│   ├── pipeline/
│   │   └── dspa.yaml                ← DataSciencePipelinesApplication (wave 3)
│   ├── workbench/
│   │   └── notebook.yaml            ← Notebook CR + PVC (wave 4)
│   └── serving/
│       ├── serving-runtime.yaml     ← ServingRuntime (wave 5)
│       └── inference-service.yaml   ← InferenceService (wave 6)
├── argocd/
│   └── application.yaml             ← ArgoCD Application CR
├── gitops-install/
│   └── gitops-subscription.yaml     ← OpenShift GitOps operator Subscription
├── notebook-image/
│   └── Dockerfile                   ← FROM nvcr.io/nvidia/nemo:25.09.01
├── training/
│   └── pytorchjob.yaml              ← PyTorchJob CR (manual — not GitOps)
├── templates/
│   └── nemo-tfm-template.yaml       ← OpenShift Template (oc process)
├── scripts/
│   └── bootstrap-project.sh         ← imperative fallback bootstrap
└── secrets/
    ├── README.md
    ├── cluster-credentials.template.env
    ├── ai-platform.template.env
    ├── registry-credentials.template.yaml   ← gitignored when populated
    └── workbench-secret.template.yaml       ← gitignored when populated
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
