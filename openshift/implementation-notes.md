# OpenShift AI Integration — Implementation Notes

## Why operator-managed Custom Resources, not raw Kubernetes manifests

### What we are NOT doing

Hand-rolling Pod, Deployment, or Job manifests and manually wiring:
- `securityContext` / `runAsUser` / `fsGroup`
- `ServiceAccount` + `ClusterRoleBinding`
- `Service` + `Route` + TLS termination
- Liveness / readiness probes
- PVC mounts and storage class selection
- Image pull secrets on the pod spec

This is fragile, not upgrade-safe, and re-invents what the operators already
provide — correctly, and with Red Hat's support contract behind them.

### What we ARE doing

Declare intent via operator Custom Resources. The operator controllers
reconcile the intent into the correct lower-level Kubernetes objects,
applying platform-appropriate defaults for security context, RBAC, probes,
networking, and storage.

| Property | Raw manifests | Operator CRs |
|----------|--------------|--------------|
| **Red Hat supported** | No — you own every field | Yes — operator defaults carry the support contract |
| **Upgrade-safe** | Manual rework on every RHOAI/OCP upgrade | Operator reconciles to new platform defaults automatically |
| **RBAC-clean** | Must create SA, RoleBinding, SCC manually | Operator manages its own SA and SCC grants |
| **Security context** | Must match node SCC — breaks across clusters | Operator applies `restricted-v2` / `anyuid` as appropriate per platform |
| **GitOps-friendly** | Large, cluster-specific manifests in git | Small, intent-level CRs; cluster detail stays in operator config |
| **Compliance-friendly** | Auditors must review your pod spec | Auditors review the operator, which is in the platform compliance scope |

---

## Custom Resource inventory — RHOAI 3.3

All CRs below are **namespace-scoped** unless noted. Apply to your project
namespace with `oc apply -f <file> -n <namespace>`.

> **API versions verified against official Red Hat documentation for RHOAI
> 3.3.** Sources:
> - [RHOAI 3.3 Distributed Workloads](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.3/html/working_with_distributed_workloads/running-kfto-based-distributed-training-workloads_distributed-workloads)
> - [RHOAI 3.0 Creating a Workbench (CRD)](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.0/html/creating_a_workbench/api-workbench-creating_api-workbench)
> - [RHOAI 3.0 Custom Image via ImageStream CRD](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.0/html/creating_a_workbench/api-custom-image-creating_api-workbench)
> - [RHOAI 3.3 Deploying Models (KServe RawDeployment)](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.3/html-single/deploying_models/index)
> - [RHOAI DSP 2.0 migration guide](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_cloud_service/1/html/working_with_data_science_pipelines/enabling-data-science-pipelines-2_ds-pipelines)

---

### Cluster-level CRs (admin applies once, not per project)

These are managed by the cluster admin via OperatorHub and are not
included in this repo. Listed here for context.

| CR | apiVersion | Scope | Purpose |
|----|-----------|-------|---------|
| `DSCInitialization` | `dscinitialization.opendatahub.io/v2` | Cluster | Bootstrap RHOAI operator; sets application namespace |
| `DataScienceCluster` | `datasciencecluster.opendatahub.io/v2` | Cluster | Enables/disables RHOAI components (workbenches, kserve, pipelines, etc.) |

---

### 1. Custom workbench image — `ImageStream`

**Operator:** RHOAI / OpenShift Image Registry (built-in)
**apiVersion:** `image.openshift.io/v1`
**Kind:** `ImageStream`
**File:** `notebook-image/imagestream.yaml`

There is **no `NotebookImage` CR**. Custom notebook images are registered
with the RHOAI dashboard by creating an `ImageStream` with the label
`opendatahub.io/notebook-image: "true"`. The RHOAI dashboard controller
watches for ImageStreams with this label and makes them selectable in the
Workbench creation UI.

The `ImageStream` must be in the **same namespace** as the workbench
(your Data Science Project namespace).

---

### 2. Workbench — `Notebook`

**Operator:** Kubeflow Notebook Controller (bundled with RHOAI workbenches component)
**apiVersion:** `kubeflow.org/v1`
**Kind:** `Notebook`
**File:** `workbench/notebook.yaml`

This is the RHOAI Workbench CR. The Kubeflow Notebook Controller
reconciles it into a StatefulSet, Service, and Route. **Do not write the
pod spec by hand** — security context, service account, OAuth proxy
sidecar, probes, and the Route are all injected by the controller.

Key annotations the controller uses:
- `notebooks.opendatahub.io/inject-oauth: 'true'` — injects the oauth-proxy sidecar
- `notebooks.opendatahub.io/last-image-selection` — tracks selected image tag
- `opendatahub.io/image-display-name` — display name for the dashboard
- `opendatahub.io/username` — owner identity
- `notebooks.opendatahub.io/oauth-logout-url` — dashboard logout redirect

---

### 3. Distributed training — `PyTorchJob`

**Operator:** Kubeflow Training Operator (KFTO), enabled as a component of RHOAI
**apiVersion:** `kubeflow.org/v1`
**Kind:** `PyTorchJob`
**File:** `training/pytorchjob.yaml`

> **Note on KFTO v2 (`TrainJob`):** The `TrainJob` CR
> (`trainer.kubeflow.org/v1alpha1`) is the upstream KFTO v2 API but is
> **not documented or supported in RHOAI 3.3**. `PyTorchJob` (`kubeflow.org/v1`)
> is the supported distributed training CR. Verify with
> `oc get crd pytorchjobs.kubeflow.org` before applying.

Spec structure:
- `pytorchReplicaSpecs.Master` — always `replicas: 1`; the rank-0 process
- `pytorchReplicaSpecs.Worker` — set `replicas` to number of additional nodes
- Training script either in a ConfigMap volume or baked into the container image
- GPU resources declared as `nvidia.com/gpu` in `resources.limits`
- Uses `torchrun` as the container command (NCCL backend for multi-GPU)

---

### 4. Pipeline server — `DataSciencePipelinesApplication`

**Operator:** Data Science Pipelines Operator (DSPO), enabled as a component of RHOAI
**apiVersion:** `datasciencepipelinesapplications.opendatahub.io/v1alpha1`
**Kind:** `DataSciencePipelinesApplication`
**File:** `pipeline/dspa.yaml`

Applying this CR provisions the full KFP 2.x pipeline server stack in your
namespace: API server, persistence agent, scheduled workflow controller,
and the Kubeflow Pipelines UI. Requires S3-compatible object storage
(provided via a Kubernetes Secret referenced in the spec).

> **RHOAI 3.3 uses Kubeflow Pipelines 2.5.0 (DSP 2.0).** The `v1alpha1`
> apiVersion is current for RHOAI 3.3. The DSP 1.0 `v1alpha1` schema is
> different — if migrating from DSP 1.0, follow the
> [Red Hat migration guide](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_cloud_service/1/html/working_with_data_science_pipelines/enabling-data-science-pipelines-2_ds-pipelines).

---

### 5. Model serving — `ServingRuntime` + `InferenceService`

**Operator:** KServe v0.15 (managed by RHOAI operator, enabled via `DataScienceCluster`)
**RHOAI 3.3 serving mode:** KServe **RawDeployment** (not Serverless/Knative)

> **These manifests are NOT in the ArgoCD sync path.**
> They live in `openshift/serving/` (not `openshift/gitops/serving/`) and
> are applied manually once a trained model checkpoint is available.
> See "Why serving is excluded from GitOps" below.

#### ServingRuntime
**apiVersion:** `serving.kserve.io/v1alpha1`
**Kind:** `ServingRuntime`
**File:** `serving/serving-runtime.yaml`

Defines the model server container template (e.g. vLLM, Triton). RHOAI
ships several built-in runtimes. Define a custom `ServingRuntime` only if
none of the built-in runtimes fit.

#### InferenceService
**apiVersion:** `serving.kserve.io/v1beta1`
**Kind:** `InferenceService`
**File:** `serving/inference-service.yaml`

Declares a model endpoint. KServe reconciles this into a Deployment,
Service, and Route. The model storage location (S3, PVC, or URI) is
declared in `spec.predictor`. GPU resources are declared in
`spec.predictor.model.resources`.

> **GPU type consistency:** If `nvidia.com/gpu` is set in the
> `ServingRuntime`, do not also set it in the `InferenceService` predictor
> — the operator merges them and conflicting types cause errors.

---

## What the Workbench Dockerfile does (and does not do)

```dockerfile
FROM nvcr.io/nvidia/nemo:25.09.01

# Add only this repo's additional Python requirements on top of the NeMo base.
COPY requirements.txt /opt/app-root/src/requirements.txt
RUN pip install --no-cache-dir -r /opt/app-root/src/requirements.txt
```

The Dockerfile does **not**:
- Set `USER`, `UID`, or `runAsUser` — the Notebook controller injects these from the namespace SCC
- Expose ports — the Notebook controller adds the Jupyter and OAuth proxy ports
- Configure networking or ingress — the controller creates the Route
- Mount secrets or ConfigMaps — declared in the `Notebook` CR `envFrom` / `volumeMounts`

---

## Directory layout

```
openshift/
├── implementation-notes.md          ← this file
├── notebook-image/
│   ├── Dockerfile                   ← minimal NeMo-based workbench image
│   ├── imagestream.yaml             ← ImageStream (registers image with RHOAI dashboard)
│   └── buildconfig.yaml             ← BuildConfig (build.openshift.io/v1) — Docker strategy
├── rbac/
│   └── service-accounts.yaml        ← SA + ClusterRole + RoleBinding for anyuid SCC
├── workbench/
│   └── notebook.yaml                ← Notebook CR (kubeflow.org/v1)
├── training/
│   └── pytorchjob.yaml              ← PyTorchJob CR (kubeflow.org/v1) — KFTO v1
├── pipeline/
│   └── dspa.yaml                    ← DataSciencePipelinesApplication CR
├── serving/
│   ├── serving-runtime.yaml         ← ServingRuntime CR (serving.kserve.io/v1alpha1)
│   └── inference-service.yaml       ← InferenceService CR (serving.kserve.io/v1beta1)
├── templates/
│   └── nemo-tfm-template.yaml       ← OpenShift Template (template.openshift.io/v1)
├── scripts/
│   └── bootstrap-project.sh         ← one-shot oc-native bootstrap script
└── secrets/
    ├── README.md
    ├── cluster-credentials.template.env
    ├── ai-platform.template.env
    ├── registry-credentials.template.yaml   ← pull secret for ImageStream
    └── workbench-secret.template.yaml       ← runtime env referenced by Notebook CR
```

---

## Apply order

```bash
# 0. Verify required CRDs are present (operator must be installed first)
oc get crd notebooks.kubeflow.org
oc get crd pytorchjobs.kubeflow.org
oc get crd datasciencepipelinesapplications.opendatahub.io
oc get crd inferenceservices.serving.kserve.io

# 1. Create the project (oc new-project, not kubectl create namespace)
oc new-project <your-project> --display-name="NeMo Transaction Foundation Model"

# 2. Pull secret — for nvcr.io (build + pod pulls)
oc apply -f openshift/secrets/registry-credentials.yaml -n <your-project>
oc secrets link default registry-pull-secret --for=pull -n <your-project>
oc secrets link builder registry-pull-secret -n <your-project>

# 3. Runtime secret — env vars the Notebook CR injects into the workbench pod
oc apply -f openshift/secrets/workbench-secret.yaml -n <your-project>

# 4. RBAC — SA + SCC ClusterRole + RoleBinding (requires cluster-admin)
oc apply -f openshift/rbac/service-accounts.yaml -n <your-project>

# 5. Register custom workbench image with RHOAI dashboard
oc apply -f openshift/notebook-image/imagestream.yaml -n <your-project>

# 6. Build the workbench image inside the cluster
oc apply -f openshift/notebook-image/buildconfig.yaml -n <your-project>
oc start-build nemo-tfm-workbench -n <your-project> --follow

# 7. Provision the KFP pipeline server
oc apply -f openshift/pipeline/dspa.yaml -n <your-project>

# 8. Launch the workbench (Notebook controller reconciles within ~30s)
oc apply -f openshift/workbench/notebook.yaml -n <your-project>

# 9. Run a training job
oc apply -f openshift/training/pytorchjob.yaml -n <your-project>

# 10. Deploy model serving
oc apply -f openshift/serving/serving-runtime.yaml -n <your-project>
oc apply -f openshift/serving/inference-service.yaml -n <your-project>
```

Or use the one-shot script:
```bash
source openshift/secrets/cluster-credentials.env
NAMESPACE=<your-project> bash openshift/scripts/bootstrap-project.sh
```

---

## OpenShift-native primitives

These additions use the full OpenShift-native stack beyond the RHOAI operator CRs.

### Projects over Namespaces

Use `oc new-project` not `kubectl create namespace`. An OpenShift Project is a
Kubernetes namespace extended with display metadata, default RBAC role bindings,
and RHOAI dashboard registration.

### BuildConfig — image build inside the cluster

`BuildConfig` (`build.openshift.io/v1`) runs a Docker build inside the cluster.
**Docker strategy is correct** for the NeMo Dockerfile — S2I is not appropriate
because the NVIDIA NeMo base image is not S2I-compatible (no `assemble` script).

The builder SA automatically has push rights to the integrated registry. The
`dockerStrategy.pullSecret` field grants pull rights to `nvcr.io` for the FROM
instruction.

### ImageStream — stable internal image references

`ImageStream` (`image.openshift.io/v1`) decouples workbench pods from specific
image digests. The `opendatahub.io/notebook-image: "true"` label registers the
stream with the RHOAI dashboard automatically.

### RBAC + SCC for training pods

The NVIDIA NeMo container runs as root (uid 0). OpenShift's default
`restricted-v2` SCC does not permit root. The pattern:
1. `ClusterRole` — `use` verb on `securitycontextconstraints/anyuid`
2. `RoleBinding` — namespace-scoped; binds ClusterRole to `nemo-tfm-training` SA
3. `PyTorchJob` pod spec — `serviceAccountName: nemo-tfm-training`

The Notebook controller manages its own SA for workbench pods — do NOT add
`serviceAccountName` to the Notebook CR.

Do NOT modify default SCCs. Red Hat docs:
> "Customizing the default SCCs can lead to issues when some of the platform
> pods deploy or OpenShift Container Platform is upgraded."

### Template — parameterised deployment

`Template` (`template.openshift.io/v1`) is the native OpenShift parameterisation
mechanism. Parameters: `NAMESPACE`, `GIT_REPO_URL`, `GIT_REF`, `OPENSHIFT_USERNAME`,
`DASHBOARD_URL`, `S3_ENDPOINT_HOST`, `PIPELINE_BUCKET`, `TRAINING_REPLICAS`.

```bash
oc process -f openshift/templates/nemo-tfm-template.yaml \
  -p NAMESPACE=nemo-tfm \
  -p GIT_REPO_URL=https://... \
| oc apply -f -
```

---

## Why serving is excluded from GitOps auto-sync

The `ServingRuntime` and `InferenceService` manifests live in
`openshift/serving/` — outside the ArgoCD sync path (`openshift/gitops/`) —
and are applied imperatively once a trained model checkpoint is ready.

### The problem with including serving in GitOps

KServe reconciles the `InferenceService` into a running `Deployment`
immediately on sync. The predictor pod starts, mounts the model PVC, and
tries to load the model. If the PVC is empty — which it always is before
the training pipeline has run — the predictor crash-loops indefinitely.

This causes the ArgoCD application to report `Degraded` health
permanently, masking real infrastructure failures in the workbench, image
build, or pipeline stack. A permanently red dashboard is worse than no
dashboard entry.

### The principle

GitOps should only manage resources it can fully reconcile to a healthy
state from a clean git clone. A trained model checkpoint is a runtime
artifact produced by a GPU training job — it is not a git-tracked input
and cannot be produced by the sync process itself. Including serving in the
sync path conflates infrastructure state (what git declares) with runtime
state (what training produced).

### The approach

- Infrastructure and workbench resources are in `openshift/gitops/` and
  reconciled to `Synced / Healthy` by ArgoCD with no external dependencies.
- Serving manifests are retained in `openshift/serving/` as reference
  templates. The deploying team applies them imperatively once their model
  checkpoint is in place:

  ```bash
  oc apply -f openshift/serving/model-output-pvc.yaml  -n nemo-tfm
  oc apply -f openshift/serving/serving-runtime.yaml   -n nemo-tfm
  oc apply -f openshift/serving/inference-service.yaml -n nemo-tfm
  ```

- Training (`openshift/training/pytorchjob.yaml`) follows the same
  principle — it is always manual, never in the GitOps sync path.
