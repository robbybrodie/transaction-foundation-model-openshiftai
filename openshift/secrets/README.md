# OpenShift Secrets — Team Instructions

This directory holds **template skeletons only**. No real credentials are
stored here. The `.template.*` files are tracked in git; the populated copies
they produce are gitignored.

---

## Files in this directory

| File | Purpose |
|------|---------|
| `cluster-credentials.template.env` | OpenShift cluster API URL, OAuth token, namespace |
| `ai-platform.template.env` | RHOAI dashboard/API URLs, NGC API key, model storage keys |
| `registry-credentials.template.yaml` | Kubernetes pull Secret for `nvcr.io` (NeMo container) |
| `workbench-secret.template.yaml` | Kubernetes Opaque Secret mounted into the RHOAI workbench pod |

---

## Workflow: copy → populate → apply

### 1. Copy the templates

```bash
cd openshift/secrets/

cp cluster-credentials.template.env   cluster-credentials.env
cp ai-platform.template.env           ai-platform.env
cp registry-credentials.template.yaml registry-credentials.yaml
cp workbench-secret.template.yaml     workbench-secret.yaml
```

### 2. Populate the `.env` files

Open each `.env` file and replace every `<placeholder>` with its real value.
Obtain values from your team's password manager / vault — do not generate
or share credentials over chat or email.

### 3. Apply Kubernetes secrets to the cluster

```bash
# Log in first
oc login $OPENSHIFT_API_URL --token=$OPENSHIFT_TOKEN

# Registry pull secret (needed once per namespace)
oc apply -f registry-credentials.yaml -n <your-namespace>
oc secrets link default registry-pull-secret --for=pull -n <your-namespace>

# Workbench runtime secret
oc apply -f workbench-secret.yaml -n <your-namespace>
```

### 4. Source `.env` files for local CLI use (optional)

```bash
set -a; source cluster-credentials.env; source ai-platform.env; set +a
```

---

## Gitignore protection

The following patterns are in `.gitignore` at the repo root and will prevent
populated files from being committed:

- `**/secrets/*.env` — matches `cluster-credentials.env`, `ai-platform.env`
- `**/secrets/*.yaml` — matches `registry-credentials.yaml`, `workbench-secret.yaml`
- `**/credentials.yaml`, `**/secrets.yaml`
- `.env`, `.env.*`, `.kube/`, `kubeconfig`

Template files (`*.template.env`, `*.template.yaml`) and this `README.md` are
explicitly **re-allowed** with negation rules so they remain version-controlled.

---

## Production secret management

For production workloads, do not use manually applied `oc apply` secrets.
Prefer one of these approaches:

**Sealed Secrets** (Bitnami)
```bash
kubeseal --format yaml < workbench-secret.yaml > workbench-sealed-secret.yaml
# Only the sealed manifest is committed; the controller decrypts it in-cluster.
```

**External Secrets Operator (ESO)**
Create an `ExternalSecret` CR pointing at your vault backend (AWS Secrets
Manager, HashiCorp Vault, Azure Key Vault). ESO syncs values into a native
Kubernetes Secret at runtime — no plaintext ever touches git.

**HashiCorp Vault + Vault Agent Injector**
Annotate workbench pods so Vault Agent sidecar injects secrets as files or
environment variables directly into the container.

---

## Pre-commit checklist

Before every `git commit` in this repo, confirm:

- [ ] No `.env` files in `git status` output
- [ ] No `*.yaml` files without `.template` in `openshift/secrets/`
- [ ] No real tokens, passwords, or keys visible in `git diff --cached`
- [ ] The pre-commit hook in `.githooks/pre-commit` is active
      (`git config core.hooksPath` returns `.githooks`)
- [ ] Ran `git diff --cached | grep -E 'AKIA|sk-|eyJ|ghp_|sha256~'` and got no output
