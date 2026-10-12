# `fraud-mlops.yml`: Line-by-Line Explanation

This is a reference guide to the GitHub Actions workflow at [`.github/workflows/fraud-mlops.yml`](../.github/workflows/fraud-mlops.yml).
It explains every block of the file, what it does, and why it is there.

---

## The big picture

```
 push / pull request / manual button
              │
              ▼
 ┌──────────────────── job: ci ────────────────────┐
 │ checkout → python → install → train (gate)      │
 │ → pytest → docker build → smoke test → upload    │
 └──────────────────────┬──────────────────────────┘
                        │ needs: ci   +   only on master
                        ▼
 ┌──────────────────── job: cd ────────────────────┐
 │ checkout → download model → login → build+push  │
 └─────────────────────────────────────────────────┘
                        │
                        ▼
          ghcr.io/nuelbundi/fraud-api:latest
```

**Key rules to remember**

1. A workflow file must live in `.github/workflows/` at the **root** of the repo. GitHub ignores it anywhere else.
2. Each **job** runs on a **brand-new computer** that is deleted afterwards. Jobs share nothing unless you pass files between them.
3. **Steps** inside a job run **in order**. If any step fails (exits with an error), the job stops.
4. YAML uses **indentation** (spaces, never tabs) to show structure. Wrong indentation is the most common mistake.

---

## Block 1: Comments and name

```yaml
# CI: train + quality gate (MLflow) -> test the API -> build the Docker image and check it runs
# CD: only on master, and only if CI passed -> push the Docker image to GitHub Container Registry
name: fraud-mlops
```

| Line | Meaning |
|---|---|
| `# ...` | Comments. GitHub ignores them; they are for humans. |
| `name: fraud-mlops` | The name shown in the **Actions** tab on GitHub. |

---

## Block 2: `on:` (WHEN does the workflow run?)

```yaml
on:
  push:
    branches: [master]
    paths: ["fraud_mlops/**", "Data/CleanedFeaturesFromLoan.csv", ".github/workflows/fraud-mlops.yml"]
  pull_request:
    paths: ["fraud_mlops/**", "Data/CleanedFeaturesFromLoan.csv", ".github/workflows/fraud-mlops.yml"]
  workflow_dispatch:
```

| Line | Meaning |
|---|---|
| `on:` | The list of **triggers**: events that start the workflow. |
| `push:` | Run when someone pushes commits… |
| `branches: [master]` | …but only pushes to the `master` branch. |
| `paths: [...]` | …and only if one of these files changed. `**` means "anything inside this folder, at any depth". Pushing a change to `lab09_rag/` will **not** start this workflow. |
| `pull_request:` | Also run when a pull request is opened or updated, using the same path filter. This lets you test a change **before** merging it. |
| `workflow_dispatch:` | Adds a **"Run workflow"** button in the Actions tab so you can start it manually. It takes no options, so nothing follows the colon. |

> **Why include the dataset and the workflow file in `paths`?** If the data changes, the model must be retrained. If the workflow itself changes, you want to see that it still works.

---

## Block 3: `defaults:` (where commands run)

```yaml
defaults:
  run:
    working-directory: fraud_mlops
```

| Line | Meaning |
|---|---|
| `defaults: run: working-directory:` | Every `run:` step executes **inside `fraud_mlops/`**, as if you had typed `cd fraud_mlops` first. So `python train.py` finds `train.py`. |

> ⚠️ **This only affects `run:` steps, not `uses:` steps.** That's why `upload-artifact` (a `uses:` step) needs the full path `fraud_mlops/model/`.

---

## Block 4: `env:` (variables for the whole workflow)

```yaml
env:
  IMAGE: ghcr.io/nuelbundi/fraud-api
```

| Line | Meaning |
|---|---|
| `env:` | Environment variables available to every job and step. |
| `IMAGE: ghcr.io/nuelbundi/fraud-api` | The name of the Docker image. `ghcr.io` = GitHub Container Registry, `nuelbundi` = the owner (**must be lowercase**), `fraud-api` = the image name. Steps use it as `$IMAGE`. |

> Defining it once means that if you rename the image, you change one line, not several.

---

## Block 5: `jobs:` → `ci:` (Continuous Integration)

```yaml
jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
```

| Line | Meaning |
|---|---|
| `jobs:` | The list of jobs. Jobs run **in parallel** unless linked with `needs:`. |
| `ci:` | The job's ID. You choose the name; it is used by `needs: ci` later. |
| `runs-on: ubuntu-latest` | Run on a fresh Linux virtual machine provided free by GitHub. It already has Git, Docker, curl and Python installed. |
| `steps:` | The ordered list of things to do. Each step starts with `-`. |

### Step 1: Get the code

```yaml
      - uses: actions/checkout@v4
```

| Part | Meaning |
|---|---|
| `uses:` | Run a ready-made **action** written by someone else, instead of your own command. |
| `actions/checkout` | The official action that downloads (clones) your repo onto the machine. **The machine starts empty**, so without this step there is no code. |
| `@v4` | The version. Pinning a version stops your workflow breaking when the action is updated. |

### Step 2: Install Python

```yaml
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
```

| Part | Meaning |
|---|---|
| `actions/setup-python@v5` | Official action that installs a specific Python version. |
| `with:` | Inputs (options) for the action. Each action's page on the GitHub Marketplace lists the inputs it accepts. |
| `python-version: "3.12"` | Use Python 3.12, the same version as the Docker image (`python:3.12-slim`). Quoted so YAML treats it as text and not the number `3.12`. (Unquoted `3.10` would become `3.1`!) |

### Step 3: Install packages

```yaml
      - name: Install
        run: pip install -r requirements.txt
```

| Part | Meaning |
|---|---|
| `name:` | A label shown in the GitHub log. Optional, but makes logs readable. |
| `run:` | A shell command, exactly as you'd type it in a terminal. |
| `pip install -r requirements.txt` | Installs mlflow, scikit-learn, fastapi, pytest, etc. Runs inside `fraud_mlops/` (Block 3). |

### Step 4: Train and apply the quality gate

```yaml
      - name: Train models + quality gate (MLflow)
        run: python train.py
```

What `train.py` does:
1. Trains 4 models and logs each to **MLflow** (parameters, metrics and the model).
2. Picks the best by F1-score.
3. **Quality gate:** if F1 < 0.50, it calls `sys.exit(1)`. A non-zero exit code **fails this step**, which **stops the whole pipeline**. Nothing gets deployed.
4. Otherwise it registers the model as `production` and saves it to `model/`.

> This is the core idea of CI/CD for ML: **a bad model is treated like a failing test.**

### Step 5: Test the API

```yaml
      - name: Test the API
        run: pytest -q
```

| Part | Meaning |
|---|---|
| `pytest` | Finds and runs `tests/test_api.py` (4 tests). If any test fails, pytest exits with an error and the job stops. |
| `-q` | "Quiet": shorter output. |

### Step 6: Build the Docker image

```yaml
      - name: Build the Docker image
        run: docker build -t fraud-api .
```

| Part | Meaning |
|---|---|
| `docker build` | Follows the instructions in `Dockerfile` to create an image. |
| `-t fraud-api` | "Tag": gives the image the name `fraud-api`. |
| `.` | Build from the current folder (`fraud_mlops/`), which contains the `Dockerfile`, `api/` and the `model/` just created by Step 4. |

> If the Dockerfile has a mistake, this step fails. You find out **before** deploying.

### Step 7: Smoke test the container

```yaml
      - name: Smoke test the container
        run: |
          docker run -d -p 8000:8000 --name api fraud-api
          for i in $(seq 1 30); do curl -s http://localhost:8000/health && break; sleep 1; done
          curl --fail http://localhost:8000/health
          curl --fail http://localhost:8000/model-info
```

| Line | Meaning |
|---|---|
| `run: \|` | The `\|` lets you write **several commands** on separate lines. They run one after another. |
| `docker run -d` | Start the container in the background ("detached"). |
| `-p 8000:8000` | Connect port 8000 on the machine to port 8000 inside the container. |
| `--name api` | Name the running container `api`. |
| `for i in $(seq 1 30); do ... done` | The API takes a few seconds to start. This tries `/health` up to 30 times, once per second, and stops (`break`) as soon as it answers. |
| `curl --fail .../health` | Call the endpoint. `--fail` makes curl exit with an error if the response is an HTTP error (or never comes), which fails the step. |
| `curl --fail .../model-info` | Check that the model loaded and its info is returned. |

> A **smoke test** is a quick "does it turn on without catching fire?" check. pytest (Step 5) tested the Python code; this tests the **actual Docker image** that will be shipped.

### Step 8: Save the results

```yaml
      - name: Save the model and MLflow results
        uses: actions/upload-artifact@v4
        with:
          name: fraud-model
          path: |
            fraud_mlops/model/
            fraud_mlops/mlflow.db
            fraud_mlops/mlruns/
```

| Part | Meaning |
|---|---|
| `actions/upload-artifact@v4` | Saves files from this machine to GitHub storage **before the machine is deleted**. |
| `name: fraud-model` | The artifact's name, used to download it later. |
| `path: \|` | The files and folders to save (several lines). Full paths are needed because `uses:` steps ignore `working-directory`. |
| `model/` | The trained model + metadata, needed by the `cd` job. |
| `mlflow.db`, `mlruns/` | The MLflow tracking data. Download them from the run page and open them with `mlflow ui` to inspect the experiments. |

> Artifacts appear at the bottom of the run's page in the Actions tab and are kept for 90 days by default.

---

## Block 6: `cd:` (Continuous Deployment)

```yaml
  cd:
    needs: ci
    if: github.ref == 'refs/heads/master'
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write   # allows pushing to ghcr.io with the built-in GITHUB_TOKEN
    steps:
```

| Line | Meaning |
|---|---|
| `cd:` | Second job's ID. |
| `needs: ci` | **Wait for `ci` to finish, and only run if it succeeded.** Without this, both jobs would start at the same time. |
| `if: github.ref == 'refs/heads/master'` | Only deploy for the `master` branch. On a pull request, `ci` runs but `cd` is **skipped**: you test proposals but only deploy approved code. |
| `github.ref` | A built-in variable holding the branch that triggered the run. |
| `runs-on: ubuntu-latest` | A **new, separate machine**. It does not have the files from `ci`. |
| `permissions:` | What this job's automatic token (`GITHUB_TOKEN`) is allowed to do. |
| `contents: read` | May read the repo code. |
| `packages: write` | May push packages (Docker images) to `ghcr.io`. Without this, the push is refused. |

### Step 1: Get the code again

```yaml
      - uses: actions/checkout@v4
```

New machine, empty again, so the repo must be cloned again (it needs `Dockerfile`, `api/` and `requirements-api.txt`).

### Step 2: Get the model trained in CI

```yaml
      - name: Get the model trained in CI
        uses: actions/download-artifact@v4
        with:
          name: fraud-model
          path: fraud_mlops
```

| Part | Meaning |
|---|---|
| `actions/download-artifact@v4` | Downloads an artifact uploaded earlier in the same run. |
| `name: fraud-model` | Same name as in `ci` Step 8. |
| `path: fraud_mlops` | Put the files back inside `fraud_mlops/`, so `model/` ends up at `fraud_mlops/model/`. |

> **Why not just run `train.py` again?** That takes ~3 minutes, and you'd be shipping a *different* model from the one that passed the tests. Re-using the artifact guarantees **the exact model that was tested is the one deployed.**

### Step 3: Log in to the registry

```yaml
      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
```

| Part | Meaning |
|---|---|
| `docker/login-action@v3` | Official Docker action to log in to an image registry. |
| `registry: ghcr.io` | GitHub Container Registry (free storage for Docker images). |
| `${{ ... }}` | **Expression syntax**: GitHub replaces it with a value before running. |
| `github.actor` | The username of whoever triggered the run. |
| `secrets.GITHUB_TOKEN` | A temporary password **created automatically** for every run and deleted afterwards. You do not need to create it. It is allowed to push because of `packages: write` above. |

> For Docker Hub instead, you would add your own secrets in **Settings → Secrets and variables → Actions** and use `${{ secrets.DOCKERHUB_TOKEN }}`.

### Step 4: Build and push

```yaml
      - name: Build and push the image
        run: |
          docker build -t $IMAGE:latest -t $IMAGE:${{ github.sha }} .
          docker push $IMAGE:latest
          docker push $IMAGE:${{ github.sha }}
```

| Line | Meaning |
|---|---|
| `docker build ... .` | Build the image again on this machine (the image from `ci` was deleted with its machine). |
| `-t $IMAGE:latest` | Tag 1: `ghcr.io/nuelbundi/fraud-api:latest`, always the newest version. |
| `-t $IMAGE:${{ github.sha }}` | Tag 2: the **commit ID** (e.g. `35ec6bf…`). This is a permanent label for this exact version, so you can **roll back** to it later. |
| `docker push` | Upload each tag to ghcr.io. |

After this, anyone can run the deployed model with:
```bash
docker run -p 8000:8000 ghcr.io/nuelbundi/fraud-api:latest
```
(The package is private at first: GitHub profile → **Packages** → **fraud-api** → **Package settings** → **Change visibility** → Public.)

---

## Cheat sheet

| Keyword | Means |
|---|---|
| `name` | Label shown on GitHub |
| `on` | When to run (push, pull_request, workflow_dispatch, schedule) |
| `branches` / `paths` | Filters on the trigger |
| `defaults.run.working-directory` | Folder for all `run:` steps |
| `env` | Variables (`$NAME` in commands) |
| `jobs` | Groups of steps; each job is a fresh machine |
| `runs-on` | Which machine type |
| `needs` | Run this job after another one succeeds |
| `if` | Run only when a condition is true |
| `permissions` | What the automatic token may do |
| `steps` | Ordered list of actions/commands |
| `uses` | Run a ready-made action (`owner/name@version`) |
| `with` | Inputs for a `uses:` action |
| `run` | Run shell command(s); `\|` for several lines |
| `${{ ... }}` | Insert a GitHub value (`github.sha`, `github.actor`, `secrets.X`) |

## What makes a step fail?

Any command that exits with a **non-zero code**:

| Command | Fails when |
|---|---|
| `python train.py` | `sys.exit(1)`: model below the quality gate |
| `pytest` | any test fails |
| `docker build` | Dockerfile error |
| `curl --fail` | API not answering / HTTP error |

One failed step leads to a failed job. A failed `ci` means `cd` never runs, and **nothing broken reaches production.**
