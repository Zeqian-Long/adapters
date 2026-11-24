# VM Handoff: First B_01 Validation of the 4DCodeBench Harbor Adapter

This document is for a coding agent on a fresh Linux GPU host with no context from earlier work. Read it to the end before running anything. Run the steps in order.

## 1. What this is

- **4DCodeBench** ([github.com/4DCodeBench/4DCodeBench](https://github.com/4DCodeBench/4DCodeBench)) is a benchmark in which an agent watches a video of a dynamic scene and writes a 4D world (`world/`: camera, render, per-frame meshes, dynamics) plus `solution/build.sh`. The original scorer (`python -m scorer`, GPU) compares it with reference data and writes per-metric readings to `results/reward.json`.
- **This adapter** (`src/4dcodebench/` in the Harbor adapters repository) turns each case into a Harbor task:
  - The agent container is built `FROM 4dcb/agent:dev`.
  - The separate verifier container is built `FROM 4dcb/scorer-checkpoints:dev`. It runs the unmodified upstream scorer and converts its readings into Harbor rewards.
  - The `README.md` next to this file describes the design.
- **The job here:** the first real validation, on one synthetic case, `synthetic/B_01`:
  1. Score the published reference world with the original scorer, outside Harbor (the "original side").
  2. Run the Harbor oracle on the generated B_01 task. It submits the same reference world.
  3. Compare the two results metric by metric.

### Facts already established (do not re-derive)

- **The oracle needs no reconstruction.** The synthetic oracle submits the published reference world from Hugging Face `4DCodeBench/Dataset-Synthetic`, `worlds/B_01.h5`, unpacked with the upstream `unpack`.
- **`solution/build.sh` is a labelled placeholder that exits 1.**
  - The scorer only checks that the file exists (`scorer/gate.py`, `WORLD_FILES`) and never runs it.
  - It does not regenerate the scene; the authors' generating programs are not released.
  - The same placeholder is used on both scoring paths:
    - the Harbor oracle installs it at `/workspace/solution/build.sh`;
    - `scripts/run_original_sanity.sh` places it at `<data>/synthetic/B_01/solution/build.sh`.
- **The reference world does not necessarily score 1.0.**
  - Geometry-quality metrics (`mesh_*`, `interpenetration`) can be below 1 by definition.
  - Some readings are legitimately `not_applicable` or `uncomputable`. The real-only metrics (`flow_distribution`, `track2d_dtw`, `depth_error`, `uni3d_moge_scene`) are `not_applicable` on synthetic cases.
  - GPU nondeterminism can cause small differences between runs.
- **What this validation checks:**
  - the Harbor oracle trial completes;
  - Harbor reproduces the original scorer's readings for the same world within normal run-to-run numerical variation.
- **Open, not for you to decide:** the official meaning of Harbor's "100% oracle" requirement for this multi-metric benchmark. Do not invent a reward, threshold or acceptance policy. Report what you observe.

## 2. State at handoff

**Done and checked on the preparation machine (Windows, no Docker):**
- Adapter code and 47 tests: 46 pass, and one skips without the scorer's numpy. Plus fixture-based Harbor schema checks.
- `harbor lint`: the default rules are clean. `--select ALL` flags only `unpinned-image` (local tags) and `root-agent` (as upstream).
- Real B_01 data, downloaded and verified:
  - The SHA-256 of every downloaded file matches the Hugging Face LFS hash.
  - The upstream `checker.check.check_world` reports 0 violations on the unpacked world.
  - `world/render.mp4` is byte-identical to `reference.mp4`.
- A B_01 Harbor task generated from that data, which loads in Harbor 0.27.0 as a separate-verifier GPU task.

**Never run anywhere yet:**
- any Docker image build;
- the GPU scorer;
- the original sanity run;
- the Harbor oracle trial;
- the comparison on real outputs.

Expect first-run problems in these.

## 3. What is copied to the VM, and what is fetched there

The user copies one folder, `vm-transfer/` (about 13 MB):

```
vm-transfer/
├── README.txt                      # pointer to this document
├── adapters-4dcodebench.bundle     # git bundle of branch 4dcodebench-adapter (the adapter, this file)
└── 4dcb/                           # B_01 reference data, upstream layout, outside Git
    ├── SHA256SUMS                  # every file below, for sha256sum -c
    ├── cases/synthetic/B_01/reference.mp4
    └── data/synthetic/B_01/
        ├── world/                  # camera.json, render.mp4, meshes/ (72 .npz), dynamics/ (48 .npz)
        └── estimates/              # semantic_dinov3.npz, semantic_tips.npz, geophys_dinov3.npz
```

**Fetched on the VM:**

| What | From | Size |
|---|---|---|
| 4DCodeBench checkout at `6c466be34107d7f388f30f97bf059b68ba1a21b0` | GitHub, public | small |
| Harbor CLI 0.27.0 | PyPI (`uv tool install harbor==0.27.0`) | small |
| Scorer checkpoints | Hugging Face, via upstream `scripts/download_checkpoints.sh`; DINOv3 is gated | 6.7 GB |
| Base images and packages for the image builds | Docker Hub, PyPI, GitHub, download.blender.org, npm and others | tens of GB |

**Kept outside Git, always:**
- the data home (`$BENCH`) and the checkpoints (`$CKPT`);
- generated tasks (`datasets/`, git-ignored at the repo root);
- trial output and validation output (`$OUT`);
- `.venv/`.

Do not commit any of them.

## 4. Host requirements

**Confirmed by the upstream code and docs:**
- **Platform:** Linux x86_64. The images download x86_64 builds of Python and Blender.
- **GPU:** an NVIDIA GPU. The images build CUDA extensions for compute capability 7.5 to 12.0 (`TORCH_CUDA_ARCH_LIST`), so Turing or newer.
- **Driver:** must support CUDA 13.0. The bases are `nvidia/cuda:13.0.2-devel-ubuntu24.04` and `pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel`. CUDA 13.0 needs the R580 driver branch or newer (check NVIDIA's CUDA 13.0 release notes for the exact minimum).
- **Docker:**
  - Docker Engine with the Compose v2 plugin (Harbor drives `docker compose`);
  - the NVIDIA Container Toolkit;
  - the default (`docker`) BuildKit builder, so that task images can build `FROM` the locally built `4dcb/*` images.
- **Scorer container limits:** 8 CPUs, 32768 MB of memory and 1 GPU (upstream `[scorer.environment]`; the same values are in the task's `[verifier.environment]`).
- **Outbound internet** for the builds and downloads above. The scorer itself runs with no network.

**Estimates, not upstream requirements:**
- **GPU memory:** 24 GB or more recommended. Upstream states no minimum; the scorer loads ViT-L-class models (DINOv3, TIPSv2, Uni3D) and rasterises at 1280×720.
- **System memory:** 64 GB recommended (the scorer alone is limited to 32 GB).
- **CPU:** 8 cores minimum, 16 recommended.
- **Disk:** 200 GB free on the Docker root directory recommended. That covers the four images, the two per-task images Harbor builds, and the 6.7 GB checkpoint layer.
- **Build time:** the scorer image compiles CUDA extensions and OpenVDB. Expect it to take tens of minutes or more.

## 5. Procedure

Each step lists its commands and what to expect.

**Shell state.** If your shell does not keep environment variables between commands (for example Claude Code's Bash tool), start every command with `source ~/4dcb-work/env.sh`.

### Step 0: Variables

```bash
mkdir -p ~/4dcb-work
cat > ~/4dcb-work/env.sh <<'EOF'
export WORK="$HOME/4dcb-work"
export TRANSFER="$WORK/vm-transfer"     # the copied folder
export ADAPTERS="$WORK/adapters"        # adapters repository, branch 4dcodebench-adapter
export UPSTREAM="$WORK/4DCodeBench"     # upstream checkout, read-only
export BENCH="$WORK/4dcb"               # data home: cases/, data/, SHA256SUMS
export CKPT="$WORK/checkpoints"         # scorer checkpoints
export OUT="$WORK/b01-validation"       # everything this validation produces
export PATH="$HOME/.local/bin:$PATH"
EOF
source ~/4dcb-work/env.sh
```

Place the copied `vm-transfer/` at `$TRANSFER`. If it is elsewhere, edit `env.sh`.

### Step 1: Check the transferred project and data; fetch the upstream checkout

```bash
source ~/4dcb-work/env.sh
git clone -b 4dcodebench-adapter "$TRANSFER/adapters-4dcodebench.bundle" "$ADAPTERS"
git -C "$ADAPTERS" remote rename origin bundle
git -C "$ADAPTERS" remote add upstream https://github.com/harbor-framework/adapters.git
git -C "$ADAPTERS" log --oneline -3

git clone https://github.com/4DCodeBench/4DCodeBench.git "$UPSTREAM"
git -C "$UPSTREAM" checkout 6c466be34107d7f388f30f97bf059b68ba1a21b0

cp -a "$TRANSFER/4dcb" "$BENCH"
(cd "$BENCH" && sha256sum --quiet -c SHA256SUMS && echo "B_01 data OK")
```

**Expected:** `B_01 data OK`. Do not modify `$UPSTREAM`.

If the data must be fetched again, run this after Step 2 (about 13 MB, public, no login):

```bash
cd "$ADAPTERS/src/4dcodebench" && uv run scripts/fetch_case.py --repo "$UPSTREAM" --root "$BENCH" synthetic/B_01
```

### Step 2: Host software

Skip any part that is already in place. `nvidia-smi` must work on the host first. If it doesn't, install an R580+ NVIDIA driver (provider-specific; on Ubuntu, `sudo ubuntu-drivers install` or the provider's GPU image) and reboot.

```bash
nvidia-smi        # driver version and GPU memory

# Docker Engine and the Compose plugin (Ubuntu; docs.docker.com/engine/install/ubuntu/)
sudo apt-get update && sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker "$USER"

# NVIDIA Container Toolkit (docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Docker group.** The `docker` group applies to new logins only. Either ask the user to restart the session, or run Docker commands through `sg docker -c '...'` until then. Do not `chmod` the Docker socket.

**GPU inside Docker:**

```bash
docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi
```

**Expected:** the same GPU table as on the host.

**uv, Harbor, and the Hugging Face CLI:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/4dcb-work/env.sh
uv tool install harbor==0.27.0
uv tool install "huggingface_hub[cli]"
harbor --version && uv --version && hf --version
```

### Step 3: Checkpoints (6.7 GB; gated DINOv3)

This step needs the user:
1. The user's Hugging Face account must have been granted access to [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m). Request it on that page; approval is not instant.
2. The user must authenticate on the VM, either by typing `! hf auth login` or by exporting `HF_TOKEN` themselves.

Never print, log or commit the token.

```bash
source ~/4dcb-work/env.sh
hf auth whoami
bash "$UPSTREAM/scripts/download_checkpoints.sh" "$CKPT"
```

**Expected:** seven `OK` lines from `sha256sum -c`, then `checkpoints ready in …`. The script pins every revision. A 401 or 403 on DINOv3 means access was not granted or the login is missing. The images in Step 4 can build while this runs.

### Step 4: Build the images

```bash
source ~/4dcb-work/env.sh
mkdir -p "$OUT"
bash "$ADAPTERS/src/4dcodebench/scripts/build_images.sh" "$UPSTREAM" "$CKPT" 2>&1 | tee "$OUT/build_images.log"
```

**Build order:**
1. `4dcb/sim-base:dev` (upstream `harness/base/Dockerfile`)
2. `4dcb/agent:dev` (`harness/agent/Dockerfile`)
3. `4dcb/scorer:dev` (`harness/scorer/Dockerfile`)
4. `4dcb/scorer-checkpoints:dev` (`src/4dcodebench/docker/scorer-checkpoints.Dockerfile`: the scorer image plus `$CKPT` at `/checkpoints`)

The first three use the upstream README's own build commands. To build the first three before the checkpoints finish, omit `"$CKPT"`, then run the script again with it (the cache makes the rerun cheap).

**Smoke tests (cheap GPU checks):**

```bash
docker run --rm --gpus all 4dcb/sim-base:dev        # upstream verify_gpu.py
docker run --rm --gpus 1 --entrypoint python 4dcb/scorer-checkpoints:dev \
    -c "import torch, os; print(torch.cuda.is_available(), torch.cuda.get_device_name(0)); print(sorted(os.listdir('/checkpoints')))"
```

**Expected:** `True <GPU name>` and the seven checkpoint directories.

### Step 5: Preflight and unit tests

```bash
source ~/4dcb-work/env.sh
bash "$ADAPTERS/src/4dcodebench/scripts/vm_preflight.sh" "$UPSTREAM" "$BENCH" "$CKPT" | tee "$OUT/preflight.txt"
cd "$ADAPTERS/src/4dcodebench"
uv sync
uv run pytest -q
HARBOR_4DCODEBENCH_UPSTREAM="$UPSTREAM" uv run pytest -q tests/test_upstream.py
```

**Expected:**
- `preflight: all checks passed` (WARN lines are informational).
- `pytest`: all pass. `test_reward_keys_match_the_scorer` skips, because the adapter environment lacks the scorer's numpy and h5py.

### Step 6: Generate the Harbor task for B_01

```bash
source ~/4dcb-work/env.sh
cd "$ADAPTERS/src/4dcodebench"
uv run 4dcodebench --repo "$UPSTREAM" --cases-root "$BENCH/cases" --data-root "$BENCH/data" \
    --task-ids synthetic/B_01 --output-dir "$ADAPTERS/datasets/4dcodebench" --overwrite
harbor lint "$ADAPTERS/datasets/4dcodebench/synthetic-b-01"
```

**Expected:**
- `1 task(s) written`, and `datasets/4dcodebench/synthetic-b-01/` plus `datasets/4dcodebench/metric.py`.
- `harbor lint` prints no findings.

Generate before Step 7. On the same filesystem, files under `tests/bench/` and `solution/world/` are hard links into `$BENCH`, so never edit them in place.

### Step 7: Original side (upstream scorer on the reference world)

```bash
source ~/4dcb-work/env.sh
for run in sanity-1 sanity-2; do
    bash "$ADAPTERS/src/4dcodebench/scripts/run_original_sanity.sh" \
        "$BENCH/cases" "$BENCH/data" "$CKPT" synthetic/B_01 "$run" 2>&1 | tee "$OUT/original-$run.log"
    cp -a "$BENCH/data/synthetic/B_01/$run" "$OUT/original-$run"
done
```

**What the script does:**
- Runs `4dcb/scorer:dev` with the upstream `--sanity` layout: `--network none`, 8 CPUs, 32 GB, 1 GPU, checkpoints mounted read-only.
- Installs the placeholder `build.sh` beside the reference world first.

**Two runs.** The second run measures run-to-run variation. Each run needs a new run name, because the scorer merges into an existing `reward.json`; the script refuses an existing name.

**Expected:**
- `<run>/results/reward.json` and `reward.detail.json`.
- `"status": "ok"` in the detail, which means both gates passed.

A nonzero scorer exit status is printed but does not stop the script; read the results. Files the container writes into `$BENCH/data` are owned by root.

### Step 8: Harbor side (the B_01 oracle trial)

```bash
source ~/4dcb-work/env.sh
cd "$ADAPTERS"
for n in 1 2; do
    harbor trial start -p datasets/4dcodebench/synthetic-b-01 -a oracle \
        --trials-dir "$OUT/trials" --trial-name "harbor-oracle-$n" 2>&1 | tee "$OUT/harbor-oracle-$n.log"
done
```

**The first trial** builds the two task images:
- `environment/`, `FROM 4dcb/agent:dev`;
- `tests/`, `FROM 4dcb/scorer-checkpoints:dev`, plus the B_01 reference data.

**What happens in each trial:**
1. The oracle copies the reference world and the placeholder `build.sh` into `/workspace`.
2. Harbor hands `/workspace/world` and `/workspace/solution` to the verifier.
3. The verifier runs the scorer.

**Expected per trial,** under `$OUT/trials/harbor-oracle-<n>/`:
- `result.json` with no exception, and no `exception.txt`.
- `verifier/reward.json`: finite numbers only, including `executable: 1.0` when both gates passed.
- `verifier/4dcodebench_readings.json`: every original reading next to its Harbor value.
- `verifier/4dcodebench_verifier.log`: the scorer's output.
- `verifier/4dcodebench/results/reward.json` and `reward.detail.json`: the scorer's own output, the input to Step 9.

### Step 9: Compare metric by metric

```bash
source ~/4dcb-work/env.sh
cd "$ADAPTERS/src/4dcodebench"
compare_results() { uv run 4dcodebench-compare --original "$1" --harbor "$2" --json "$OUT/compare-$3.json" | tee "$OUT/compare-$3.txt"; }
compare_results "$OUT/original-sanity-1/results" "$OUT/trials/harbor-oracle-1/verifier/4dcodebench/results" original1-vs-harbor1
compare_results "$OUT/original-sanity-2/results" "$OUT/trials/harbor-oracle-2/verifier/4dcodebench/results" original2-vs-harbor2
compare_results "$OUT/original-sanity-1/results" "$OUT/original-sanity-2/results"                            original1-vs-original2
compare_results "$OUT/trials/harbor-oracle-1/verifier/4dcodebench/results" "$OUT/trials/harbor-oracle-2/verifier/4dcodebench/results" harbor1-vs-harbor2
```

**Output.** `4dcodebench-compare` prints, for all 18 keys, both readings and their absolute difference, plus the status, the gate failure and the registration residual. Both arguments are just `results/` directories, so it also compares original with original. Its exit status is 1 if:
- a reading's state differs (number vs `error`, `not_applicable`, …); or
- the gate verdicts differ.

With no `--atol` it does not fail on numeric differences.

**Read it as follows.** Compare the original-vs-Harbor differences with the original-vs-original and Harbor-vs-Harbor differences. Do not pick a tolerance or declare a pass threshold. Report the numbers.

### Step 10: Record the environment and package the outputs

```bash
source ~/4dcb-work/env.sh
{
  date -u; nvidia-smi; docker version
  for image in 4dcb/sim-base:dev 4dcb/agent:dev 4dcb/scorer:dev 4dcb/scorer-checkpoints:dev; do
      echo "$image $(docker image inspect --format '{{.Id}}' "$image")"; done
  echo "4DCodeBench $(git -C "$UPSTREAM" rev-parse HEAD)"; echo "adapters $(git -C "$ADAPTERS" rev-parse HEAD)"
  harbor --version; uv --version
} > "$OUT/environment.txt" 2>&1
tar -C "$WORK" -czf "$WORK/b01-validation.tar.gz" b01-validation
```

**Report to the user:**
- whether each run completed;
- `executable` and the gate status on both sides;
- the four comparison tables (or any state mismatches);
- any deviation from these commands, such as a changed timeout;
- where `b01-validation.tar.gz` is.

## 6. Failure points and diagnosis

| Symptom | Likely cause | Check / fix |
|---|---|---|
| `permission denied … docker.sock` | not in the `docker` group in this shell | new login, or `sg docker -c '…'` |
| `could not select device driver "" with capabilities: [[gpu]]` | NVIDIA Container Toolkit missing or not configured | Step 2 toolkit commands, `sudo systemctl restart docker` |
| `unsatisfied condition: cuda>=13.0` / `CUDA driver version is insufficient` | driver older than CUDA 13.0 support | upgrade to R580+ |
| upstream image build fails while downloading | network fetch in the upstream Dockerfile (Blender, NodeSource, npm, Antigravity installer, Blender docs, PyPI, GitHub) | rerun; the agent CLIs are installed unpinned upstream, so record their versions if they change |
| scorer image build fails compiling CUDA extensions | memory or toolkit problem in the `pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel` stage | check the log; give the build more memory; do not change the upstream Dockerfile |
| task image build: `pull access denied for 4dcb/agent` | base image not built locally, or a non-default buildx builder | `docker image ls '4dcb/*'`; `docker buildx use default` |
| DINOv3 download 401/403 | access not granted or no login | Step 3 |
| verifier fails, `scorer checkpoints missing` | task built `FROM` a scorer image without `/checkpoints` | build `4dcb/scorer-checkpoints:dev`; regenerate with the default `--scorer-image` |
| trial has no `verifier/reward.json` | scorer crashed or timed out (verifier timeout 3600 s; 1800 s per metric) | `verifier/4dcodebench_verifier.log`, `verifier/4dcodebench_failure.txt`, `exception.txt`; GPU or host OOM shows in the log or `dmesg` |
| every geometry metric is `error` | world gate closed | `world_gate` in `results/reward.detail.json`; on the original side, check `$BENCH/data/synthetic/B_01/solution/build.sh` exists and is executable |
| `run_original_sanity.sh`: `… exists; pass a new run name` | run name reused | use a new name; the scorer would otherwise merge with old readings |
| cannot delete `$BENCH/data/synthetic/B_01/sanity-*` | written by the container as root | `sudo rm -rf …` |
| `harbor lint` reports `unpinned-image` / `root-agent` only with `--select ALL` | local `:dev` tags; root agent as upstream | expected |

If a timeout is too short on this GPU, do not edit the task silently. Rerunning with `harbor trial start … --timeout-multiplier <x>` is a deviation, so record it.

## 7. Known unresolved issues (do not resolve them yourself)

- **The 100% oracle requirement.** Harbor's oracle rule against per-metric readings below 1.0 by definition: for the user and the Harbor team.
- **Primary reward.** There is none; upstream implements no overall score. The adapter emits all readings plus `executable`.
- **Real-world oracle.** Unresolved: no reference worlds exist, and the real-split `solve.sh` exits 1. Out of scope here.
- **Placeholder `build.sh`.** Whether the 4DCodeBench authors accept it for oracle purposes.
- **`error` mapping.** The adapter maps `error` to the worst value, per `docs/spec.md`. The upstream viewer averages over executable runs instead. The raw readings are kept in `4dcodebench_readings.json`.
- **Agent network.** Public, as upstream. The reference worlds are public on Hugging Face, which matters for real agents, not for the oracle.
- **Licensing and redistribution.** 23 real videos are not redistributed, and the Hugging Face datasets declare no licence.
- **Metadata TODO.** The builder name and contact in `adapter_metadata.json` and `README.md`.
- **Parity.** Not run. Its timing and handling are decided separately by the user.

## 8. Rules for this session

- Do not push, open or update PRs, or take any GitHub action. The `bundle` remote is a local file. Leave `upstream` fetch-only.
- Do not modify the upstream `4DCodeBench` checkout or Harbor itself.
- Do not change metrics, reward mapping, thresholds or the oracle. If something in the adapter is broken, make the smallest fix, run `uv run pytest -q`, and report it.
- Do not download the full dataset, run real agents, or run parity experiments.
- Keep data, checkpoints, generated tasks and outputs out of Git.
