# 4DCodeBench → Harbor Adapter

## Overview

This adapter converts [4DCodeBench](https://github.com/4DCodeBench/4DCodeBench) into Harbor tasks.

- **Task type:** inverse graphics of dynamic scenes. The agent watches a video and writes graphics code whose Blender render, camera, per-frame geometry and material trajectories reproduce the 4D scene.
- **Splits:** 100 synthetic cases, which have published reference 4D worlds, and 100 real-world cases, which have none. Each case becomes one Harbor task, 200 in all.
- **Upstream pin:** commit `6c466be34107d7f388f30f97bf059b68ba1a21b0`. Each generated task records the commit of the checkout it was generated from.
- **Scoring:** the original scorer (`python -m scorer`), unmodified, in a separate GPU verifier container. Its per-metric readings become Harbor rewards; nothing is re-implemented and no combined score is formed.
- **Status:** first implementation.
  - Done: task generation, unit tests with fixtures, and a Harbor schema check.
  - **Not yet executed:** Docker/GPU runs, the synthetic oracle, and parity.

## What is 4DCodeBench?

4DCodeBench (Shen et al., 2026) benchmarks coding agents on reconstructing a dynamic scene from a single video as an explicit, reproducible 4D program.

- **What the agent delivers:**
  - `world/`: `camera.json`, `render.mp4`, `meshes/` and `dynamics/`, plus `solver/` for liquids.
  - `solution/`: an executable `build.sh` that rebuilds `world/` from nothing.
- **What is compared:**
  - the render against the reference video, on every case;
  - the geometry and dynamics against the reference world, on synthetic cases;
  - against estimates from off-the-shelf models, on real cases.
- **Metric families** (`docs/spec.md`):

  | Family | Metrics |
  |---|---|
  | Perceptual | DINOv3 similarity |
  | 2D dynamics | Dynamic IoU, Flow, Track2D |
  | 2.5D geometry | Depth error |
  | 3D geometry | Scene 3D |
  | 3D dynamics | Trajectory DTW, EMD step |

- Additional metrics (TIPS, GeoPhys, Uni3D, Occupancy DTW) and geometry-quality metrics are reported separately.

## Adapter Features

- **One task per case.** Task names are `4dcodebench/<kind>-<case>`, lowercased, for example `4dcodebench/synthetic-b-01`. `--split synthetic|real|all` selects cases.
- **The original instruction.**
  - `instruction.md` is the upstream `harness/agent_context/TASK.md`, rendered with the harness's own container paths.
  - The agent also gets the original `/task/` directory (`TASK.md`, `world-format.md`, `render-settings.md`) and `/input/reference.mp4`, exactly what `harness/runtime/infer.py` mounts.
- **The original agent image.**
  - `environment/Dockerfile` builds `FROM` the upstream agent image (CUDA 13, Blender 5.2, bpy, PyTorch, Taichi, Warp, FFmpeg and the public checker) and adds only the task context and the video.
  - GPU: `gpus = 1` in `task.toml`, plus `environment/docker-compose.yaml` with an NVIDIA reservation.
- **A separate verifier.**
  - `[verifier] environment_mode = "separate"`. `tests/Dockerfile` builds `FROM` the upstream scorer image with its checkpoints, and adds the case's private reference data under `/bench`.
  - Only `/workspace/world` and `/workspace/solution` cross from the agent (task `artifacts`), the two directories the upstream harness collects.
  - The verifier runs with `no-network`, as upstream does.
- **The original scorer, unchanged.**
  - `tests/run_verifier.py` writes the same manifest entry `harness/runtime/score.py` writes for a delivered run.
  - It runs `python -m scorer` with the upstream timeouts (1800 s per metric, 600 s per preparation) and thread settings.
  - The scorer's `results/` stays under `/logs/verifier/4dcodebench/` for inspection.
- **Harbor rewards** (`rewards.py`): `/logs/verifier/reward.json` holds finite numbers only, mapped as the benchmark's specification aggregates them:

  | Original reading | Harbor reward |
  |---|---|
  | number | unchanged |
  | `error` | the metric's worst value: 0, or 2 for `depth_error` (lower is better) |
  | `error` from a closed gate, on `mesh_*` / `interpenetration` | left out: geometry quality is averaged over runs that pass the world gate |
  | `not_applicable`, `excluded`, `uncomputable` | left out |

  - `executable` is 1 when both validation gates pass.
  - `4dcodebench_readings.json` records every original reading and its cause next to the Harbor value.
- **Dataset metric** (`metric.py`, written to the dataset root):
  - It averages each key over the trials that report it.
  - Harbor's default `mean` would instead count a metric that does not apply to a case (for example `depth_error` on synthetic cases) as 0.
- **Synthetic oracle.**
  - `solution/solve.sh` copies the published reference world into `/workspace/world`.
  - It installs `solution/build.sh`, a clearly labelled placeholder that exits 1. It exists because the scorer's world gate requires `solution/build.sh`, and the generating simulation programs are not released.
- **Real-world oracle.** Marked unresolved: `solve.sh` writes nothing and exits 1.
- **No custom agents.** Harbor's standard installed agents run against these tasks.

## Generated Task Structure

```
datasets/4dcodebench/
├── metric.py                         # dataset metric (Harbor uv-script)
└── synthetic-b-01/
    ├── task.toml                     # name, authors, GPU, separate verifier, artifacts
    ├── instruction.md                # rendered upstream TASK.md
    ├── environment/                  # agent image: FROM the upstream agent image
    │   ├── Dockerfile
    │   ├── docker-compose.yaml       # NVIDIA reservation
    │   ├── task/                     # TASK.md, world-format.md, render-settings.md
    │   └── input/reference.mp4
    ├── tests/                        # verifier image: FROM the upstream scorer image + checkpoints
    │   ├── Dockerfile
    │   ├── docker-compose.yaml
    │   ├── test.sh
    │   ├── run_verifier.py           # manifest, `python -m scorer`, reward conversion
    │   ├── rewards.py
    │   ├── case.json
    │   └── bench/                    # private: cases/<kind>/<case>/reference.mp4,
    │                                 #          data/<kind>/<case>/{world|annotation, estimates}
    └── solution/
        ├── solve.sh                  # synthetic: submit the reference world; real: UNRESOLVED
        ├── build.sh                  # synthetic only: oracle placeholder
        └── world/                    # synthetic only: the published reference world
```

The adapter package lives at `src/4dcodebench/src/harbor_4dcodebench/`: `adapter.py`, `main.py`, `rewards.py`, `metric.py`, `compare.py` and `task-template/`. The Python module is `harbor_4dcodebench` because a module name cannot start with a digit.

## Run Evaluation / Harness

Once the dataset is registered:

```bash
harbor run -d 4dcodebench/4dcodebench -a <agent> -m <model>
```

Before registration, from the adapters repository root, after generating tasks:

```bash
# oracle over both splits (real cases are unresolved and exit 1)
harbor run -c src/4dcodebench/run_4dcodebench.yaml

# synthetic oracle only
harbor run -c src/4dcodebench/run_4dcodebench_synthetic_oracle.yaml

# one task with an agent
harbor trial start -p datasets/4dcodebench/synthetic-b-01 -a claude-code -m anthropic/claude-opus-5-5
```

Pass `metrics: [{type: uv-script, kwargs: {script_path: datasets/4dcodebench/metric.py}}]`, as the configs do, so that aggregation skips the keys a case does not report.

## Usage: Create Task Directories

```bash
cd src/4dcodebench
uv sync

# all 200 cases from a 4DCodeBench checkout whose cases/ and data/ were filled by
# its own scripts/download_data.py --estimates
uv run 4dcodebench --repo /path/to/4DCodeBench --output-dir ../../datasets/4dcodebench

# one case, data kept outside the checkout
uv run 4dcodebench --repo /path/to/4DCodeBench \
    --cases-root /data/4dcb/cases --data-root /data/4dcb/data \
    --task-ids synthetic/B_01 --output-dir ../../datasets/4dcodebench --overwrite
```

| Flag | Meaning |
|---|---|
| `--repo` | a 4DCodeBench checkout (read only): agent context and commit |
| `--cases-root`, `--data-root` | default `<repo>/cases`, `<repo>/data` (the upstream download layout) |
| `--split` | `all` (default), `synthetic`, `real` |
| `--task-ids` | task names (`synthetic-b-01`), `<kind>/<case>` or bare case names |
| `--limit`, `--overwrite` | standard adapter flags |
| `--agent-image` | default `4dcb/agent:dev` |
| `--scorer-image` | default `4dcb/scorer-checkpoints:dev` |
| `--agent-network`, `--agent-allowed-host` | `public` (default, as upstream) or `allowlist` |
| `--skip-incomplete` | skip cases with missing reference data instead of failing |

A case whose reference data is incomplete stops generation and lists the missing files.

## Comparison with Original Benchmark (Parity)

Parity has **not** been run. Agents, models, the subset and the number of runs are still to be agreed with the Harbor team.

| Agent | Model | Metric | Number of Runs | Dataset Size | Original Performance | Harbor Performance |
|-------|-------|--------|----------------|--------------|----------------------|--------------------|
| TBD | TBD | TBD | TBD | 200 (100%) | TBD | TBD |

The synthetic oracle check compares the original scorer's `--sanity` readings (the reference world against itself) with the Harbor verifier's readings on the same world, case by case:

```bash
# from the adapters repository root

# original side: the reference world scored by the upstream scorer container
bash src/4dcodebench/scripts/run_original_sanity.sh \
    /data/4dcb/cases /data/4dcb/data /data/4dcb/checkpoints synthetic/B_01 sanity-1

# Harbor side: the oracle on the same case
harbor trial start -p datasets/4dcodebench/synthetic-b-01 -a oracle --trials-dir trials --trial-name b01-oracle-1

# metric-by-metric comparison of the two results/ directories
(cd src/4dcodebench && uv run 4dcodebench-compare \
    --original /data/4dcb/data/synthetic/B_01/sanity-1/results \
    --harbor ../../trials/b01-oracle-1/verifier/4dcodebench/results)
```

The scorer merges into an existing `reward.json`, so each original run takes a new run name (the fifth argument; default `sanity`, as upstream `--sanity`).

The original side needs one addition. The scorer's world gate requires `solution/build.sh` beside the scored world, and the released data has none. Without it, the original `--sanity` run marks every geometry metric `error`. `run_original_sanity.sh` therefore places the same placeholder `build.sh` beside the reference world in the data root. The upstream repository is not touched.

## Notes & Caveats

- **Validation status.** This version has passed static checks, unit tests with fixtures, and Harbor's task-config validation. None of the following has run yet:
  - the Docker images;
  - the GPU scorer inside Harbor;
  - the synthetic oracle;
  - parity.
- **The oracle is not a 100% reward.**
  - The synthetic oracle's score is the case's per-metric ceiling: the reference world against itself. Several metrics are not expected to be 1. For example, the geometry-quality metrics of open cloth or shell meshes are below 1 by construction.
  - No metric is normalised or thresholded to make the oracle pass.
  - Validation means the Harbor readings match the original `--sanity` readings case by case.
- **The oracle's `build.sh` is a placeholder.**
  - It exists only because the world gate checks that the file is there.
  - It states that it is not the program that produced the reference world, and it exits 1.
  - The authors' simulation programs are not released.
- **Real-world oracle: unresolved.** The real split has no reference 4D world. Its `solve.sh` exits 1, and no reconstruction is fabricated. Accepting it needs a separate decision with the Harbor team.
- **Primary reward: unresolved.**
  - The scorer writes per-metric readings only. The paper's Overall score (the average of the five families) is not implemented upstream, so the adapter forms no combined score.
  - It also adds no pass/fail threshold.
  - Which key, if any, Harbor Index should treat as primary is open.
- **Mapping non-numeric readings.**
  - `error` becomes the worst value, following `docs/spec.md` ("When computing aggregates, `error` is assigned the metric's worst possible value").
  - The upstream viewer instead averages metrics over executable runs only. The raw readings in `4dcodebench_readings.json` support either aggregation.
- **`executable` differs slightly from upstream.**
  - Harbor's `executable` is the gate verdict.
  - The upstream viewer's executability also requires the agent CLI's exit status to be 0. Harbor records agent failures separately.
- **`metric.py` aggregates over trials, not per case.** It equals the original per-case-then-mean whenever every case contributes the same number of trials.
- **Agent network.**
  - Upstream gives the agent the default (public) network, which this adapter keeps.
  - The synthetic reference worlds are public on Hugging Face, so an agent with internet access could fetch its case's answer, upstream and here alike.
  - `--agent-network allowlist` restricts the agent to its inference host. That would be a deviation that parity must apply on both sides.
- **No leakage through packaging.**
  - The agent image is built from `environment/` only: the task context and the video.
  - Reference worlds, annotations, estimates and checkpoints are only in the verifier image (`tests/`) and in `solution/`, which Harbor uploads only for the oracle agent.
- **Licensing and redistribution.**
  - 23 of the 100 real videos are not redistributed by the authors. They are rebuilt locally by the dataset's `prepare_videos.py`, and generated tasks contain them.
  - The Hugging Face datasets declare no licence.
  - Both must be settled before publishing the dataset to the registry.
- **Dataset size.** Synthetic reference worlds occupy about 65 GB once unpacked (upstream README); B_01 alone is about 11 MB. Generation hard-links them into `tests/bench/` and `solution/world/` when the filesystem allows.
- **Unpinned agent CLIs.** The upstream agent image installs the Claude Code and Codex CLIs unpinned. Parity must pin the agent versions on both sides.

## Installation / Prerequisites

1. **Harbor and uv:** the Harbor CLI (`uv tool install harbor`) and `uv`.
2. **Upstream checkout and data.** A 4DCodeBench checkout, with its data downloaded by its own script, into the checkout or elsewhere:

   ```bash
   python scripts/download_data.py --estimates
   ```

   Synthetic reference worlds occupy about 65 GB once unpacked, and the estimates 53 GB. For a few cases only, `scripts/fetch_case.py` downloads the same files from the same datasets at pinned revisions, checks their SHA-256 against the Hub, and unpacks worlds with the upstream `unpack`:

   ```bash
   # from src/4dcodebench; about 13 MB for B_01
   uv run scripts/fetch_case.py --repo /path/to/4DCodeBench --root /data/4dcb synthetic/B_01
   ```
3. **Checkpoints** for the scorer, about 6.7 GB. They require approved access to the gated `facebook/dinov3-vitl16-pretrain-lvd1689m` on Hugging Face:

   ```bash
   # in the 4DCodeBench checkout
   scripts/download_checkpoints.sh /data/4dcb/checkpoints
   ```

4. **Images.** A Linux host with an NVIDIA GPU, a driver that supports CUDA 13, Docker and the NVIDIA Container Toolkit. Build the images once:

   ```bash
   # from the adapters repository root
   bash src/4dcodebench/scripts/build_images.sh /path/to/4DCodeBench /data/4dcb/checkpoints
   ```

   This builds `4dcb/sim-base:dev`, `4dcb/agent:dev` and `4dcb/scorer:dev` with the upstream Dockerfiles, then `4dcb/scorer-checkpoints:dev` from `docker/scorer-checkpoints.Dockerfile`. Without the checkpoint directory it builds the upstream three only.

5. **Host check.** `scripts/vm_preflight.sh <4DCodeBench checkout> <data home> <checkpoint dir>` checks the tools, Docker and GPU access, the upstream commit, the data checksums, the checkpoints and the images, without changing anything. `VM_HANDOFF.md` gives the full setup and B_01 validation procedure for a fresh GPU host.

## Troubleshooting

- **`reference data missing`** during generation: download the data with `download_data.py --estimates`, or pass `--skip-incomplete`.
- **The verifier fails with `scorer checkpoints missing`:** the verifier was built `FROM` a scorer image without `/checkpoints`. Build `4dcb/scorer-checkpoints:dev`, or pass `--scorer-image`.
- **A verifier failure with no `reward.json`:** the scorer crashed or timed out. See `verifier/4dcodebench_verifier.log` and `4dcodebench_failure.txt` in the trial directory. This is reported as an infrastructure error, not as a score of 0.
- **Every geometry metric is `error`:** look at `world_gate` in `verifier/4dcodebench/results/reward.detail.json`. A missing `solution/build.sh` or a checker violation closes the world gate.

## Citation

```bibtex
@article{shen20264dcodebench,
  title={{4DCodeBench}: Benchmarking Agents on Inverse Graphics of Dynamic Scenes},
  author={Shen, Ruihong and Kova{\v{c}}i{\v{c}}, {\v{Z}}iga and Kulits, Peter and Wang, Xingrui and Li, Zizhang and Tenenbaum, Joshua B. and Yuille, Alan and Chen, Jieneng and Wu, Jiajun},
  journal={arXiv preprint arXiv:2610.03715},
  year={2026}
}
```

## Authors & Contributions

- Adapter: TODO (name and contact).
- Benchmark: the 4DCodeBench authors listed in the citation.
- Questions about the adapter go to the adapters repository; questions about the benchmark go upstream.
