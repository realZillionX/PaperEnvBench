# PaperEnvBench

> A benchmark for AI paper-repository environment reproduction.

PaperEnvBench 面向一个很具体、也很真实的问题：给定一篇 AI 论文的代码仓库，agent 是否能在受控环境中读懂仓库、配置依赖、修复环境失败，并产出一个可以被 verifier 检查的最小复现 artifact？

它刻意不把“仓库能 import”当作主成功标准。PaperEnvBench 的主目标是最小可验证复现：agent 必须走过仓库理解、环境安装、核心导入、最小入口运行和语义 artifact 验证，最终由 evaluator 给出 L0 到 L4 的层级结果与连续分数。

本仓库是独立 benchmark 仓库。它不 vendor 上游论文源码；每个任务通过固定 commit、gold 安装脚本、requirements lock、日志、expected artifact、verifier 和 registry 记录定义复现边界。

## Current Snapshot

| Item | Status |
| --- | --- |
| Benchmark version | `0.1.0` |
| Main tasks | `50` |
| Splits | Dev `20`，Val `10`，Test `20` |
| Modalities | `10` 类 |
| Gold task packages | `50 / 50` |
| Unified verifier CLI | `python3 verify.py --check-only --json` |
| Evaluator | `paperenvbench.evaluator` |
| Attempt toolchain | `repo_profile.json -> install_plan.json -> failure_report.json -> score.json` |

Task modality distribution：

| Modality | Count |
| --- | ---: |
| Audio / Speech | 5 |
| Image Classification / Representation | 5 |
| Object Detection / Segmentation | 6 |
| Video Understanding / Generation | 5 |
| Vision-Language / Multimodal | 6 |
| LLM / NLP | 6 |
| Diffusion / Generation | 5 |
| Graph Learning / Recommender | 4 |
| Reinforcement Learning / Simulation | 4 |
| Scientific / Geometry / 3D | 4 |

## Why This Exists

AI paper repositories often fail for reasons that are invisible in ordinary package tests：

- README commands are stale。
- PyTorch、CUDA、torchvision、torchaudio 版本矩阵漂移。
- demo 入口不清楚，只有完整训练路线。
- checkpoint 链接失效、巨大、需要认证，或路径约定不稳定。
- 系统库缺失，例如 `ffmpeg`、`libsndfile`、`libGL`、`cmake`。
- import 成功，但没有证据证明核心功能真的跑通。

PaperEnvBench 把这些失败机制结构化为可评测任务，使 agent 的环境复现能力可以被比较、分析和改进，而不是只留下“我好像跑通了”的手工日志。

## Benchmark Design

每个任务对应一个真实 AI 论文代码仓库。任务输入包括：

- `repo_url` 和固定 commit。
- agent-visible 仓库文件与任务提示。
- 平台限制，例如 CPU / GPU、Docker 禁止、网络和资产策略。
- 允许工具与安全边界。

agent 输出包括：

- `repo_profile.json`：仓库画像和 taxonomy hints。
- `install_plan.json`：安装、修复和验证计划。
- `attempt.log`：执行日志。
- `artifacts/`：最小复现 artifact。
- `trajectory.json`：agent 行为轨迹。
- `failure_report.json`：失败归因。
- `score.json`：evaluator 输出。

评测侧保留：

- gold install script。
- expected output。
- artifact checksum。
- task-local verifier。
- failure tags。
- scoring rubric。
- trajectory registry。

## Paper Environment Taxonomy

PaperEnvBench 使用三个互相独立的标注轴来描述任务：

| Axis | Purpose | Examples |
| --- | --- | --- |
| Modality | 任务所属研究领域 | audio、segmentation、video、diffusion、graph、3D |
| Failure mechanism | 环境失败原因 | `torch_cuda_matrix`、`checkpoint_download`、`native_extension_build` |
| Verification entry | 复现证据入口 | CLI help、import graph、single-sample inference、output artifact、semantic check |

完整 taxonomy 位于：

```text
paperenvbench/taxonomy.yaml
```

## Task Package Anatomy

每个任务目录都是一个可检查的 gold 包：

```text
paperenvbench/tasks/<task_id>/
  meta.yaml
  repo_snapshot.json
  taxonomy.yaml
  gold_install.sh
  requirements_lock.txt
  verify.py
  expected_output.json
  assets_manifest.yaml
  failure_tags.yaml
  scoring.yaml
  README_eval.md
  logs/
    gold_install.log
    gold_verify.log
  artifacts/
    expected_artifact.*
```

关键约定：

- `meta.yaml` 固定仓库、commit、split、硬件和可见性边界。
- `taxonomy.yaml` 描述模态、失败机制和验证入口。
- `gold_install.sh` 记录 gold 环境构建路线，不进入 Test agent 上下文。
- `verify.py` 是任务 verifier，统一支持 `--check-only --json`。
- `artifacts/expected_artifact.*` 是 packaged artifact，用于本地结构和语义 smoke。
- `scoring.yaml` 记录任务级评分重点或 required checks。

## Scoring

PaperEnvBench 同时报告层级结果和连续分数。

层级结果强调“必须逐级通过”：

| Level | Meaning | Required Evidence |
| --- | --- | --- |
| L0 | Repository analysis | 找到 README、依赖文件、入口和平台约束 |
| L1 | Environment install | 安装脚本完成，核心依赖存在 |
| L2 | Core import | 核心模块和关键依赖可导入 |
| L3 | Minimal entrypoint | help、demo、toy command 或 checkpoint loading 可运行 |
| L4 | Semantic artifact | 单样本 inference 或输出 artifact 通过 verifier |

连续分数用于部分分和误差分析。默认权重为：

$$
\operatorname{Score}=0.10s_{\mathrm{repo}}+0.20s_{\mathrm{install}}+0.20s_{\mathrm{import}}+0.25s_{\mathrm{entrypoint}}+0.20s_{\mathrm{semantic}}+0.05s_{\mathrm{safety}}
$$

`Level` 不由总分阈值直接切分。L4 必须先通过 L0 到 L3，再通过 semantic artifact 检查。Safety 不提升 `Level`，但严重违规可以限制最终等级。

## Quick Start

Clone and install：

```bash
git clone https://github.com/realZillionX/PaperEnvBench.git
cd PaperEnvBench
python3 -m pip install -e .
```

Run the structural validation：

```bash
python3 tools/paper_repo_env/validate_task_package.py
# ok registry=50 tasks=50
```

Run all packaged-artifact verifiers：

```bash
python3 tools/paper_repo_env/validate_task_package.py --run-verifiers
# ok registry=50 tasks=50
```

Run one task verifier directly：

```bash
cd paperenvbench/tasks/clip_zeroshot_minimal
python3 verify.py --check-only --json
```

## Evaluating an Agent Attempt

An attempt directory should look like this：

```text
attempt/
  repo_profile.json
  install_plan.json
  failure_report.json
  attempt.log
  artifacts/
  trajectory.json
```

Evaluate it：

```bash
python3 tools/paper_repo_env/evaluate_attempt.py \
  --task-id clip_zeroshot_minimal \
  --attempt-dir /path/to/attempt \
  --output /path/to/attempt/score.json
```

The generated `score.json` contains：

```json
{
  "task_id": "clip_zeroshot_minimal",
  "level": "L4",
  "score": 1.0,
  "dimensions": {
    "repo": 1.0,
    "install": 1.0,
    "import": 1.0,
    "entrypoint": 1.0,
    "semantic": 1.0,
    "safety": 1.0
  }
}
```

To also append a benchmark trajectory record：

```bash
python3 tools/paper_repo_env/evaluate_attempt.py \
  --task-id clip_zeroshot_minimal \
  --attempt-dir /path/to/attempt \
  --model <model_name> \
  --condition <condition_name> \
  --update-registry
```

## Toolchain

### 1. Inspect a Repository

```bash
python3 tools/paper_repo_env/inspect_repo.py /path/to/repo --output repo_profile.json
```

`repo_profile.json` records package metadata、entrypoints、dependencies、asset hints and taxonomy hints.

### 2. Build an Install Plan

```bash
python3 tools/paper_repo_env/build_install_plan.py \
  repo_profile.json \
  --task-id <task_id> \
  --output install_plan.json
```

`install_plan.json` converts repository observations into a reproducible plan：environment setup、system packages、Python dependencies、editable install and verification route.

### 3. Write a Failure Report

```bash
python3 tools/paper_repo_env/write_failure_report.py \
  --attempt-dir /path/to/attempt \
  --task-id <task_id>
```

`failure_report.json` records failed phase、failure tags、verifier error and next actions.

### 4. Score the Attempt

```bash
python3 tools/paper_repo_env/evaluate_attempt.py \
  --task-id <task_id> \
  --attempt-dir /path/to/attempt
```

This writes `score.json` by default.

## Registries

PaperEnvBench keeps benchmark state in explicit registries：

```text
paperenvbench/registries/task_registry.yaml
paperenvbench/registries/asset_registry.yaml
paperenvbench/registries/trajectory_registry.yaml
```

- `task_registry.yaml` is the canonical list of 50 tasks、splits、taxonomy labels and gold readiness。
- `asset_registry.yaml` records checkpoint、input and output artifact metadata。
- `trajectory_registry.yaml` records gold construction trajectories and evaluated agent attempts。

## Data and Leakage Boundaries

PaperEnvBench is designed for controlled agent evaluation. Keep these boundaries intact：

- Gold install scripts and hidden verifier details must not enter Test agent context。
- Dev trajectory can be used for experience extraction。
- Val is for threshold、prompt and experiment selection。
- Test is final-report only and must not feed back into skill construction。
- Secrets must be passed through temporary environment variables only。
- Docker may be read as documentation, but final benchmark solutions should use conda / venv / pip / shell routes unless a task explicitly permits otherwise。
- Algorithm patches must be distinguished from environment compatibility patches。

## Relation to ICOPD

PaperEnvBench is the benchmark substrate for ICOPD，In-Context On-Policy Distillation。ICOPD studies whether an agent can extract reusable procedural experience from its own Dev trajectories, then load that experience on future tasks without updating model parameters.

PaperEnvBench itself stays independent：it defines tasks、artifacts、verifiers、scores and registries. ICOPD consumes PaperEnvBench trajectories, but PaperEnvBench does not require ICOPD to run.

## Development Checks

Before committing benchmark changes：

```bash
python3 -m py_compile $(rg --files -g '*.py')
python3 tools/paper_repo_env/validate_task_package.py
python3 tools/paper_repo_env/validate_task_package.py --run-verifiers
git diff --check
```

Expected result：

```text
ok registry=50 tasks=50
ok registry=50 tasks=50
```

## Non-Goals

PaperEnvBench intentionally does not try to：

- fully reproduce paper-reported metrics for every task；
- vendor upstream repositories；
- accept import-only success as the main metric；
- make Docker the default final route；
- hide environment failures behind opaque scripts；
- mix Test outcomes back into future context experience。

The narrow promise is more useful：given a real AI paper repository, determine whether an agent can reach a minimal, auditable, semantically checked reproduction artifact under explicit constraints.
