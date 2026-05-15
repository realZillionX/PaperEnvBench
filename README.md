# PaperEnvBench

PaperEnvBench 是面向 AI 论文代码仓环境自动配置的 benchmark。它不以“仓库能 import”为主成功标准，而以最小可验证复现为主目标：agent 需要分析仓库、生成安装方案、执行环境配置、处理失败、产出 artifact，并由 evaluator 判定 L0–L4 分数。

本仓库是独立 Git 仓库，父级研究工作区不属于发布范围。仓库不 vendor 上游论文源码；每个任务通过固定 commit、gold 安装脚本、requirements lock、日志、expected artifact 和 verifier 描述可验证复现边界。

## Status

- 50 个主任务已固化，Dev / Val / Test 为 20 / 10 / 20。
- 任务覆盖 10 类模态：音频 / 语音、图像分类 / 表征学习、检测 / 分割、视频理解 / 生成、视觉语言 / 多模态、LLM / NLP、Diffusion / Generation、Graph / Recommender、Reinforcement Learning / Simulation、Scientific / Geometry / 3D。
- 50 个任务均包含标准 gold 包：`meta.yaml`、`taxonomy.yaml`、`gold_install.sh`、`requirements_lock.txt`、`verify.py`、`expected_output.json`、`scoring.yaml`、日志和 `artifacts/expected_artifact.*`。
- 所有 task verifier 已统一支持 `--check-only --json`。
- `paperenvbench.evaluator` 可对 agent attempt 目录生成 `score.json`，并可选择更新 `paperenvbench/registries/trajectory_registry.yaml`。
- `repo_profile.json -> install_plan.json -> failure_report.json -> score.json` 工具链已固定。

## Layout

```text
paperenvbench/
  taxonomy.yaml
  evaluator.py
  registries/
    task_registry.yaml
    trajectory_registry.yaml
    asset_registry.yaml
  tasks/
    <task_id>/
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
tools/paper_repo_env/
  inspect_repo.py
  build_install_plan.py
  write_failure_report.py
  evaluate_attempt.py
  validate_task_package.py
```

## Validation

Run the structural check:

```bash
python3 tools/paper_repo_env/validate_task_package.py
# ok registry=50 tasks=50
```

Run the full packaged-artifact verifier sweep:

```bash
python3 tools/paper_repo_env/validate_task_package.py --run-verifiers
# ok registry=50 tasks=50
```

The verifier sweep executes each task from its own task directory as:

```bash
python3 verify.py --check-only --json
```

## Attempt Contract

An agent attempt directory should contain these files when available:

```text
repo_profile.json
install_plan.json
failure_report.json
attempt.log
artifacts/
trajectory.json
score.json
```

The evaluator accepts partial attempts, but L4 requires a verifier-accepted artifact bundle.

## Toolchain

Generate a repository profile:

```bash
python3 tools/paper_repo_env/inspect_repo.py /path/to/repo --output repo_profile.json
```

Build an installation plan:

```bash
python3 tools/paper_repo_env/build_install_plan.py repo_profile.json --task-id <task_id> --output install_plan.json
```

Evaluate an attempt:

```bash
python3 tools/paper_repo_env/evaluate_attempt.py \
  --task-id <task_id> \
  --attempt-dir /path/to/attempt \
  --output /path/to/attempt/score.json
```

Create or refresh a failure report after a failed attempt:

```bash
python3 tools/paper_repo_env/write_failure_report.py \
  --attempt-dir /path/to/attempt \
  --task-id <task_id>
```

Update the trajectory registry only for attempts that should become benchmark records:

```bash
python3 tools/paper_repo_env/evaluate_attempt.py \
  --task-id <task_id> \
  --attempt-dir /path/to/attempt \
  --model <model_name> \
  --condition <condition_name> \
  --update-registry
```

## Scoring

The evaluator reports continuous sub-scores for repository analysis、install、import、entrypoint、semantic 和 safety。`Level` uses cumulative gates：L4 requires L0–L3 to pass and the semantic artifact check to pass. Safety does not raise `Level`，but severe violations can cap the final level.

## Data Boundary

Gold install scripts、hidden verifier behavior 和 Test trajectory must not enter agent context. Dev trajectory can be used for ICOPD experience extraction；Val is for prompt and threshold selection；Test is final-report only. Secrets must be passed through temporary environment variables and must not be written into logs、trajectory、registry or reports.
