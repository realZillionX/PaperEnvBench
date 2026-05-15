# PaperEnvBench

> 面向 AI 论文代码仓环境自动配置的可验证复现基准。

PaperEnvBench 关注一个具体、真实、经常被低估的问题：给定一篇 AI 论文的代码仓库，agent 是否能在受控环境中读懂仓库、配置依赖、修复环境失败，并产出一个可以被 verifier 检查的最小复现 artifact？

它不把“仓库能 import”当作主成功标准。PaperEnvBench 的主目标是最小可验证复现：agent 必须依次完成仓库理解、环境安装、核心导入、最小入口运行和语义 artifact 验证，最终由 evaluator 给出 L0 到 L4 的层级结果与连续分数。

本仓库是独立 benchmark 仓库。它不 vendor 上游论文源码；每个任务通过固定 commit、gold 安装脚本、requirements lock、日志、expected artifact、verifier 和 registry 记录定义复现边界。

## 当前状态

| 项目 | 状态 |
| --- | --- |
| Benchmark 版本 | `0.1.0` |
| 主任务数量 | `50` |
| 数据划分 | Dev `20`，Val `10`，Test `20` |
| 任务模态 | `10` 类 |
| Gold 任务包 | `50 / 50` |
| Verifier 统一接口 | `python3 verify.py --check-only --json` |
| Evaluator | `paperenvbench.evaluator` |
| Attempt 工具链 | `repo_profile.json -> install_plan.json -> failure_report.json -> score.json` |

任务模态分布：

| 模态 | 数量 |
| --- | ---: |
| 音频 / 语音 | 5 |
| 图像分类 / 表征学习 | 5 |
| 目标检测 / 分割 | 6 |
| 视频理解 / 视频生成 | 5 |
| 视觉语言 / 多模态 | 6 |
| LLM / NLP | 6 |
| Diffusion / Generation | 5 |
| Graph / Recommender | 4 |
| Reinforcement Learning / Simulation | 4 |
| Scientific / Geometry / 3D | 4 |

## 为什么需要这个基准

AI 论文仓库的环境复现失败，往往不是普通 package test 能暴露的问题：

- README 命令过期。
- PyTorch、CUDA、torchvision、torchaudio 版本矩阵漂移。
- demo 入口不清楚，只有完整训练路线。
- checkpoint 链接失效、体积巨大、需要认证，或路径约定不稳定。
- 系统库缺失，例如 `ffmpeg`、`libsndfile`、`libGL`、`cmake`。
- import 成功，但没有证据证明核心功能真的跑通。

PaperEnvBench 把这些失败机制结构化为可评测任务，使 agent 的环境复现能力可以被比较、分析和改进，而不是只留下“我好像跑通了”的手工日志。

## 基准设计

每个任务对应一个真实 AI 论文代码仓库。任务输入包括：

- `repo_url` 和固定 commit。
- agent 可见的仓库文件与任务提示。
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

| 标注轴 | 作用 | 示例 |
| --- | --- | --- |
| 任务模态 | 标注任务所属研究领域 | audio、segmentation、video、diffusion、graph、3D |
| 环境失败机制 | 标注环境失败原因 | `torch_cuda_matrix`、`checkpoint_download`、`native_extension_build` |
| 验证入口 | 标注复现证据入口 | CLI help、import graph、single-sample inference、output artifact、semantic check |

完整 taxonomy 位于：

```text
paperenvbench/taxonomy.yaml
```

## 任务包结构

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

## 评分方式

PaperEnvBench 同时报告层级结果和连续分数。

层级结果强调“必须逐级通过”：

| 层级 | 含义 | 需要的证据 |
| --- | --- | --- |
| L0 | 仓库理解 | 找到 README、依赖文件、入口和平台约束 |
| L1 | 环境安装 | 安装脚本完成，核心依赖存在 |
| L2 | 核心导入 | 核心模块和关键依赖可导入 |
| L3 | 最小入口 | help、demo、toy command 或 checkpoint loading 可运行 |
| L4 | 语义 artifact | 单样本 inference 或输出 artifact 通过 verifier |

连续分数用于部分分和误差分析。总分记为 $S$，默认计算方式为 $S=0.10s_{\mathrm{repo}}+0.20s_{\mathrm{install}}+0.20s_{\mathrm{import}}+0.25s_{\mathrm{entrypoint}}+0.20s_{\mathrm{semantic}}+0.05s_{\mathrm{safety}}$。

`Level` 使用累计门控，而不是用总分阈值直接切分。令 $s_0=s_{\mathrm{repo}}$、$s_1=s_{\mathrm{install}}$、$s_2=s_{\mathrm{import}}$、$s_3=s_{\mathrm{entrypoint}}$、$s_4=s_{\mathrm{semantic}}$；每个任务可以在 `scoring.yaml` 中定义阈值 $\alpha_0,\ldots,\alpha_4$。对层级 $L_\ell$，累计 gate 为 $G_\ell(\tau,\xi)=\prod_{r=0}^{\ell}\mathbf{1}[s_r(\tau,\xi)\ge\alpha_r]$，最终层级为 $\mathrm{Level}(\tau,\xi)=\max\{L_\ell\mid G_\ell(\tau,\xi)=1\}$。

因此，L4 必须先通过 L0 到 L3，再通过 semantic artifact 检查。Safety 不提升 `Level`，但严重违规可以限制最终等级。

## 快速开始

克隆并安装：

```bash
git clone https://github.com/realZillionX/PaperEnvBench.git
cd PaperEnvBench
python3 -m pip install -e .
```

运行结构校验：

```bash
python3 tools/paper_repo_env/validate_task_package.py
# ok registry=50 tasks=50
```

运行全部 packaged-artifact verifier：

```bash
python3 tools/paper_repo_env/validate_task_package.py --run-verifiers
# ok registry=50 tasks=50
```

直接运行单个任务 verifier：

```bash
cd paperenvbench/tasks/clip_zeroshot_minimal
python3 verify.py --check-only --json
```

## 评测 Agent Attempt

一个 attempt 目录建议包含：

```text
attempt/
  repo_profile.json
  install_plan.json
  failure_report.json
  attempt.log
  artifacts/
  trajectory.json
```

运行 evaluator：

```bash
python3 tools/paper_repo_env/evaluate_attempt.py \
  --task-id clip_zeroshot_minimal \
  --attempt-dir /path/to/attempt \
  --output /path/to/attempt/score.json
```

生成的 `score.json` 包含：

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

如果该 attempt 应成为 benchmark 记录，可以同时更新 trajectory registry：

```bash
python3 tools/paper_repo_env/evaluate_attempt.py \
  --task-id clip_zeroshot_minimal \
  --attempt-dir /path/to/attempt \
  --model <model_name> \
  --condition <condition_name> \
  --update-registry
```

## 工具链

### 1. 生成仓库画像

```bash
python3 tools/paper_repo_env/inspect_repo.py /path/to/repo --output repo_profile.json
```

`repo_profile.json` 记录 package metadata、entrypoints、dependencies、asset hints 和 taxonomy hints。

### 2. 生成安装计划

```bash
python3 tools/paper_repo_env/build_install_plan.py \
  repo_profile.json \
  --task-id <task_id> \
  --output install_plan.json
```

`install_plan.json` 将仓库观察转换为可复现计划：环境创建、系统依赖、Python 依赖、editable install 和验证路线。

### 3. 生成失败报告

```bash
python3 tools/paper_repo_env/write_failure_report.py \
  --attempt-dir /path/to/attempt \
  --task-id <task_id>
```

`failure_report.json` 记录失败阶段、failure tags、verifier error 和后续修复方向。

### 4. 生成评分结果

```bash
python3 tools/paper_repo_env/evaluate_attempt.py \
  --task-id <task_id> \
  --attempt-dir /path/to/attempt
```

默认写入 `score.json`。

## Registries

PaperEnvBench 用显式 registry 维护 benchmark 状态：

```text
paperenvbench/registries/task_registry.yaml
paperenvbench/registries/asset_registry.yaml
paperenvbench/registries/trajectory_registry.yaml
```

- `task_registry.yaml` 是 50 个任务、split、taxonomy labels 和 gold readiness 的唯一任务清单。
- `asset_registry.yaml` 记录 checkpoint、输入资产和输出 artifact 元数据。
- `trajectory_registry.yaml` 记录 gold construction trajectories 和已评测 agent attempts。

## 数据边界与泄漏控制

PaperEnvBench 面向受控 agent 评测，必须保持以下边界：

- Gold install scripts 和 hidden verifier details 不得进入 Test agent 上下文。
- Dev trajectory 可用于经验抽取。
- Val 只用于阈值、prompt 和实验设置选择。
- Test 只用于最终报告，不能回流到 skill 构建。
- Secrets 只能通过临时环境变量传入。
- Docker 可以作为 README 信息来源，但最终 benchmark solution 默认应走 conda / venv / pip / shell 路线，除非任务显式允许。
- Algorithm patch 必须与 environment compatibility patch 区分记录。

## 与 ICOPD 的关系

PaperEnvBench 是 ICOPD（In-Context On-Policy Distillation）的 benchmark 底座。ICOPD 研究 agent 能否从自己的 Dev trajectory 中抽取可复用过程经验，并在未来任务中按需加载这些经验，而不更新模型参数。

PaperEnvBench 本身保持独立：它定义任务、artifact、verifier、score 和 registry。ICOPD 可以消费 PaperEnvBench trajectory，但运行 PaperEnvBench 不依赖 ICOPD。

## 开发校验

提交 benchmark 改动前，建议运行：

```bash
python3 -m py_compile $(rg --files -g '*.py')
python3 tools/paper_repo_env/validate_task_package.py
python3 tools/paper_repo_env/validate_task_package.py --run-verifiers
git diff --check
```

期望结果：

```text
ok registry=50 tasks=50
ok registry=50 tasks=50
```

## 非目标

PaperEnvBench 有意不做这些事：

- 要求每个任务完整复现论文报告指标；
- vendor 上游论文仓库；
- 把 import-only success 当作主指标；
- 默认使用 Docker 作为最终路线；
- 用不透明脚本掩盖环境失败；
- 将 Test 结果回流到未来上下文经验中。

它的承诺更窄，也更可验证：给定一个真实 AI 论文仓库，判断 agent 能否在明确约束下达到最小、可审计、语义可检查的复现 artifact。
