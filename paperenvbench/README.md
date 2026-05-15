# PaperEnvBench

PaperEnvBench 是面向 AI 论文代码仓环境自动配置的 benchmark。它不以“仓库能 import”为主成功标准，而以最小可验证复现为主目标：agent 需要生成安装方案、执行环境配置、处理失败、产出 artifact，并由隐藏评测器判定 L0–L4 分数。

## 当前状态

本仓库是 PaperEnvBench 的独立 Git 仓库，远程为 `https://github.com/realZillionX/PaperEnvBench.git`。父级研究工作区不属于本仓库发布范围。

`paperenvbench/registries/task_registry.yaml` 已固化 50 个有效主任务，Dev / Val / Test 数量为 20 / 10 / 20。主 registry 只记录有效任务和 gold 构建状态，不记录拒绝候选或临时调研信息。

当前已补齐 50 个标准 gold 任务包：音频 / 语音类 5 题、图像分类 / 表征学习类 5 题、检测 / 分割类 6 题、视频理解 / 视频生成类 5 题、视觉语言 / 多模态类 6 题、LLM / NLP 类 6 题、Diffusion / Generation 类 5 题、Graph / Recommender 类 4 题、Reinforcement Learning / Simulation 类 4 题、Scientific / Geometry / 3D 类 4 题。

当前基线验证：

```bash
python3 tools/paper_repo_env/validate_task_package.py
# ok registry=50 tasks=50
```

50 个任务的 `verify.py --check-only` sweep 已全部通过。

## 目录约定

```text
paperenvbench/
  taxonomy.yaml
  registries/
    task_registry.yaml
    experiment_registry.yaml
    skill_registry.yaml
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
```

本仓库不 vendor 上游论文仓库源码；每个任务通过固定 commit、安装脚本、requirements lock、日志、expected artifact 和 verifier 描述可验证复现边界。部分任务的 `--check-only` verifier 使用已纳入仓库的 gold artifact 做 artifact-level 验证，clean-room rebuild 由后续 hidden evaluator runner 统一编排。

## 数据边界

Gold solution、hidden verifier 和 Test trajectory 不得进入 agent 上下文。Dev trajectory 可以用于 Skill-ICOPD 的 on-policy Skill 提炼；Val 只用于阈值和路由策略选择；Test 只用于最终报告。

## 本地校验

```bash
python3 tools/paper_repo_env/validate_task_package.py
```

该命令验证 50 个主任务的 split / taxonomy 配额，并检查全部 ready gold 任务包的标准文件、YAML / JSON、`gold_install.sh` 语法和 `verify.py` 编译。
