# PaperEnvBench

PaperEnvBench 是面向 AI 论文代码仓环境自动配置的 benchmark。它不以“仓库能 import”为主成功标准，而以最小可验证复现为主目标：agent 需要生成安装方案、执行环境配置、处理失败、产出 artifact，并由隐藏评测器判定 L0–L4 分数。

## 当前状态

本目录目前是 benchmark 构建版，不是完整数据集发布版。`registries/task_registry.yaml` 已固化 50 个有效主任务，Dev / Val / Test 数量为 20 / 10 / 20。主 registry 只记录有效任务和 gold 构建状态。

当前已补齐 27 个标准 gold 任务包：音频 / 语音类 5 题、图像分类 / 表征学习类 5 题、检测 / 分割类 6 题、视频理解 / 视频生成类 5 题、视觉语言 / 多模态类 6 题。下一批按 LLM / NLP 6 题推进，受控保持每轮最多 5 个 SubAgent。

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
      taxonomy.yaml
      assets_manifest.yaml
      failure_tags.yaml
      scoring.yaml
      README_eval.md
```

完整任务发布时还需要补齐：

```text
gold_install.sh
environment.yaml
requirements_lock.txt
verify.py
expected_output.json
logs/gold_install.log
logs/gold_verify.log
artifacts/expected_artifact.*
```

## 数据边界

Gold solution、hidden verifier 和 Test trajectory 不得进入 agent 上下文。Dev trajectory 可以用于 Skill-ICOPD 的 on-policy Skill 提炼；Val 只用于阈值和路由策略选择；Test 只用于最终报告。

## 本地校验

```bash
python3 tools/paper_repo_env/validate_task_package.py
```

该命令验证 50 个主任务的 split / taxonomy 配额，并检查全部 ready gold 任务包的标准文件、YAML / JSON、`gold_install.sh` 语法和 `verify.py` 编译。
