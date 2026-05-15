# speechbrain_keyword_spotting

该任务验证 agent 是否能把 SpeechBrain 仓库固定到指定 commit，选择 CPU-only PyTorch wheel，安装关键词识别所需依赖，并通过 `speechbrain/google_speech_command_xvector` 的 pretrained interface 对单个合成音频做最小推理。

L4 成功不要求完整 Google Speech Commands 训练，也不要求合成正弦音频有真实语义，只要求模型 checkpoint 成功加载、`EncoderClassifier.classify_file()` 在 CPU 上完成、输出 JSON 中包含非空预测标签、类别数为 `12`、分数为有限数，并生成 `artifacts/expected_artifact.json` 与 `artifacts/expected_artifact.wav`。

已验证远端环境：

- Notebook：`paper-repro-ci-prep`
- 远端目录：`paper-repro:runs/paperenvbench/audio/speechbrain_keyword_spotting`
- Python：`3.12.3`
- Torch：`2.11.0+cpu`
- SpeechBrain commit：`8a89ebad72af734b75bbd37565ae96a6819e146b`
- Hugging Face model revision：`b0cec0fb42423936ca0da2724ce52d82eb807e20`

复现命令：

```bash
bash gold_install.sh
. .venv/bin/activate
python verify.py
```
