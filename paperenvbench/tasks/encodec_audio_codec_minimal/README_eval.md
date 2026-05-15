# encodec_audio_codec_minimal

该任务验证 agent 是否能在 CPU Notebook 中复现 EnCodec 的最小音频编解码路径：固定 `facebookresearch/encodec` commit，安装 Python 包依赖，加载 `24 kHz` 预训练 checkpoint，对 1 秒合成正弦波执行 encode / decode，并输出 summary JSON 与重构 WAV。

远端 gold 路径已在启智 CPU Notebook `paper-repro-ci-prep` 中验证，运行目录为 `paper-repro:runs/paperenvbench/audio/encodec_audio_codec_minimal`。验收日志保存在 `logs/gold_install.log` 与 `logs/gold_verify.log`。

复现命令：

```bash
bash gold_install.sh 2>&1 | tee logs/gold_install.log
python verify.py 2>&1 | tee logs/gold_verify.log
```

重要边界：

- 不使用 Docker。
- 仓库必须克隆到 `src/encodec`，不能克隆到任务根目录的 `encodec/`，否则会遮蔽已安装包。
- `verify.py` 使用 Python 标准库 `wave` 写入 WAV，避免把 `torchcodec` 变成额外必需依赖。
- 首次验证会从 `https://dl.fbaipublicfiles.com/encodec/v0/encodec_24khz-d7cc33bc.th` 下载 checkpoint，网络或缓存不可用时应标记为 `checkpoint_download` 阻塞。
