# silero_vad_speech_activity

该任务验证 agent 是否能在 CPU 环境中安装 Silero VAD，发现 `load_silero_vad`、`read_audio` 和 `get_speech_timestamps` 入口，并对固定仓库中的小音频执行语音活动检测。

Gold 路线固定到 `snakers4/silero-vad` commit `980b17e9d56463e51393a8d92ded473f1b17896a`。验证使用仓库自带的 `tests/data/test.wav` 和包内 `src/silero_vad/data/silero_vad.jit`，因此不依赖 `torch.hub` 隐式下载。若 agent 改用 `torch.hub`，必须记录 hub cache 路径和任何下载失败原因。

L4 成功要求生成 `speech_activity_summary.json`，其中 speech timestamp 非空、总语音时长为正、VAD 最大概率超过阈值，并且仓库 commit 与任务固定 commit 一致。不要求 GPU，不允许 Docker。
