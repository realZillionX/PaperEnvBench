# whisper_asr_minimal

该任务验证 agent 是否能从 Whisper 仓库发现 CLI 入口、安装音频系统依赖、处理 `tiny.en` checkpoint 下载，并对真实语音样例运行最小 ASR。该任务按原始实训参考文档的 Whisper 成功标准定义：必须跑通 `whisper --help`，并完成一次命令行转录或等价最小推理。

完成任务不能只证明安装成功，也不能使用无语义的合成正弦波冒充 ASR。标准复现必须在 4090 / CUDA runtime 中加载 `tiny.en` checkpoint，对 `tests/jfk.flac` 或同等真实短语音完成转录，生成 JSON、TXT 和至少一个字幕 / 表格 side artifact，并记录 repo commit、checkpoint SHA、输入音频 SHA、CUDA 设备、分段数和非空英文转录。
