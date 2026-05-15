# whisper_asr_minimal

该任务验证 agent 是否能从 Whisper 仓库发现 CLI 入口、安装音频系统依赖、处理 `tiny` checkpoint 下载，并对 1 秒合成音频运行最小 ASR。L4 成功不要求真实语音识别准确率，只要求命令执行成功、输出文件落盘、JSON 中包含可解析转录字段。
