# 在 Mac 上预检 QAT 输入

用已有本地浮点快照和训练清单检查 QAT，不会加载模型、导入 Torch、启动训练或下载权重。

```sh
python3 -m quantdoctor qat-validate qat.json --snapshot /path/to/Qwen3-ASR-0.6B
python3 -m quantdoctor qat-validate qat.json --snapshot /path/to/Qwen3-ASR-0.6B --port 8765
```

省略 `--snapshot` 时只检查配方格式。有快照时会核对 Qwen3-ASR-0.6B 配置、选定 decoder 投影的实际 dtype 与 shape、训练和校准的音频摘要、token 范围、音频占位及清单隔离。16 kHz 和最长 20 秒的检查覆盖本机可读取的 PCM WAV 头。其他编码与完整音频解码由 Worker 检查。结果中的 `training_verified` 与 `deployment_verified` 仍为 false。 [Qwen3-ASR-0.6B 官方配置](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/main/config.json)

可在配方里增加 `"evaluation_manifest": "evaluation.json"`。使用 [ASR 数据集清单](DATASETS.md)格式，至少包含 evaluation split。预检检查留出集与校准、训练是否共享样本 ID、音频内容或已标记的说话人。未提供留出集或部分样本没有说话人标签时会显示警告。报告包含各 split 数量与摘要。字段 `model_revision` 来自配方声明，权重内容的完整快照指纹由 Worker 在执行前计算。

CUDA Worker 会复用相同的输入检查，随后由实际模型加载、SoundFile 解码和训练步骤继续验证。评测清单只用于隔离检查与记录摘要，训练只使用校准和训练清单。板端验收需对同一留出集比较同配方 PTQ 与 QAT。
