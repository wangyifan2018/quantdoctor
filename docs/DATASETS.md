# 自定义 Qwen3-ASR 校准集与评测集

## Edge-LLM 的默认行为

本项目对应的 Edge-LLM checkout 为 `1231d6415ce38d0b2f20b3308ac24e58ec4100a8`。Qwen3-ASR 的 PTQ 默认选择 `librispeech`，从 Hugging Face `openslr/librispeech_asr` 流式读取 `clean/train.100` 的音频与转写。即便只做 LLM NVFP4、Audio 保持 FP16，也会走联合音频校准。当前分支最多接受 `min(num_samples,128)` 条，过滤非 16 kHz、超过 20 秒、无转写、无法解码或不符合音频占位契约的样本。 [默认数据源](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/1231d6415ce38d0b2f20b3308ac24e58ec4100a8/tensorrt_edgellm/quantization/datasets/builtin.py#L105) · [ASR 联合校准](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/1231d6415ce38d0b2f20b3308ac24e58ec4100a8/tensorrt_edgellm/quantization/quantize.py#L1185)

有三个定制入口：

- Python API 的 `audio_dataset` 接受一个无参 generator callable，逐条返回 `(audio_bytes, transcript)`。QuantDoctor 使用这个入口，读取本地文件。
- CLI 的 `--audio_dataset NAME` 选择已注册的名称。可以在同一个 Python 进程注册自己的生成器，再调用 CLI 入口，或通过 Python API 直接传 callable。官方 v0.11.1 统一 CLI 对应参数拼写为 `--audio-dataset`。
- `EDGELLM_QUANT_DATASET_LIBRISPEECH` 可将内置数据源指向本地 JSON/JSONL 或兼容的 Hugging Face Dataset 目录。仍需符合内置 loader 的 `audio` 与 `text` 字段契约。QuantDoctor 的 split 清单使用自己的格式，经 callable 交给 Edge-LLM。

[NVIDIA 校准集定制指南](https://nvidia.github.io/TensorRT-Edge-LLM/0.11.1/developer_guide/customization/calibration-datasets.html)

## 本地清单

在同一个 `dataset.json` 中标出角色。路径相对于清单所在目录。将音频放在该目录下，例如 `audio/`。

```json
{
  "schema_version": 1,
  "samples": [
    {"id": "cal-001", "split": "calibration", "audio": "audio/cal-001.wav", "text": "Open the window.", "language": "en"},
    {"id": "eval-001", "split": "evaluation", "audio": "audio/eval-001.wav", "text": "打开右侧车窗。", "language": "zh"}
  ]
}
```

`dataset-validate` 检查文件、唯一 ID 和跨 split 的相同音频文件内容，计算每个 split 的 SHA-256。摘要包含有序样本 ID、音频文件内容摘要、转写、语言和可选的说话人 ID，所以换存放路径不改变数据集身份，换样本顺序会改变身份。它能查出改名后的相同文件。重新编码的同源录音需要在准备清单时控制。为行添加可选的 `speaker_id` 可检查说话人隔离。两边使用同一套去标识的 ID，即使文件内容不同，出现重复说话人也会拒绝。缺 ID 时报告覆盖不完整。用 `dataset-validate --require-speakers` 要求每行有 ID。ID 只进入评分清单，不传给推理请求。清单核对的是声明 ID，无法自动识别同一人的声音。也可用单 split 清单，跨 split 检查要求将两套样本放在同一个清单。

校准接受 PCM WAV 和 FLAC。WAV 的采样率和时长在本机检查，FLAC 的采样率、时长与实际解码由设备侧校准 loader 确认。评测请求使用小写后缀 `.wav`、`.flac`、`.mp3`，对应 Edge-LLM 的扩展名检查。设备推理前仍须确认实际音频文件、音频 profile 及最大输入长度。

```sh
python3 -m quantdoctor dataset-validate /data/asr/dataset.json
```

## 自定义校准

先在 Mac 上用合并清单检查校准与评测交集。可以将完整清单和引用的文件复制到 CUDA Worker，也可以仅复制校准行和对应音频，保留校准行顺序与相对路径。在安装了兼容 Edge-LLM 与 QuantDoctor 的环境执行：

```python
from quantdoctor.datasets import calibration_source
from tensorrt_edgellm.quantization import quantize_and_export

source = calibration_source("/data/asr/dataset.json")
quantize_and_export(
    model_dir="/data/models/Qwen3-ASR-0.6B",
    output_dir="/data/runs/qwen3-asr-nvfp4-custom-cal",
    quantization="nvfp4",
    audio_dataset=source,
    num_samples=128,
)
```

`source.calib_name` 包含校准集摘要，便于和 Edge-LLM 日志核对。实际接收样本数以校准日志为准。本地校验和 generator 已通过 fixture 测试，尚未在 CUDA Worker 执行 PTQ。QAT 仍使用带 token IDs 的独立训练合同。

**中文校准的版本限制**。当前 ASR loader 把 `language English` 写在校准转写前面。传入中文数据会改变音频与转写分布，但保留英语前缀。QuantDoctor 会提示这项差异。需要按样本语言设置前缀时，后续通过独立校准 forward-loop 适配器接入并做设备验证。中文评测可直接使用独立评测 split。 [当前校准 prompt](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/1231d6415ce38d0b2f20b3308ac24e58ec4100a8/tensorrt_edgellm/quantization/qwen3_asr_loader.py#L154)

## 自定义评测

准备 `comparison-metadata.json`。格式参考 [演示对照](../examples/demo-comparison.json) 的 `schema_version`、`model`、`baseline`、`candidate`。填写两次运行的模型 revision、有效精度、设备环境与解码设置，真实设备结果使用 `source: "target"`。评分清单中的 `samples` 由工具生成。

```sh
python3 -m quantdoctor prepare-eval /data/asr/dataset.json comparison-metadata.json \
  --audio-root /target/data/asr --output eval-bundle
```

输出包括 `input.json` 和 `manifest.json`。前者仅包含评测音频请求，使用 batch 1、greedy top-k 1。后者含 `request_idx` 与参考转写，以及评测集和规范化请求 JSON 的摘要。 `--audio-root` 指向目标板上与清单目录具有相同子目录结构的位置，文件需另行同步，音频内容需与校验时一致。省略时使用本机路径。两组推理都用同一 `input.json`，避免命令行覆盖解码参数。生成请求格式对应本地 checkout 的 `llm_inference`。 [NVIDIA ASR 推理输入](https://nvidia.github.io/TensorRT-Edge-LLM/0.11.0/user_guide/examples/asr.html)

回传两份 `--outputFile` 结果后，导入 QuantDoctor：

```sh
python3 -m quantdoctor import-edge eval-bundle/manifest.json fp16-output.json nvfp4-output.json \
  --requests eval-bundle/input.json -o comparison.json
python3 -m quantdoctor serve comparison.json
```

对比校准数据效果时，固定模型、量化配置和设备，只把候选的校准集从默认 LibriSpeech 换成自定义集。将这两份结果在同一独立评测集上与 FP16 基线配对，优先查看之前的空转写样本和整体 CER/WER。评测集和请求摘要保留在 QuantDoctor 报告及修复工单中。

## 跨数据集核对请求设置

```sh
python3 -m quantdoctor request-audit diagnosis/input.json --against holdout/input.json
```

两批录音通常有不同的完整请求摘要。对共享同一消息模板和请求选项的单音频 ASR 请求，QuantDoctor 另计算一个跨数据集的设置摘要。摘要忽略音频路径和录音数量，保留消息角色、顺序、提示文本和逐请求选项。提示、停止字符串和其他复杂设置以摘要保存。不同设置会列出字段名，例如 `max_generate_length`、`requests.stop` 或 `requests.messages`。混合模板、图像和预计算 Mel 文件会报告 `unsupported`。该命令的 JSON 状态用于检查差异。

标准标量缺省值按固定 checkout 的请求解析器填充，例如 `batch_size=1`、`max_generate_length=256`、`top_k=50` 和 `top_p=0.8`。`prepare-eval` 明确写入 greedy 的 `top_k=1`、`top_p=1.0`。`import-edge --requests` 先核对完整请求摘要与索引，再附加设置摘要。`import-worker-edge` 从已经核对过字节摘要的工单输入文件自动提取。旧结果仍可导入，缺少文件证据时留出集复验只检查运行声明。

文件审计范围包括请求 JSON。`llm_inference` 的 `--batchSize`、`--maxGenerateLength` 和 `--numLogprobs` 可覆盖文件。运行命令、实际加载的引擎和 tokenizer 模板仍需设备记录核对。 [文件默认值与覆盖](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/1231d6415ce38d0b2f20b3308ac24e58ec4100a8/examples/utils/requestFileParser.cpp#L229-L294) · [音频输入分支](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/1231d6415ce38d0b2f20b3308ac24e58ec4100a8/examples/utils/requestFileParser.cpp#L479-L520) · [逐请求设置](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/1231d6415ce38d0b2f20b3308ac24e58ec4100a8/examples/utils/requestFileParser.cpp#L540-L586) · [运行入口](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/1231d6415ce38d0b2f20b3308ac24e58ec4100a8/examples/llm/llm_inference.cpp#L1512-L1518)
