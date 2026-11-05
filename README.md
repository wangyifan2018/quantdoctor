# QuantDoctor

本地 ASR 量化诊断和独立的选择性 QAT Worker。Python 3.11+ 的本机诊断功能仅依赖标准库。产品边界、接口与后续计划见 [产品架构](docs/ARCHITECTURE.md)。

## 实现顺序

1. `model.py` 定义运行与样本合同，`edge.py` 导入运行产物，`diagnostics.py` 负责分诊、配对评分与工单。CLI 和 Web 共用这些规则。本机测试通过模拟结果覆盖这条路径。
2. `qat.py` 定义可独立校验的配方和训练样本。 `qat_worker.py` 执行版本绑定的模型加载、校准、训练与导出。Worker 与 Web 之间用结果文件交换。
3. 第一轮设备接入时复现原始空转写，先验证单变量恢复，再用相同配方与数据对照 PTQ/QAT。拿到真实板端结果后，再扩展激活采集和逐层搜索。

## 本机工作台

在 `quantdoctor/` 目录执行。

```sh
python3 -m quantdoctor report examples/demo-comparison.json
python3 -m quantdoctor serve examples/demo-comparison.json
python3 -m quantdoctor plan examples/demo-comparison.json ASR-014
python3 -m unittest discover -s tests -v
```

在浏览器打开终端打印的 loopback 地址。示例为模拟数据。报告区分首步 EOS、EOS 位置未知、文本清理异常、运行错误和缺证据。工单是可审查的配置差异与复验条件，状态为待复验。CER/WER 为微平均，中文按字符、英文按词，统一 NFKC、casefold 和标点/空白规范化。缺结果和运行错误单列。空文本在有效配对中按删除计分。

已有 Edge-LLM 运行结果可这样导入。

```sh
python3 -m quantdoctor import-edge manifest.json fp16-output.json nvfp4-output.json -o comparison.json
python3 -m quantdoctor serve comparison.json
```

`manifest.json` 顶层字段与 [演示对照](examples/demo-comparison.json) 一致，`samples` 改为 `id`、`request_idx`、可选的 `reference` 和 `language`（`en` / `zh`）。两个运行必须使用同一请求清单和索引。生产记录建议以稳定哈希填写 `environment_id` 和 `request_settings_id`，分别覆盖设备/软件与请求/预处理/解码设置。使用 `target` 标识真实板端结果，精度字段录入已审计的有效值，缺证据时填 `unknown`。如需比对 token 或清理前文本，可在规范化对照中附上 `first_token_id`、`stop_token_ids`、`raw_text`。常规 Edge CLI 产物只使用实际存在的字段。

## 自定义校准与评测数据

用 `dataset-validate` 校验本地音频清单、检测校准与评测之间的相同音频文件，并生成 split 摘要。用 `prepare-eval` 生成 Edge-LLM 请求和独立评分清单。校准 split 可通过 `calibration_source` 传给 Edge-LLM Python API。完整示例见 [自定义数据集](docs/DATASETS.md)。

## 修复复验

把同一个 FP16 基线与修复前后两次候选放在一起，可以查看恢复文本、新增空转写、三组共同样本的 CER/WER 和可选性能代价。运行前检查数据、请求和精度变化是否符合原工单。见 [修复复验命令与演示](docs/REPLAY.md)。

有逐步解码记录时，工作台展示实际选择、Top-1 和可比的 EOS 分差。见 [解码证据导入合同](docs/DECODE_EVIDENCE.md)。

已有本地 HF 权重时，可在 Mac 上用 `inspect` 查看组件体积、存储 dtype 和 scale，用 `audit-checkpoints` 检查修复前后的实际张量变化。见 [模型产物检查](docs/CHECKPOINT.md)。

多个候选可在同一批样本上比较覆盖、CER/WER 和相对指定参照的配对区间。见 [多候选对照](docs/COMPARE.md)。

## 选择性 QAT

本机生成并校验配方。

```sh
python3 -m quantdoctor qat-template -o qat.json
python3 -m quantdoctor qat-validate qat.json
```

选择诊断实验锁定的 1–14 个精确 decoder 线性投影名。模板中的层名仅为语法示例。填入固定 HF revision 和两份相互独立的数据清单。清单路径相对于配方，音频路径相对于清单。每份清单是 JSON 数组，条目包含：

- `id` 和 `audio`，音频为本地 16 kHz 文件且长度最多 20 秒。
- `audio_sha256`，用于内容校验与划分检查。
- `prompt_ids`，来自固定模型 tokenizer 的已渲染提示，包含唯一的 audio placeholder。
- `target_ids`，与提示拼接兼容的参考转写及按实验约定添加的结束 token。生成这些 ID 时检查 tokenizer 边界。清单负责前缀、语言标记及终止符语义。

在准备好工具链的 Linux/CUDA Worker 执行。该适配器绑定 Edge-LLM `1231d6415ce38d0b2f20b3308ac24e58ec4100a8`、ModelOpt 0.45.0、PyTorch 2.13.0 和 Transformers 5.14.1，并需要 Edge-LLM 量化工具依赖。

```sh
python3 -m quantdoctor qat-run qat.json \
  --snapshot /path/to/pinned-float-hf-snapshot \
  --edge-repo /path/to/TensorRT-Edge-LLM \
  --output /path/to/new-qat-run
```

Worker 使用同一量化配方校准 ASR 联合模型，保留 fake quant，冻结所选投影之外的参数。以 teacher forcing 的转写 token CE 训练选定权重，FP32 master weights 用于 AdamW。首版训练策略是 batch 1、FP8 或 NVFP4 decoder、FP16 Audio/LM Head/KV。输出 `checkpoint/` 和 `qat-result.json`，后者记录配方、输入指纹、损失、已核对的 packed 权重与待验证状态。QAT 阶段复用 checkout 的 Python API，版本检查与 artifact 审计将未知行为拦在出包前。 [ModelOpt QAT 指南](https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html)

部署验收在对应板卡完成。保持同一数据、目标精度和运行设置，对比同配方 PTQ 与 QAT，执行 Edge-LLM 导出、构建和推理后导入工作台。首轮真实 CUDA smoke 仍待设备。QAD、浮点修复、逐层自动搜索和训练续跑属于后续范围。

## 离线设备工单

```sh
python3 -m quantdoctor job-validate examples/worker-job.json
python3 -m quantdoctor job-result-validate examples/worker-job.json examples/worker-result.json
```

示例是接口 fixture。校验只检查阶段字段、产物角色和 SHA-256 形状，以及回传是否匹配该 job。设备适配器和实际执行证据按 [产品架构](docs/ARCHITECTURE.md) 分阶段接入。

有真实输入或回传文件时，可以生成摘要并校验实际产物。见 [离线产物校验](docs/ARTIFACTS.md)。

选定候选后，可先用 [目标设备预检](docs/PREFLIGHT.md)检查已知精度限制，再生成并核对各阶段工单。
