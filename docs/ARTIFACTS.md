# 离线工单产物校验

QuantDoctor 可以在 Mac 上核对工单输入和 Worker 回传文件的内容。每个角色对应根目录中的同名文件或目录，例如 `inputs/engine_bundle/`、`inputs/requests` 和 `outputs/responses`。应用格式由对应阶段决定。

```sh
python3 -m quantdoctor artifact-digest inputs/engine_bundle
python3 -m quantdoctor job-input-verify job.json inputs
python3 -m quantdoctor job-result-validate job.json result.json --artifact-root outputs
```

用 `artifact-digest` 的值填写工单或结果的对应角色。文件摘要为 SHA-256，输入为 UTF-8 前缀 `quantdoctor:file:v1`、一个 NUL 字节和原始文件 SHA-256 的 32 字节值。目录摘要使用 `quantdoctor:directory:v1`、一个 NUL 字节及规范 JSON 文件清单。清单包含 `schema_version: 1` 和按路径排序的 `files`，每条为相对 POSIX 路径与原始文件的十六进制 SHA-256。JSON 使用 ASCII 转义、排序键、无分隔空格。复制文件树到新目录时摘要相同，改名或改内容时摘要改变。检查要求角色内有普通文件，拒绝符号链接及空目录工件。

工单 ID、角色和内容一致意味着回传文件匹配声明。验证输出保留来源，且 `execution_verified` 为 false。设备能力仍以实际成功执行及其独立验收证据为准。HF checkpoint 用于修复复验的张量内容摘要有独立用途，见 [Checkpoint 审计](CHECKPOINT.md)。

## 生成并串联工单

在阶段输入根目录放好同名角色，再生成工单：

```sh
python3 -m quantdoctor job-create export edge-onnx-cli-v1 linux-export-v1 export-inputs -o export.json
```

Worker 回传结果后，把下一阶段需要的产物复制到下游输入目录的同名角色，并放入新增配置。例如将 `export-outputs/onnx_bundle` 复制为 `build-inputs/onnx_bundle`，并准备 `build-inputs/build_config`：

```sh
python3 -m quantdoctor job-next export.json export-result.json export-outputs \
  build edge-onnx-cli-v1 thor-sm110-sdk-v1 build-inputs -o build.json
```

支持 PTQ/QAT → Export → Build → Infer 或 Profile。命令校验上游工单 ID、成功状态、所有输出摘要、下游输入角色与复制过来的产物。Engine 从 Build 转交 Infer 或 Profile 时要求使用相同 profile ID。要试跑合成文件链，可显式传 `--allow-fixture`。实验 provenance 以原始结果文件为准，保留每个上游工单和结果供复核。下游工单本身描述输入内容和执行配置。

## 从 Infer 工单导入评分

为基线和每个候选分别准备一个成功返回的 Infer 工单目录。两个工单使用同一评测请求，并通过不同工单 ID 区分。

```text
candidate/
  job.json
  result.json
  inputs/
    engine_bundle/
    requests
  outputs/
    responses
```

`inputs/requests` 保留实际交给 Infer Worker 的 JSON 文件，`outputs/responses` 使用 `llm_inference` 原始输出。评测清单和原始请求可用 `prepare-eval` 生成。清单的运行 `environment_id` 与工单 `profile_id` 一致，`source` 使用 `target` 对应 `worker` 回传、`fixture` 对应模拟回传。

```sh
python3 -m quantdoctor import-worker-edge eval/manifest.json fp16/ nvfp4/ -o nvfp4-comparison.json
```

导入时复核 engine、请求和响应的文件摘要、规范化请求摘要及完整的请求索引。规范化结果的每个 run 持有 `worker_evidence`，其中 `scoring_sha256` 覆盖样本 ID、参考、语言、预期音频摘要、说话人及该 run 的最终文本、停止原因、生成数。对同一模板的原始音频请求，额外保留 `request_protocol_sha256` 以支持 [跨数据集设置核对](DATASETS.md#跨数据集核对请求设置)。手工改变这些评分字段后需重新从原始工件导入。也可使用 `import-edge` 的音频和解码 sidecar 参数，附加采集独立关联。多候选页面的「运行产物」可展开查看摘要。

`edge-asr-infer-v1` 的成功回传还包含 `outputs/invocation`，用于核对固定 CLI 参数、二进制摘要和运行读取的音频摘要。旧工单仍可按原格式导入。见 [ASR Infer 适配器](INFER.md)。

此记录用于本地复核，工单及摘要没有签名。评测清单及附加 sidecar 仍由实验提供方维护。板端实际加载的 engine、动态库、内核和有效精度需另用目标日志和测量记录核实。
