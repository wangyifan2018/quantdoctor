# 离线工单产物校验

QuantDoctor 可以在 Mac 上核对工单输入和 Worker 回传文件的内容。每个角色对应根目录中的同名文件或目录，例如 `inputs/engine_bundle/`、`inputs/requests` 和 `outputs/responses`。应用格式由对应阶段决定。

```sh
python3 -m quantdoctor artifact-digest inputs/engine_bundle
python3 -m quantdoctor job-input-verify job.json inputs
python3 -m quantdoctor job-result-validate job.json result.json --artifact-root outputs
```

用 `artifact-digest` 的值填写工单或结果的对应角色。文件摘要为 SHA-256，输入为 UTF-8 前缀 `quantdoctor:file:v1`、一个 NUL 字节和原始文件 SHA-256 的 32 字节值。目录摘要使用 `quantdoctor:directory:v1`、一个 NUL 字节及规范 JSON 文件清单。清单包含 `schema_version: 1` 和按路径排序的 `files`，每条为相对 POSIX 路径与原始文件的十六进制 SHA-256。JSON 使用 ASCII 转义、排序键、无分隔空格。复制文件树到新目录时摘要相同，改名或改内容时摘要改变。检查要求角色内有普通文件，拒绝符号链接及空目录工件。

工单 ID、角色和内容一致意味着回传文件匹配声明。验证输出保留来源，且 `execution_verified` 为 false。设备能力仍以实际成功执行及其独立验收证据为准。HF checkpoint 用于修复复验的张量内容摘要有独立用途，见 [Checkpoint 审计](CHECKPOINT.md)。
