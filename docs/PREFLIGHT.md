# 目标设备静态预检

在 Mac 上可对照原候选和一个精度恢复工单，查看已知硬件限制与精度风险：

```sh
python3 -m quantdoctor preflight examples/demo-comparison.json examples/target-orin.json \
  --sample-id ASR-014 --port 8765
```

省略 `--port` 得到 JSON。省略 `--sample-id` 时只检查原候选。

Profile 需要 `schema_version: 1`、设备配置 ID、`platform` 和完整 `edge_commit`。平台可为 `macos`、`jetson_orin`、`jetson_thor`、`drive_thor` 或 `other`。为不同设备、SDK、TensorRT 或插件版本使用不同配置 ID，并在真实运行记录中绑定设备信息。静态规则仅对内置的 Edge-LLM commit 生效。其他版本的 profile 会提示缺少对应规则。Mac 的设备预检会明确标记 CUDA/TRT 路径不可用。

以合成案例为例，恢复 Audio 到 FP16 可以移除 Qwen3-ASR-0.6B 的 NVFP4 LLM + FP8 Audio 组合风险提示。Jetson Orin 仍受 NVFP4 LLM 限制。换 Thor profile 后，恢复方案显示待验证。工作台在有文档规则时附上具体依据。预检输出仅有 `blocked` 或 `unverified`，设备实测另行记录。

接入板端时，将选定方案的实际配方作为离线工单输入。预检 JSON 用于人工复核，生成工单的命令不会自行判断配方、engine 格式或可部署性。 [工单串联](ARTIFACTS.md#生成并串联工单)
