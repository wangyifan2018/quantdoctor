# 检查与审计 HF 模型产物

此功能在 Mac 上读取已有本地目录，无需 Torch、模型初始化或网络。支持单个 `model.safetensors` 和 `model.safetensors.index.json` 分片。 [Safetensors 格式](https://github.com/safetensors/safetensors#format)

```sh
python3 -m quantdoctor inspect /path/to/snapshot -o inventory.json
python3 -m quantdoctor inspect /path/to/snapshot --port 8765
```

工作台按 Qwen3-ASR checkpoint 命名划分 Audio、Backbone、LM Head 和其他张量，显示实际存储 dtype、字节数、shape 和 scale 张量。声明量化方案与实际存储分开显示。像 NVFP4 这样的 packed 载体可能存为整数 bytes，存储 dtype 不能独立证明计算精度。绑定 Head 可能没有独立权重，界面保留配置声明。没有可识别的命名时归到其他组件。

做一次「只恢复 Audio」实验后，可以对照原始 checkpoint 和修复后的 checkpoint：

```sh
python3 -m quantdoctor audit-checkpoints /path/to/original /path/to/repaired \
  --expect-component audio --port 8765
```

审计以固定内存按块读取两个 checkpoint 的张量内容，比较名称、dtype、shape 和内容 SHA-256。分片文件重新布局不影响张量身份。报告列出新增、删除和内容变化，检查实际变更张量是否局限于所选组件，并单独显示 tied-weights 声明变化。JSON 输出包含全部张量记录，Web 支持按组件浏览。

这条证据和 [修复复验](REPLAY.md)互补。前者检查 checkpoint 权重和 scales 发生了什么变化，后者检查样本与指标。运行时精度、量化配置的其他字段、ONNX、engine 和 CUDA kernel 仍需独立证据。默认 `inspect` 只读头部，`--hash-weights` 可额外读取并计算张量内容摘要。
