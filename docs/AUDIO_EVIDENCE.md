# 音频边界证据

QuantDoctor 可以在 Mac 上比较已采集的音频输入、Mel 特征、Encoder 输出与解码选择。采集器和实际板端数值仍需在设备上验证。仓库示例使用合成数据。

```sh
python3 -m quantdoctor serve examples/demo-comparison.json \
  --retest examples/demo-retest.json --sample-id ASR-014
```

在样本 ASR-014 中，输入和 Mel 摘要一致，原候选最早在 Encoder 输出观察到差异。下方仍可看到 [EOS 分差反转和选中的停止符](DECODE_EVIDENCE.md)。修复候选的合成 Encoder 记录与基线一致。这个例子演示定位顺序。真实运行的浮点输出可能有正常偏差，摘要变化无法单独证明故障原因。

## 导入合同

与解码 sidecar 类似，可在标准 Edge 结果导入时附加音频证据：

```sh
python3 -m quantdoctor import-edge eval-bundle/manifest.json fp16.json nvfp4.json \
  --baseline-audio fp16-audio.json --candidate-audio nvfp4-audio.json -o comparison.json
```

每个 sidecar 包含 `schema_version: 1`、`run_id`、`request_document_sha256` 和按样本 ID 索引的 `samples`。样本必须能对应到运行结果。也可以将同一个样本对象直接写到比较文件的 `baseline.audio` 或 `candidate.audio`。允许只采集部分边界。

```json
{
  "input_sha256": "<实际读入的编码音频字节 SHA-256>",
  "mel": {
    "shape": [128, 100],
    "representation": "mel-valid-frames-f32-le-c-v1",
    "sha256": "<规范张量字节 SHA-256>"
  },
  "encoder": {
    "shape": [13, 1024],
    "representation": "proj2-valid-rows-f32-le-c-v1",
    "sha256": "<规范张量字节 SHA-256>",
    "sketch": [[0, 0.25], [1024, -0.5]]
  }
}
```

同一 `representation` 必须约定同一个语义边界、有效行或帧裁切、轴顺序、dtype、端序和连续布局。例中以 Mel 有效帧及 `proj2` 有效输出行为边界。SHA-256 对约定后的实际张量字节计算，shape 单独核对。选填的 `sketch` 是按同一顺序展平后的索引与有限数值，最多 256 个，索引递增。只有两边索引完全一致时才报抽样 RMSE 和最大绝对差。抽样指标描述这些位置，无法代表完整张量分布。

`prepare-eval` 把预期音频摘要写入评分清单。运行侧报告的实际摘要与清单或另一组运行冲突时，该样本会退出配对 CER/WER、EOS 分差和修复收益统计，且保留冲突状态。多候选性能表保留测量值，同时暂停前沿判断。历史结果缺少摘要仍可查看，界面标记证据缺失。前序缺采或表示不兼容时，报告只说明最早观察到的差异，明确显示前序缺口。
