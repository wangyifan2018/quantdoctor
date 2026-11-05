# 生成证据与 EOS 分差

打开 [修复复验演示](REPLAY.md)，选择 ASR-014。三列显示同一个生成步在浮点、NVFP4、恢复 Audio 后的选择 ID、Top-1 和 EOS 分差。示例使用合成数据。

分差定义为「FP16 实际选择的非停止 token 分数减最高有效停止 token 分数」。分差为负表示某个停止 token 在这两个选项中分数更高。界面单独显示实际选择，因为采样和分数处理可以影响结果。即使同一步出现分差反转和提前停止，也应结合受控修复复验来判断量化归因。

## 采集合同

可在对照文件的每个输出对象添加 `trace` 数组，也可为标准 Edge `output.json` 导入独立 sidecar：

```sh
python3 -m quantdoctor import-edge eval-bundle/manifest.json fp16.json nvfp4.json \
  --baseline-trace fp16-trace.json --candidate-trace nvfp4-trace.json -o comparison.json
```

sidecar 结构：

```json
{
  "schema_version": 1,
  "run_id": "candidate-nvfp4",
  "request_document_sha256": "<来自评分清单的 SHA-256>",
  "samples": {
    "ASR-014": [{
      "step": 0,
      "context_sha256": "<该步实际输入及生成前缀的 SHA-256>",
      "score_stage": "processed_logits",
      "selected_id": 23,
      "top1_id": 23,
      "stop_ids": [23],
      "scores": [{"id": 101, "logit": 3.2}, {"id": 23, "logit": 4.0}]
    }]
  }
}
```

采集方应统一对实际音频内容标识、渲染后的 prompt token IDs 与已生成的 token 前缀做规范化哈希。量化后的 embedding 数值不进入该摘要，以便在同一输入条件下比较不同精度。step 从第一个生成 token 起计数，Top-1 对应采集分数所在阶段，selected ID 是运行时实际选择。允许 `raw_logits` 和 `processed_logits`。候选分数应覆盖同前缀下 FP16 实际选中的参考 token 和所有有效停止 token。日志中的 top-k 如果未包含这些分数，保留缺失状态。

界面在请求摘要、运行条件、步号、上下文、分数阶段和有效停止集合均一致时才比较分差。不同的自然生成轨迹在分歧后的上下文通常不再可比。可以浏览后续的实际选择，但不会生成误导性的分差。缺失步骤保留在时间线中。

设备侧抓取器仍待板端验证。当前实现覆盖本地合同校验、sidecar 导入、对齐、计算、报告与 Web 展示。
