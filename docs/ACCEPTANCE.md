# 用显式规则评估部署门槛

先固定留出评测集、目标环境、设备测量协议和验收阈值，再比较候选。QuantDoctor 可在 Mac 复算这些规则。下面的示例使用合成数据。

```sh
python3 -m examples.make_quality_demo quality-demo
python3 -m quantdoctor compare quality-demo/original.json quality-demo/repaired.json quality-demo/fp8.json \
  --costs quality-demo/costs.json --policy quality-demo/policy.json --port 8765
```

省略 `--port` 得到 JSON。验收相对同环境 FP16 基线，保持默认参照。若用原 NVFP4 作参照分析某次修复，应单独运行 [候选对照](COMPARE.md)。

## 冻结规则与证据

用示例生成的 `policy.json` 作模板。替换 `environment_id`、`evaluation_dataset_sha256`、`request_document_sha256` 和 `protocol_id`，绑定同一评测和测量协议。报告保留规范 JSON 的规则摘要。成本记录按 run ID 提供，验收时也为 FP16 基线记录成本。

| 字段 | 判定 |
|---|---|
| `quality.cer` / `quality.wer` | 绝对误差率增量上限，单位为百分点。出现于评测的语言必须有相应规则。95% 配对说话人区间完全在上限内才算满足，完全超出才算未达。跨过门槛为证据不足。 |
| `min_common_samples` / `min_coverage` | 各指标的最少共同样本和最低覆盖比例。少于要求或没有说话人区间时待补证。 |
| `max_empty_increase_pp` | 空转写率相对基线的增量上限。使用两次运行都有结果且输入未冲突的请求，需满足最低覆盖率。该项使用观测点值。 |
| `max_unavailable_rate` | 在全部计划样本中，候选运行失败或缺失的比例上限，取 0 到 1。 |
| `max_rtf_increase_pct` | RTF P95 相对于同协议 FP16 基线的增量上限，单位为百分比。 |
| `min_resource_savings_pct` | 峰值内存或产物大小至少有一项达到的降幅，单位为百分比。若一项缺失、另一项未达，保留待补证。 |

候选精度未知或存在已知音频摘要冲突时，规则报告会指出证据缺口。`target` 来源还要求两侧的 Infer 产物经 [本地导入核对](ARTIFACTS.md)。任何一项明确未达时，总体显示未达到规则。其余情况下，全部满足才显示满足规则。门槛数字应在真实评测之前选定。示例中的 1 个百分点 CER/WER、0.5 个百分点空转写、5% RTF 和 10% 资源节省用于演示，正式项目应使用各自冻结的要求。

页面同时保留输入来源和逐项证据。即使合成数据或导入的板端记录满足规则，`deployment_verified` 仍为 false。真实部署验收还需要核对运行与目标 engine、实际硬件、日志和产物的关联，并确认留出集独立。
