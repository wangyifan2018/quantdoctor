# 单变量修复复验

修复工单提出一个精度改变。复验将原始候选、修复候选和同一 FP16 参考放回同一批样本，检查工单是否按声明执行。示例展示了恢复原失败样本、出现新空转写和性能代价同时发生时，如何作出判断。示例的转写、摘要和性能数字均为合成数据。

```sh
python3 -m quantdoctor replay examples/demo-comparison.json examples/demo-retest.json \
  ASR-014 --costs examples/demo-costs.json -o replay.json

python3 -m quantdoctor serve examples/demo-comparison.json \
  --retest examples/demo-retest.json --sample-id ASR-014 --costs examples/demo-costs.json
```

## 复验条件

1. 用 [评测集工具](DATASETS.md) 生成同一份推理请求和评分清单。获取原候选与修复后候选的两份运行结果。两个 comparison 都复用同一份 FP16 运行结果、相同模型 revision、设备环境、解码设置及请求和数据集摘要。修复后的候选应有新 run ID。
2. 按 [结果导入](../README.md) 生成原始 comparison。复制评分清单并更新 candidate 的 ID 和实际精度，生成修复 comparison。保持基线元数据与输出一致，包括缺结果样本。用原始工单的样本 ID 运行 `replay`。目前支持修改一个精度字段的工单。
3. QuantDoctor 检查声明的差异是否恰好是工单中的精度变化。需要复验的精度字段应填实际值。额外精度变化、不同摘要、不同基线输出或丢失样本记录会阻止复验。运行失败和没有返回的结果可以保留为失败状态和空记录。

## 如何读结果

- **转写恢复与副作用**按样本分别列出恢复文本、新增空转写、仍为空和缺失/失败。编辑数分布单独列出改善、退化、相同和无法评分的录音。恢复正文后仍可能增加编辑数。点击两组计数可筛选录音，查看修复前后实际文本和参考。
- **共同样本的质量**用基线、原始候选与修复候选三组均有有效输出记录且带参考文本的交集计算 CER/WER。成功运行产生的空文本按删除计分，缺记录和运行错误另计。不同语言的评分分母分别显示。
- **成本**可选导入 [成本文件](../examples/demo-costs.json)。用相同测量协议和两次候选 run ID 配对，协议应覆盖测试输入、预热、重复次数、功率模式和采样方式。可提供 RTF P95、峰值内存字节数、产物字节数。仅对两侧都有值的指标计算差异。
- **部署结论**保持待验收。可选的 [checkpoint 绑定审计](CHECKPOINT.md#将审计加入修复复验) 会核对摘要与实际张量范围。engine、日志和运行环境仍需板端证据。对于原始问题，先看失败样本能否恢复，再用独立留出集看恢复是否稳定，以及是否产生新的失败。

用 [候选对照的 `--replay`](COMPARE.md#在同一页面查看修复复验) 可将一组复验与其他量化候选、性能前沿和部署门槛放在同一页。单候选 `serve --retest` 也支持相同的状态和编辑数筛选。

## 独立留出集复验

冻结独立评测清单与验收门槛，先用诊断集选出干预，再用固定的基线、原方案和修复方案运行留出请求。`--diagnosis` 指定产生工单的原始对照，位置参数仍指向留出集的结果。下面的演示使用合成记录和模拟产物标识。

```sh
python3 -m examples.make_quality_demo holdout-demo --holdout
python3 -m quantdoctor compare holdout-demo/original.json holdout-demo/repaired.json holdout-demo/fp8.json \
  --diagnosis holdout-demo/diagnosis.json \
  --replay holdout-nvfp4-demo holdout-nvfp4-fp16-audio-demo ASR-014 \
  --costs holdout-demo/costs.json --policy holdout-demo/policy.json --port 8765
```

只输出单次复验的 JSON 时，可用 `quantdoctor replay holdout-demo/original.json holdout-demo/repaired.json ASR-014 --diagnosis holdout-demo/diagnosis.json`。`serve --retest` 同样接受 `--diagnosis`。单次复验若附带成本文件，其 `evaluation_dataset_sha256` 和 `request_document_sha256` 必须与留出集对照匹配。多候选成本文件已校验这两个字段，可以直接使用。

- **检查数据隔离。** 两批请求和数据集的摘要须不同。两批清单中的样本 ID、原始音频 SHA-256 和说话人 ID 都须完整且互不重叠。同一说话人的 ID 应在两份清单里使用同一个命名空间。已有运行采集时，实际音频摘要须与清单一致。用 `prepare-eval` 生成清单可从本地音频文件计算摘要。
- **检查运行关联。** 留出集基线和原方案与诊断集的对应运行固定模型 revision、环境、请求设置、运行来源和精度。运行 ID 可不同。两边都有 engine 或 checkpoint 摘要时要求对应摘要一致，缺少对等摘要时显示仅有运行声明。修复前后继续检查只改工单指定的一个精度字段。
- **查看泛化证据。** 工单样本只在诊断集。留出集的 CER/WER 和配对差值使用留出集三次运行的共同可评分记录。页面显示修复后相对原方案的百分点变化。每个语言至少有 10 位可配对说话人时提供 95% bootstrap 区间，同一说话人的样本一起重采样。

隔离核对以提交的清单、音频字节摘要和说话人 ID 为依据。改编码的重复录音、同一说话人的不同 ID、选择策略时是否看过留出结果，都需要实验方另外控制。若使用 QAT，另用 [QAT 数据预检](QAT_PREFLIGHT.md) 检查训练、校准与留出数据的隔离。报告保留设备验收待评估状态。

## 下一批功能

| 顺序 | 功能 | 形成的产品价值 |
|---|---|---|
| 1 | 接入板端解码证据采集。 [本地 EOS 分差视图与导入合同](DECODE_EVIDENCE.md)已实现。 | 在同一前缀下对齐分数和实际选择，定位量化前后观察到的分歧。 |
| 2 | 已支持 [checkpoint 张量审计与复验绑定](CHECKPOINT.md)。下一步扩展 ONNX、engine 精度和配置证据。 | 将复验从声明的精度差异推进到实际产物证据，减少误归因。 |
| 3 | 目标设备验收与敏感度排名。在同一套评测与性能协议下比较分组恢复、不同校准集和选择性 QAT。 | 依据任务恢复、延迟和内存代价筛选可部署配方，区分 Torch 与设备结果。 |
