# 多候选对照与配对区间

在 Mac 上可以把多份已经导入的 ASR 结果并排比较。默认相对 FP16 基线计算差值，也可以选原始 NVFP4 作为参照，直接观察修复或换配方的变化。

```sh
python3 -m examples.make_quality_demo quality-demo
python3 -m quantdoctor compare quality-demo/original.json quality-demo/repaired.json quality-demo/fp8.json \
  --reference-run nvfp4-demo --costs quality-demo/costs.json --port 8765
```

省略 `--port` 输出 JSON，可用 `-o comparison-report.json` 保存。示例中的转写与区间均来自合成数据。

## 可比条件与评分

- 每次可比较 2–12 个候选。候选须使用同一个模型和浮点基线，评测集与请求摘要一致。候选的运行来源、环境、模型 revision 与请求配置须和基线匹配。样本、语言、参考文本和浮点基线输出相同，结果文件的样本顺序可以不同。
- 每种语言独立评分。报告有标注样本数、共同可评分样本数和参考单位。空转写按删除计入 CER/WER。运行失败或缺失会把该样本移出所有候选的共同评分集合，并在每个候选的覆盖状态保留。不要只看排名而忽略覆盖差异。
- 区间使用固定 seed 的 1,000 次配对重采样。说话人标记完整时，抽样单位为说话人，每次取被抽中说话人的所有共同样本，再重新聚合编辑数差及参考单位数。取 2.5% 与 97.5% 分位数，单位为误差率百分点。至少需要 10 个重采样单位。部分样本缺标记时省略区间。旧数据完全没有说话人 ID 时保留按录音重采样，并注明相关性未知。页面还可查看单个说话人的共同样本及候选误差。 [ASR 分块重采样研究](https://www.isca-archive.org/interspeech_2020/liu20c_interspeech.pdf)
- 页面按 CER、WER 分开排序，并支持逐样本查看共同评分资格和实际转写。默认参照 FP16。若要判断恢复 Audio 相比原 NVFP4 的变化，应直接选原 NVFP4 为参照，而不是比较两个各自相对 FP16 的区间是否重叠。
- 候选选定后在独立留出集复测。当前区间和前沿均不构成部署验收。

## 设备代价与候选前沿

可用 `--costs FILE` 导入同一测量协议下的候选成本。生成的 `quality-demo/costs.json` 是完整的合成示例。真实记录需要：

- `schema_version: 1`、`protocol_id`、`source`、`environment_id`、`evaluation_dataset_sha256` 与 `request_document_sha256`。协议应固定预热、重复、功率模式与计时范围。环境、来源和数据摘要须与候选对照一致。
- `runs` 按候选 run ID 存放 `rtf_p95`、`peak_memory_bytes`、可选的 `artifact_bytes`。缺测项可省略，不能拿别的 run 填补。

报告保留精度、空转写、失败与缺失、设备成本。前沿比较使用共同样本的 CER/WER 点估计、失败与缺失总数、空转写数、RTF P95 和峰值内存，均为越低越好。若另一个方案各目标均不差且至少一项更好，则标出支配它的方案。产物大小作为补充信息单列。至少两份候选有完整成本且所有出现的语言都有共同评分样本时才判断前沿。缺测方案仍在表里，前沿只描述记录完整的方案。

用 [修复复验](REPLAY.md)查看单个干预的实际权重变化和逐样本恢复。设备测量与部署资格仍需独立验收。
