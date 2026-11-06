# 从 ASR Engine 到可复核的推理结果

`edge-asr-infer-v1` 对应固定 Edge-LLM checkout 的单设备 Qwen3-ASR 推理 CLI。Mac 可以校验工单、目录结构和请求，并审阅命令。在 Linux 目标设备上，`infer-run` 用明确的参数启动已构建的二进制，记录调用、二进制摘要、音频摘要、退出码和原始响应。适配器已用模拟进程完成本地往返验证，实际 CUDA 和 TensorRT 运行待板端测试。

## 准备工单

用 [设备预检](PREFLIGHT.md) 确认当前候选与目标 profile 的已知限制。将构建输出和 `prepare-eval` 的 `input.json` 按下面的角色放入新的工单目录。`requests` 是文件名，内容仍为 JSON。`llm/` 和 `audio/` 保留构建产物，示意只列出静态检查涉及的核心文件。

```text
infer-fp16/
  inputs/
    engine_bundle/
      llm/
        config.json
        llm.engine
        embedding.safetensors
        tokenizer.json
        chat_template.model
      audio/
        config.json
        audio_encoder.engine
    requests
```

`llm/` 可用 `chat_template.jinja`。只保留一种聊天模板入口。请求支持统一模板的单个原始音频，评测音频路径可指向目标板目录。配置和引擎必须能独立运行，当前适配器范围为单设备、默认非 speculative 解码、无 LoRA。为实际 SDK 与板卡使用稳定且独立的 profile ID。示例 `jetson-thor-example` 需要和工单及评分清单的 `environment_id` 一致。

```sh
python3 -m quantdoctor job-create infer edge-asr-infer-v1 jetson-thor-example infer-fp16/inputs -o infer-fp16/job.json
python3 -m quantdoctor infer-plan infer-fp16 examples/target-thor.json --binary /opt/edge/bin/llm_inference
```

已有 Build 工单时也可用 `job-next` 绑定同一 profile 下的 engine 输出。`infer-plan` 校验工件字节、配置和请求，并输出 `argv`，在 Mac 上可以使用目标设备的二进制路径。该路径与音频文件到执行时核对。

固定 checkout 的 `LLMRankRuntime` 在 `multimodalEngineDir` 下查找 `audio/audio_encoder.engine`。适配器将 `--multimodalEngineDir` 设为 `inputs/engine_bundle`，将 `--engineDir` 设为 `inputs/engine_bundle/llm`。同一 checkout 的 ASR 示例第 99 行将 multimodal 参数指向 `engines/audio`，这个路径会被 runtime 再追加一次 `/audio`。参数以 runtime 源码为依据。 [runtime 目录查找](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/1231d6415ce38d0b2f20b3308ac24e58ec4100a8/cpp/runtime/llmRankRuntime.cpp#L588-L618) · [ASR 示例命令](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/1231d6415ce38d0b2f20b3308ac24e58ec4100a8/docs/source/user_guide/examples/asr.md#L92-L104) · [LLM 文件命名](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/1231d6415ce38d0b2f20b3308ac24e58ec4100a8/cpp/common/parallelArtifactNames.cpp#L68-L76)

## 执行与导入

在目标设备准备好 TensorRT、CUDA 和相应的插件加载环境。复制工单和音频，保留 `job.json` 中的输入内容。使用已构建的 CLI 路径执行：

```sh
python3 -m quantdoctor infer-run infer-fp16 examples/target-thor.json --binary /opt/edge/bin/llm_inference
```

`infer-run` 固定参数列表。采样、batch、生成长度和 logprobs 从请求文件读取。工单目录须是新的执行目录。启动后写 `outputs/stdout.log` 和 `outputs/stderr.log`。调用结束后重核工单输入、可执行文件与音频摘要，并发布 `result.json`。失败可使用新目录重试，原目录保留诊断日志。`--fixture` 可在 Mac 用合成可执行文件测试流程，结果标记为 `fixture`。

可导入的 `succeeded` 表示响应收集完整。Edge-LLM 可在单条请求失败后写出完整 JSON，并返回退出码 1。这种情况保留每条失败响应，CER/WER 不将它视为空转写。缺失、错误索引或不可复核的整体输出会记为工单 `failed`。 [逐请求结果](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/1231d6415ce38d0b2f20b3308ac24e58ec4100a8/examples/llm/llm_inference.cpp#L2044-L2066) · [写出与退出](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/1231d6415ce38d0b2f20b3308ac24e58ec4100a8/examples/llm/llm_inference.cpp#L2286-L2321)

```sh
python3 -m quantdoctor import-worker-edge eval/manifest.json infer-fp16 infer-nvfp4 -o comparison.json
python3 -m quantdoctor serve comparison.json
```

导入时校验 `outputs/invocation` 摘要、固定 CLI 参数、请求索引以及实际音频摘要与评测清单的一致性。两侧都有调用记录时，音频摘要必须匹配；二进制摘要不同的运行会被判为不可比。Web 的「运行产物」展示退出码、二进制摘要及音频核对覆盖。二进制摘要用于辨别文件身份，运行环境、板端有效精度和内核仍需独立测量。
