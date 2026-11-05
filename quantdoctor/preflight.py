"""Static deployment preflight against explicitly versioned device rules."""

from dataclasses import asdict, dataclass
import re

from .diagnostics import ASR_MODEL, work_order
from .model import nonempty, require
from .qat import EDGE_COMMIT


PLATFORMS = ("macos", "jetson_orin", "jetson_thor", "drive_thor", "other")
SOURCE = f"https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/{EDGE_COMMIT}/docs/source/user_guide"


@dataclass(frozen=True)
class TargetProfile:
    id: str
    platform: str
    edge_commit: str

    @classmethod
    def parse(cls, data):
        require(isinstance(data, dict) and type(data.get("schema_version")) is int
                and data["schema_version"] == 1, "expected target profile schema_version 1")
        require(nonempty(data.get("id")) and data.get("platform") in PLATFORMS,
                "target profile requires an ID and a recognized platform")
        commit = data.get("edge_commit")
        require(isinstance(commit, str) and re.fullmatch(r"[a-f0-9]{40}", commit),
                "target profile requires a 40-digit lowercase Edge-LLM commit")
        return cls(data["id"], data["platform"], commit)


def _check(model, precision, profile):
    issues = []
    if profile.platform == "macos":
        issues.append({"code": "macos_cuda_runtime", "severity": "block",
                       "detail": "本机 profile 用于离线分析。Edge-LLM 的 CUDA/TRT 构建和推理需要 NVIDIA 环境。"})
    if profile.edge_commit != EDGE_COMMIT:
        issues.append({"code": "edge_rules_unavailable", "severity": "review",
                       "detail": "此 Edge-LLM commit 没有内置静态规则。"})
    else:
        unsupported = [key for key, value in precision.items() if value in {"fp8", "nvfp4"}]
        if profile.platform == "jetson_orin" and unsupported:
            issues.append({"code": "orin_precision", "severity": "block", "components": unsupported,
                           "detail": "此 Edge-LLM 版本的 Jetson Orin 运行路径排除 FP8/NVFP4。",
                           "source": SOURCE + "/getting_started/supported-models.md#L18"})
        if model == ASR_MODEL and precision["backbone"] == "nvfp4" and precision["audio"] == "fp8":
            issues.append({"code": "asr_empty_output_risk", "severity": "review",
                           "detail": "官方示例在 0.6B 的这组精度观察到空转写，需用同样本复验。",
                           "source": SOURCE + "/examples/asr.md#L115"})
    unknown = [key for key, value in precision.items() if value == "unknown"]
    if unknown:
        issues.append({"code": "unknown_precision", "severity": "review", "components": unknown,
                       "detail": "这些组件的实际精度待审计。"})
    return {"precision": precision, "state": "blocked" if any(row["severity"] == "block" for row in issues)
            else "unverified", "issues": issues}


def preflight(comparison, profile, sample_id=None):
    original = dict(comparison.candidate.precision)
    checks = {"candidate": _check(comparison.model, original, profile)}
    order = None
    if sample_id is not None:
        order = work_order(comparison, sample_id)
        change = order["finding"]["action"]["change"]
        require(len(change) == 1 and all(key.startswith("precision.") for key in change),
                "deployment preflight requires a single precision intervention")
        proposed = {**original, **{key.split(".", 1)[1]: value for key, value in change.items()}}
        checks["proposed"] = _check(comparison.model, proposed, profile)
    return {"schema_version": 1, "kind": "target_preflight", "model": comparison.model,
            "source": comparison.candidate.source, "candidate_id": comparison.candidate.id,
            "profile": asdict(profile), "checks": checks, "work_order": order, "deployment_verified": False}
