"""Compare bounded observations at ASR frontend and encoder boundaries."""

from dataclasses import dataclass
import math

from .model import SHA256, nonempty, require
from .trace import compare_steps


def _sha256(value):
    return isinstance(value, str) and bool(SHA256.fullmatch(value))


@dataclass(frozen=True)
class TensorEvidence:
    shape: tuple[int, ...]
    representation: str
    sha256: str
    sketch: tuple[tuple[int, float], ...]

    @classmethod
    def parse(cls, data):
        require(isinstance(data, dict), "audio tensor evidence must be an object")
        shape, representation, digest = (data.get(key) for key in ("shape", "representation", "sha256"))
        require(isinstance(shape, list) and 1 <= len(shape) <= 8 and all(
            type(size) is int and 0 <= size <= 2**31 - 1 for size in shape
        ), "invalid audio tensor shape")
        require(nonempty(representation) and _sha256(digest), "audio tensor needs representation and SHA-256")
        sketch = data.get("sketch", [])
        require(isinstance(sketch, list) and len(sketch) <= 256, "audio tensor sketch may have at most 256 entries")
        previous = -1
        parsed = []
        size = math.prod(shape)
        for row in sketch:
            require(isinstance(row, list) and len(row) == 2, "expected audio sketch [index, value]")
            index, value = row
            require(type(index) is int and previous < index < size, "audio sketch indices must be sorted and in range")
            require(type(value) in (int, float) and -1e12 <= value <= 1e12 and math.isfinite(value),
                    "audio sketch values must be finite and in range")
            parsed.append((index, float(value)))
            previous = index
        return cls(tuple(shape), representation, digest, tuple(parsed))


@dataclass(frozen=True)
class AudioEvidence:
    input_sha256: str | None
    mel: TensorEvidence | None
    encoder: TensorEvidence | None

    @classmethod
    def parse(cls, data):
        require(isinstance(data, dict) and data.keys() <= {"input_sha256", "mel", "encoder"},
                "invalid audio evidence fields")
        digest = data.get("input_sha256")
        require(digest is None or _sha256(digest), "invalid audio input SHA-256")
        mel = TensorEvidence.parse(data["mel"]) if data.get("mel") is not None else None
        encoder = TensorEvidence.parse(data["encoder"]) if data.get("encoder") is not None else None
        require(digest is not None or mel is not None or encoder is not None, "empty audio evidence")
        return cls(digest, mel, encoder)


def _tensor_row(stage, baseline, candidate, available):
    row = {"stage": stage, "status": available or "missing",
           "baseline_shape": baseline.shape if baseline else None,
           "candidate_shape": candidate.shape if candidate else None}
    if available or baseline is None or candidate is None:
        return row
    if baseline.representation != candidate.representation:
        return {**row, "status": "representation_mismatch"}
    if baseline.shape != candidate.shape:
        return {**row, "status": "different", "reason": "shape"}
    if baseline.sha256 == candidate.sha256:
        first, second = dict(baseline.sketch), dict(candidate.sketch)
        if any(first[index] != second[index] for index in first.keys() & second.keys()):
            return {**row, "status": "inconsistent_capture"}
        return {**row, "status": "same"}
    row.update(status="different", reason="payload")
    if baseline.sketch and [index for index, _ in baseline.sketch] == [index for index, _ in candidate.sketch]:
        errors = [a[1] - b[1] for a, b in zip(baseline.sketch, candidate.sketch, strict=True)]
        row["sketch"] = {"count": len(errors), "rmse": math.sqrt(sum(value * value for value in errors) / len(errors)),
                         "max_abs_error": max(map(abs, errors))}
    return row


def compare_audio(baseline, candidate, comparable, expected):
    if baseline is None and candidate is None:
        return []
    empty = AudioEvidence(None, None, None)
    baseline, candidate = baseline or empty, candidate or empty
    if not comparable:
        status = "run_mismatch"
    elif len({digest for digest in (expected, baseline.input_sha256, candidate.input_sha256) if digest}) > 1:
        status = "different"
    elif not baseline.input_sha256 or not candidate.input_sha256:
        status = "missing"
    else:
        status = "same"
    unavailable = {"different": "input_mismatch", "run_mismatch": "run_mismatch"}.get(status)
    return [
        {"stage": "audio_input", "status": status},
        _tensor_row("mel", baseline.mel, candidate.mel, unavailable),
        _tensor_row("encoder", baseline.encoder, candidate.encoder, unavailable),
    ]


def boundary_summary(audio, decode):
    first = next((row for row in audio if row["status"] == "different"), None)
    if first:
        earlier = audio[:audio.index(first)]
        return {"stage": first["stage"], "earlier_evidence_complete": all(row["status"] == "same" for row in earlier)}
    difference = next((row for row in decode if row.get("selected_changed")), None)
    if difference:
        return {"stage": "decode", "step": difference["step"],
                "earlier_evidence_complete": bool(audio) and all(row["status"] == "same" for row in audio)}
    return None


def sample_evidence(baseline, candidate, comparable=True, *, expected=None):
    audio = compare_audio(baseline.audio if baseline else None, candidate.audio if candidate else None,
                          comparable, expected)
    same_input = not audio or audio[0]["status"] != "different"
    decode = compare_steps(baseline.trace if baseline else (), candidate.trace if candidate else (),
                           comparable and same_input)
    if not same_input:
        for row in decode:
            if row["status"] == "run_mismatch":
                row["status"] = "input_mismatch"
    return {"audio_evidence": audio, "decode_evidence": decode, "boundary": boundary_summary(audio, decode)}


def audio_inputs_agree(*outputs, expected=None):
    hashes = {output.audio.input_sha256 for output in outputs
              if output and output.audio and output.audio.input_sha256}
    if expected is not None:
        hashes.add(expected)
    return len(hashes) <= 1
