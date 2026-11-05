"""Compact decoder evidence and alignment under a shared generation prefix."""

from dataclasses import dataclass
import math

from .model import SHA256, require, valid_ids


@dataclass(frozen=True)
class TokenScore:
    id: int
    logit: float


@dataclass(frozen=True)
class DecodeStep:
    step: int
    context_sha256: str
    score_stage: str
    selected_id: int
    top1_id: int
    stop_ids: tuple[int, ...]
    scores: tuple[TokenScore, ...]

    @classmethod
    def parse(cls, row):
        require(isinstance(row, dict), "decode step must be an object")
        step, context, stage = (row.get(key) for key in ("step", "context_sha256", "score_stage"))
        selected, top1, stops = (row.get(key) for key in ("selected_id", "top1_id", "stop_ids"))
        require(type(step) is int and step >= 0, "decode step index must be nonnegative")
        require(isinstance(context, str) and SHA256.fullmatch(context), "decode context requires SHA-256")
        require(stage in ("raw_logits", "processed_logits"), "unsupported score stage")
        require(valid_ids([selected, top1]) and valid_ids(stops) and stops and len(set(stops)) == len(stops),
                "decode trace requires selected/top-1 IDs and unique effective stop IDs")
        scores = row.get("scores")
        require(isinstance(scores, list) and len(scores) <= 256, "expected up to 256 token scores per step")
        seen = set()
        parsed = []
        for entry in scores:
            require(isinstance(entry, dict), "token score must be an object")
            token, value = entry.get("id"), entry.get("logit")
            require(valid_ids([token]) and token not in seen, "invalid or duplicate scored token ID")
            require(type(value) in (int, float) and -1e12 <= value <= 1e12 and math.isfinite(value),
                    "logit must be finite and within the supported range")
            seen.add(token)
            parsed.append(TokenScore(token, float(value)))
        return cls(step, context, stage, selected, top1, tuple(sorted(stops)), tuple(parsed))


def parse_steps(rows):
    require(isinstance(rows, (list, tuple)) and len(rows) <= 512, "expected up to 512 trace steps")
    steps = tuple(DecodeStep.parse(row) for row in rows)
    require(all(a.step < b.step for a, b in zip(steps, steps[1:])), "trace steps must be unique and increasing")
    return steps


def _margin(step, reference_id):
    scores = {score.id: score.logit for score in step.scores}
    if reference_id in step.stop_ids or reference_id not in scores or not all(
        token in scores for token in step.stop_ids
    ):
        return None
    return scores[reference_id] - max(scores[token] for token in step.stop_ids)


def compare_steps(baseline, candidate, comparable=True):
    """Show only like-for-like margins and observed token decisions."""
    left = {row.step: row for row in baseline}
    right = {row.step: row for row in candidate}
    rows = []
    for index in sorted(left.keys() | right.keys()):
        first, second = left.get(index), right.get(index)
        entry = {
            "step": index,
            "baseline_selected": first.selected_id if first else None,
            "candidate_selected": second.selected_id if second else None,
            "baseline_top1": first.top1_id if first else None,
            "candidate_top1": second.top1_id if second else None,
        }
        if first is None or second is None:
            status = "missing_step"
        elif not comparable:
            status = "run_mismatch"
        elif first.context_sha256 != second.context_sha256:
            status = "context_mismatch"
        elif first.score_stage != second.score_stage:
            status = "score_stage_mismatch"
        elif first.stop_ids != second.stop_ids:
            status = "stop_set_mismatch"
        else:
            baseline_margin = _margin(first, first.selected_id)
            candidate_margin = _margin(second, first.selected_id)
            status = "aligned" if baseline_margin is not None and candidate_margin is not None else "missing_scores"
            entry.update({
                "reference_id": first.selected_id, "score_stage": first.score_stage,
                "selected_changed": first.selected_id != second.selected_id,
                "candidate_stopped": second.selected_id in second.stop_ids,
                "baseline_margin": baseline_margin if status == "aligned" else None,
                "candidate_margin": candidate_margin if status == "aligned" else None,
            })
        rows.append({**entry, "status": status})
    return rows
