"""Local ASR split manifests and adapters for the Edge-LLM dataset boundary."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
import wave

from .model import Comparison, document_sha256, file_sha256, load_json, nonempty, parse_speaker_id, require


@dataclass(frozen=True)
class AudioSample:
    id: str
    split: str
    audio: PurePosixPath
    transcript: str
    language: str
    sha256: str
    speaker_id: str | None = None


@dataclass(frozen=True)
class ASRDataset:
    root: Path
    samples: tuple[AudioSample, ...]

    def select(self, split):
        return tuple(row for row in self.samples if row.split == split)

    def digest(self, split):
        rows = [[row.id, row.sha256, row.transcript, row.language, *([row.speaker_id] if row.speaker_id else [])]
                for row in self.select(split)]
        return document_sha256({"schema_version": 1, "split": split, "samples": rows})

    def summary(self):
        warnings = []
        calibration = self.select("calibration")
        if len(calibration) > 128:
            warnings.append("Pinned Edge-LLM ASR calibration uses at most 128 accepted examples in order.")
        if any(row.language != "en" for row in calibration):
            warnings.append("Pinned Edge-LLM ASR calibration uses an English prompt prefix even with custom transcripts.")
        if any(row.audio.suffix.lower() == ".flac" for row in calibration):
            warnings.append("FLAC sample rate and duration are checked by the device-side calibration loader.")
        missing_speakers = sum(row.speaker_id is None for row in self.samples)
        both_splits = bool(calibration and self.select("evaluation"))
        if both_splits and missing_speakers:
            warnings.append("Speaker isolation is incomplete because some samples have no speaker_id.")
        return {
            "splits": {
                split: {"count": len(rows), "sha256": self.digest(split),
                        "speaker_count": len({row.speaker_id for row in rows if row.speaker_id is not None}),
                        "missing_speaker_ids": sum(row.speaker_id is None for row in rows)}
                for split in ("calibration", "evaluation") if (rows := self.select(split))
            },
            "speaker_isolation": ("single_split" if not both_splits else
                                  "incomplete" if missing_speakers else "disjoint_declared_ids"),
            "warnings": warnings,
        }


def _audio_path(value, root, sample_id):
    require(nonempty(value) and "\\" not in value, f"{sample_id}: use a relative POSIX audio path")
    path = PurePosixPath(value)
    require(path.parts and not path.is_absolute() and ".." not in path.parts,
            f"{sample_id}: audio path must stay inside the manifest directory")
    resolved = (root / path).resolve()
    require(resolved.is_relative_to(root) and resolved.is_file() and resolved.stat().st_size > 0,
            f"{sample_id}: expected a nonempty audio file inside the manifest directory")
    return path, resolved


def _check_calibration_wav(path, sample_id):
    try:
        with wave.open(str(path), "rb") as clip:
            require(clip.getframerate() == 16000 and 0 < clip.getnframes() <= 320000,
                    f"{sample_id}: Edge-LLM calibration requires 16 kHz WAV of at most 20 seconds")
    except (wave.Error, EOFError) as error:
        raise ValueError(f"{sample_id}: calibration WAV must be a readable PCM WAV") from error


def load_asr_dataset(manifest: Path):
    document = load_json(manifest)
    require(isinstance(document, dict) and type(document.get("schema_version")) is int
            and document["schema_version"] == 1, "expected ASR dataset schema_version 1")
    rows = document.get("samples")
    require(isinstance(rows, list) and rows, "ASR dataset requires a nonempty samples array")
    root = manifest.parent.resolve()
    ids = set()
    hashes = {}
    audio_splits = {}
    speaker_splits = {}
    samples = []
    for row in rows:
        require(isinstance(row, dict) and nonempty(row.get("id")), "sample id is required")
        sample_id = row["id"]
        require(sample_id not in ids, f"duplicate sample id {sample_id}")
        ids.add(sample_id)
        split, transcript, language = (row.get(key) for key in ("split", "text", "language"))
        require(split in ("calibration", "evaluation"), f"{sample_id}: invalid split")
        require(nonempty(transcript), f"{sample_id}: reference transcript is required")
        require(language in ("zh", "en"), f"{sample_id}: language must be en or zh")
        path, resolved = _audio_path(row.get("audio"), root, sample_id)
        extensions = (".wav", ".flac") if split == "calibration" else (".wav", ".flac", ".mp3")
        require(path.suffix.lower() in extensions, f"{sample_id}: allowed audio formats are {', '.join(extensions)}")
        if split == "calibration" and path.suffix.lower() == ".wav":
            _check_calibration_wav(resolved, sample_id)
        if resolved not in hashes:
            hashes[resolved] = file_sha256(resolved)
        digest = hashes[resolved]
        require(audio_splits.setdefault(digest, split) == split,
                f"{sample_id}: identical audio is present in both calibration and evaluation")
        speaker = parse_speaker_id(row.get("speaker_id"))
        if speaker is not None:
            require(speaker_splits.setdefault(speaker, split) == split,
                    f"{sample_id}: speaker {speaker!r} is present in both calibration and evaluation")
        samples.append(AudioSample(sample_id, split, path, transcript.strip(), language, digest, speaker))
    return ASRDataset(root, tuple(samples))


def calibration_source(manifest: Path | str):
    """Return the zero-argument audio_dataset callable used by Edge-LLM PTQ."""
    dataset = load_asr_dataset(Path(manifest))
    samples = dataset.select("calibration")
    require(samples, "manifest has no calibration samples")

    def generate():
        for sample in samples:
            audio = (dataset.root / sample.audio).read_bytes()
            require(sha256(audio).hexdigest() == sample.sha256, f"{sample.id}: audio changed after validation")
            yield audio, sample.transcript

    generate.calib_name = "quantdoctor:" + dataset.digest("calibration")
    return generate


def prepare_evaluation(dataset: ASRDataset, metadata, audio_root: PurePosixPath | None = None):
    """Prepare a greedy Edge request file and the matching scoring manifest."""
    require(isinstance(metadata, dict), "comparison metadata must be an object")
    samples = dataset.select("evaluation")
    require(samples, "manifest has no evaluation samples")
    if audio_root is not None:
        require(audio_root.is_absolute() and ".." not in audio_root.parts, "target audio root must be absolute")
    root = audio_root if audio_root is not None else dataset.root
    requests = []
    index = []
    for offset, sample in enumerate(samples):
        requests.append({"messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio", "audio": str(root / sample.audio)}]},
        ]})
        index.append({"id": sample.id, "request_idx": offset,
                      "reference": sample.transcript, "language": sample.language, "audio_sha256": sample.sha256})
        if sample.speaker_id is not None:
            index[-1]["speaker_id"] = sample.speaker_id
    request = {"batch_size": 1, "temperature": 1.0, "top_k": 1, "top_p": 1.0,
               "max_generate_length": 256, "requests": requests}
    scoring = {**metadata, "samples": index, "evaluation_dataset_sha256": dataset.digest("evaluation"),
               "request_document_sha256": document_sha256(request)}
    Comparison.parse(scoring)
    return request, scoring
