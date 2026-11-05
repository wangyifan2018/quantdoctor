"""Fingerprint a uniform raw-audio ASR request file across different cohorts."""

from dataclasses import asdict, dataclass
from pathlib import PurePosixPath

from .model import SHA256, document_sha256, nonempty, require


# Defaults from parseRequestFile at Edge-LLM 1231d64. CLI overrides are external.
FILE_DEFAULTS = {
    "batch_size": 1, "max_generate_length": 256, "num_logprobs": 0,
    "temperature": 1.0, "top_p": 0.8, "top_k": 50, "sampling_seed": None,
    "apply_chat_template": True, "add_generation_prompt": True, "enable_thinking": False,
    "spec_proposal_sampling": "auto", "context_cache_lookup_policy": "use_cache",
    "context_cache_commit_policy": "including_generated_tokens",
}
RAW_AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3")
_NO_REFERENCE = object()


def _settings(document):
    settings = {}
    for key, default in FILE_DEFAULTS.items():
        value = document.get(key, default)
        if default is None:
            if value is not None and not (type(value) is int and 0 <= value <= 2**64 - 1):
                return None
        elif type(default) is float:
            if type(value) not in (int, float) or not -3.4e38 <= value <= 3.4e38:
                return None
            value = float(value) + 0.0
        elif type(value) is not type(default):
            return None
        settings[key] = value
    return settings


def _message_template(request):
    if not isinstance(request, dict) or not isinstance(request.get("messages"), list):
        return None
    messages = []
    audio_count = 0
    for message in request["messages"]:
        if not isinstance(message, dict) or not nonempty(message.get("role")):
            return None
        content = message.get("content")
        if isinstance(content, str):
            messages.append(message)
            continue
        if not isinstance(content, list):
            return None
        items = []
        for item in content:
            if not isinstance(item, dict):
                return None
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                items.append(item)
            elif item.get("type") == "audio" and nonempty(item.get("audio")):
                if PurePosixPath(item["audio"]).suffix not in RAW_AUDIO_EXTENSIONS:
                    return None
                items.append({**item, "audio": None})
                audio_count += 1
            else:
                return None
        messages.append({**message, "content": items})
    return messages if audio_count == 1 else None


@dataclass(frozen=True)
class RequestProtocol:
    schema_version: int
    settings: dict
    additional_settings: dict[str, str]
    messages_sha256: str
    request_options: dict[str, str]
    sha256: str

    @classmethod
    def from_document(cls, document):
        if not isinstance(document, dict) or not isinstance(document.get("requests"), list) or not document["requests"]:
            return None
        settings = _settings(document)
        if settings is None:
            return None
        template = None
        for request in document["requests"]:
            messages = _message_template(request)
            if messages is None:
                return None
            options = {key: document_sha256(value) for key, value in request.items() if key != "messages"}
            sections = (document_sha256(messages), options)
            if template is not None and template != sections:
                return None
            template = sections
        extras = {key: document_sha256(value) for key, value in document.items()
                  if key != "requests" and key not in FILE_DEFAULTS}
        evidence = {"schema_version": 1, "settings": settings, "additional_settings": extras,
                    "messages_sha256": template[0], "request_options": template[1]}
        return cls(**evidence, sha256=document_sha256(evidence))

    @classmethod
    def parse(cls, data):
        if data is None:
            return None
        require(isinstance(data, dict) and data.keys() == cls.__dataclass_fields__.keys()
                and type(data["schema_version"]) is int and data["schema_version"] == 1,
                "invalid request protocol schema")
        settings = data["settings"]
        require(isinstance(settings, dict) and settings.keys() == FILE_DEFAULTS.keys(), "invalid request protocol settings")
        normalized = _settings(settings)
        require(normalized is not None, "invalid request protocol setting values")
        require(all(isinstance(data[key], str) and SHA256.fullmatch(data[key]) for key in ("messages_sha256", "sha256")),
                "invalid request protocol fingerprint")
        for key in ("additional_settings", "request_options"):
            fields = data[key]
            require(isinstance(fields, dict) and all(isinstance(name, str) and isinstance(value, str)
                                                     and SHA256.fullmatch(value) for name, value in fields.items()),
                    f"invalid request protocol fingerprints for {key}")
        payload = {key: value for key, value in data.items() if key != "sha256"}
        payload["settings"] = normalized
        require(document_sha256(payload) == data["sha256"], "request protocol fingerprint does not match its fields")
        return cls(**payload, sha256=data["sha256"])

    def differences(self, other):
        changes = [key for key in FILE_DEFAULTS if self.settings[key] != other.settings[key]]
        for left, right, prefix in ((self.additional_settings, other.additional_settings, ""),
                                    (self.request_options, other.request_options, "requests.")):
            changes.extend(prefix + key for key in sorted(left.keys() | right.keys()) if left.get(key) != right.get(key))
        if self.messages_sha256 != other.messages_sha256:
            changes.append("requests.messages")
        return changes


def inspect_request_protocol(document, reference=_NO_REFERENCE):
    first = RequestProtocol.from_document(document)
    report = {"scope": "request_file_only", "status": "read" if first else "unsupported",
              "protocol": asdict(first) if first else None}
    if reference is not _NO_REFERENCE:
        second = RequestProtocol.from_document(reference)
        differences = first.differences(second) if first and second else None
        report.update(reference=asdict(second) if second else None, differences=differences,
                      status="unsupported" if differences is None else "different" if differences else "matching")
    return report
