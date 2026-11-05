from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import unittest

from quantdoctor.edge import import_edge
from quantdoctor.model import Comparison, document_sha256, load_json
from quantdoctor.request_protocol import RequestProtocol, inspect_request_protocol


def request(path):
    return {"messages": [{"role": "system", "content": "Transcribe."},
                         {"role": "user", "content": [{"type": "audio", "audio": path}]}]}


class RequestProtocolTests(unittest.TestCase):
    def test_document_scope_matches_defaults_across_different_audio_and_cohort_sizes(self):
        diagnosis = {"requests": [request("/diagnosis/a.wav")]}
        holdout = {"requests": [request("/holdout/b.flac"), request("/holdout/c.mp3")],
                   "temperature": 1, "num_logprobs": 0, "top_k": 50, "max_generate_length": 256}
        audit = inspect_request_protocol(diagnosis, holdout)
        self.assertEqual(audit["scope"], "request_file_only")
        self.assertEqual(audit["status"], "matching")
        self.assertEqual(audit["protocol"]["sha256"], audit["reference"]["sha256"])
        self.assertNotEqual(document_sha256(diagnosis), document_sha256(holdout))

    def test_differences_include_generation_limit_prompt_stop_and_other_file_options(self):
        diagnosis = {"requests": [request("a.wav")]}
        holdout = deepcopy(diagnosis)
        holdout["max_generate_length"] = 32
        holdout["guided_decoding"] = {"choice": ["yes"]}
        holdout["requests"][0]["messages"][0]["content"] = "Summarize."
        holdout["requests"][0]["stop"] = ["."]
        audit = inspect_request_protocol(diagnosis, holdout)
        self.assertEqual(audit["status"], "different")
        self.assertEqual(audit["differences"], ["max_generate_length", "guided_decoding", "requests.stop", "requests.messages"])
        self.assertNotIn("Transcribe.", str(audit))

    def test_mixed_templates_and_nonraw_audio_leave_no_signature(self):
        mixed = {"requests": [request("a.wav"), {**request("b.wav"), "stop": [";"]}]}
        self.assertIsNone(RequestProtocol.from_document(mixed))
        self.assertIsNone(RequestProtocol.from_document({"requests": [request("mel.safetensors")]}))
        self.assertEqual(inspect_request_protocol({"requests": [request("a.wav")]}, None)["status"], "unsupported")

    def test_plain_import_only_attaches_protocol_from_the_matching_request_file(self):
        manifest = load_json(Path(__file__).resolve().parents[1] / "examples" / "demo-comparison.json")
        document = {"requests": [request("a.wav")]}
        manifest["request_document_sha256"] = document_sha256(document)
        manifest["samples"] = [{"id": "clip", "request_idx": 0}]
        manifest["request_protocol"] = asdict(RequestProtocol.from_document(document))
        empty = {"responses": []}
        bundle = import_edge(manifest, empty, empty, request_document=document)
        self.assertEqual(Comparison.parse(bundle).request_protocol.settings["batch_size"], 1)
        self.assertNotIn("request_protocol", import_edge(manifest, empty, empty))
        with self.assertRaisesRegex(ValueError, "request file does not match"):
            import_edge(manifest, empty, empty, request_document={**document, "top_k": 1})
        bundle["request_protocol"]["settings"]["top_k"] = 1
        with self.assertRaisesRegex(ValueError, "fingerprint does not match its fields"):
            Comparison.parse(bundle)
