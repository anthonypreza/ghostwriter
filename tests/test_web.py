import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from ghostwriter.data.prepare import clean_lyrics
from ghostwriter.data.tokenizer import create_tokenizer
from ghostwriter.inference.generate import generate
from ghostwriter.models.registry import build_model
from ghostwriter.training.checkpoint import load_model, save_model
from ghostwriter.web.app import create_app


def make_run(root, name, level="word", sections=True):
    run = root / name
    train = run / "processed/train"
    train.mkdir(parents=True)
    (train / "synthetic.txt").write_text(
        clean_lyrics("[Verse]\nPaper sky\n\nLittle light", sections), encoding="utf-8"
    )
    tokenizer = create_tokenizer(level, sections, 300 if level == "bpe" else None)
    tokenizer.build_vocab(train, min_freq=1)
    tokenizer.save(run / "tokenizer.json")
    model = build_model(
        "gru",
        len(tokenizer.vocab_map),
        {"embedding_size": 8, "hidden_size": 12, "dropout": 0.0},
    )
    save_model(model, tokenizer, run / "best_gru.pt", "gru", model.model_config, 16, 1)
    (run / "config.json").write_text(
        json.dumps(
            {"model": "gru", "seq_len": 16, "model_settings": model.model_config}
        ),
        encoding="utf-8",
    )
    (run / "metrics.json").write_text(
        json.dumps({"evaluation": {"val": {"perplexity": 12.5}}}), encoding="utf-8"
    )
    return run


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run = make_run(self.root, "synthetic-word")
        self.client = TestClient(create_app(self.root), base_url="http://127.0.0.1")
        self.payload = {
            "run": self.run.name,
            "checkpoint": "best_gru.pt",
            "seed": "123",
            "max_new_tokens": 8,
        }

    def tearDown(self):
        self.client.close()
        self.temporary.cleanup()

    def test_list_runs_static_assets_and_tokenizer_defaults(self):
        make_run(self.root, "synthetic-char", "char", False)
        make_run(self.root, "synthetic-bpe", "bpe", False)
        (self.root / "unfinished").mkdir()
        data = self.client.get("/api/runs").json()
        self.assertEqual(len(data["runs"]), 3)
        by_name = {run["id"]: run for run in data["runs"]}
        self.assertEqual(
            by_name["synthetic-char"]["tokenizers"][0]["max_new_tokens"], 1000
        )
        self.assertEqual(by_name["synthetic-bpe"]["tokenizers"][0]["prompt"], "")
        self.assertEqual(
            by_name["synthetic-word"]["tokenizers"][0]["prompt"], "<INTRO>"
        )
        self.assertEqual(
            by_name["synthetic-word"]["evaluation"]["val"]["perplexity"], 12.5
        )
        for path in ("/", "/style.css", "/app.js"):
            self.assertEqual(self.client.get(path).status_code, 200)

    def test_api_matches_cli_generation_and_preserves_seed_precision(self):
        for level in ("word", "char", "bpe"):
            with self.subTest(level=level):
                run = (
                    self.run
                    if level == "word"
                    else make_run(self.root, level, level, False)
                )
                body = {
                    **self.payload,
                    "run": run.name,
                    "prompt": "Paper sky",
                    "seed": str(2**63 - 2),
                    "temperature": 0.9,
                    "top_k": 5,
                    "color": "never",
                    "plain": True,
                }
                response = self.client.post("/api/generate", json=body)
                self.assertEqual(response.status_code, 200, response.text)
                data = response.json()
                model, tokenizer = load_model(
                    run / "best_gru.pt", run / "tokenizer.json"
                )
                expected = generate(
                    model,
                    tokenizer,
                    prompt=body["prompt"],
                    max_new_tokens=8,
                    temperature=0.9,
                    top_k=5,
                    seed=int(body["seed"]),
                )
                self.assertEqual(data["text"], expected)
                self.assertEqual(data["settings"]["seed"], body["seed"])
                self.assertEqual(data["settings"]["context_length"], 16)
                self.assertIn("--temperature 0.9", data["command"])
                self.assertIn("--plain", data["command"])
                self.assertIn(body["seed"], data["command"])

    def test_validation_context_mismatch_and_cache(self):
        with patch("ghostwriter.web.app.load_model", wraps=load_model) as loader:
            response = self.client.post(
                "/api/generate", json={**self.payload, "context_length": 8}
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("trained with context length 16", response.json()["detail"])
            self.assertEqual(
                self.client.post("/api/generate", json=self.payload).status_code, 200
            )
            self.assertEqual(loader.call_count, 1)
        for options in (
            {"temperature": -1},
            {"top_k": -1},
            {"seed": "invalid"},
            {"seed": str(2**63)},
            {"max_new_tokens": -2},
        ):
            self.assertEqual(
                self.client.post(
                    "/api/generate", json={**self.payload, **options}
                ).status_code,
                422,
            )
        with patch("ghostwriter.web.app.torch.cuda.is_available", return_value=False):
            self.assertEqual(
                self.client.post(
                    "/api/generate", json={**self.payload, "device": "cuda"}
                ).status_code,
                400,
            )

    def test_filesystem_origin_and_host_boundaries(self):
        for options in (
            {"run": ".."},
            {"checkpoint": "../outside.pt"},
            {"tokenizer": "../outside.json"},
        ):
            self.assertEqual(
                self.client.post(
                    "/api/generate", json={**self.payload, **options}
                ).status_code,
                400,
            )
        self.assertEqual(
            self.client.post(
                "/api/generate",
                json=self.payload,
                headers={"origin": "https://example.com"},
            ).status_code,
            403,
        )
        self.assertEqual(
            self.client.get("/api/runs", headers={"host": "example.com"}).status_code,
            400,
        )
        self.assertEqual(self.client.get("/tokenizer.json").status_code, 404)

    def test_empty_and_corrupt_runs_do_not_break_browser(self):
        self.assertEqual(
            TestClient(create_app(self.root / "missing"), base_url="http://127.0.0.1")
            .get("/api/runs")
            .json()["runs"],
            [],
        )
        (self.run / "metrics.json").write_text("broken", encoding="utf-8")
        self.assertEqual(self.client.get("/api/runs").status_code, 200)
        self.assertEqual(
            self.client.post(
                "/api/generate", json={**self.payload, "checkpoint": "missing.pt"}
            ).status_code,
            404,
        )
        (self.run / "broken.pt").write_bytes(b"invalid checkpoint")
        self.assertEqual(
            self.client.post(
                "/api/generate", json={**self.payload, "checkpoint": "broken.pt"}
            ).status_code,
            400,
        )
        # Errors release the generation lock, allowing a valid request afterward.
        self.assertEqual(
            self.client.post("/api/generate", json=self.payload).status_code, 200
        )


if __name__ == "__main__":
    unittest.main()
