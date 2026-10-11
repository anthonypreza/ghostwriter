"""A local API around the existing checkpoint loader and generation function."""

import gc
import json
import math
import pickle
from pathlib import Path
import secrets
import shlex
import sys
from threading import Lock
import time
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
import torch

from ..inference.generate import generate
from ..training.checkpoint import PROJECT_ROOT, load_model, resolve_context_length


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    run: str
    checkpoint: str
    tokenizer: str = "tokenizer.json"
    prompt: str | None = Field(default=None, max_length=20000)
    max_new_tokens: int | None = Field(default=None, ge=0)
    temperature: float = Field(default=1.0, ge=0)
    top_k: int = Field(default=40, ge=0)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    context_length: int | None = Field(default=None, ge=1)
    device: Literal["cpu", "cuda"] = "cpu"
    color: Literal["auto", "always", "never"] = "auto"
    plain: bool = False


def read_json(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def child_path(parent, name, directory=False):
    """Only direct children of the selected local directory may be used."""
    if (
        not name
        or name in (".", "..")
        or any(char in name for char in ("/", "\\", "\0", ":"))
    ):
        raise HTTPException(400, "Choose a file or run from the available list.")
    path = (parent / name).resolve()
    if not name or path.parent != parent or path.name != name:
        raise HTTPException(400, "Choose a file or run from the available list.")
    if not (path.is_dir() if directory else path.is_file()):
        raise HTTPException(
            404, "The selected run or file no longer exists. Refresh runs."
        )
    return path


def run_metadata(path):
    config = read_json(path / "config.json")
    metrics = read_json(path / "metrics.json")
    checkpoints = [
        p.name
        for p in sorted(path.glob("*.pt"))
        if p.resolve().parent == path and p.is_file()
    ]
    tokenizers = []
    for candidate in sorted(path.glob("*.json")):
        if candidate.resolve().parent != path:
            continue
        data = read_json(candidate)
        if data.get("version") in (1, 2, 3) and isinstance(data.get("tokens"), list):
            sections = data.get("section_tokens", True)
            level = data.get("token_level", "word")
            tokenizers.append(
                {
                    "name": candidate.name,
                    "token_level": level,
                    "section_tokens": sections,
                    "vocab_size": len(data["tokens"]),
                    "prompt": "<INTRO>" if sections else "",
                    "max_new_tokens": 1000 if level == "char" else 200,
                }
            )
    if not checkpoints or not tokenizers:
        return None
    return json_safe(
        {
            "id": path.name,
            "model": config.get("model")
            or Path(checkpoints[0]).stem.removeprefix("best_"),
            "checkpoints": checkpoints,
            "tokenizers": tokenizers,
            "seq_len": config.get("seq_len"),
            "model_settings": config.get("model_settings", {}),
            "training_settings": config.get("training_settings", {}),
            "evaluation": metrics.get("evaluation", {}),
            "best_epoch": metrics.get("best_epoch"),
        }
    )


def cli_command(checkpoint, tokenizer, settings):
    quote = (
        (lambda value: "'" + str(value).replace("'", "''") + "'")
        if sys.platform == "win32"
        else shlex.quote
    )
    command = (
        f"python -m ghostwriter.inference.generate --checkpoint {quote(str(checkpoint))} "
        f"--tokenizer {quote(str(tokenizer))} --prompt {quote(settings['prompt'])} "
        f"--max-new-tokens {settings['max_new_tokens']} --temperature {settings['temperature']} "
        f"--top-k {settings['top_k']} --seed {settings['seed']} --device {settings['device']} "
        f"--color {settings['color']}"
    )
    if settings["legacy_context"]:
        command += f" --context-length {settings['context_length']}"
    if settings["plain"]:
        command += " --plain"
    return command


def create_app(runs_dir=PROJECT_ROOT / "runs"):
    root = Path(runs_dir).resolve()
    app = FastAPI(title="Ghostwriter playground", docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
    lock = Lock()
    cache = {"key": None, "model": None, "tokenizer": None}

    @app.get("/api/runs")
    def list_runs():
        runs = []
        if root.is_dir():
            for path in sorted(root.iterdir(), reverse=True):
                if path.is_dir() and path.resolve().parent == root:
                    metadata = run_metadata(path.resolve())
                    if metadata:
                        runs.append(metadata)
        return {
            "runs": runs,
            "devices": ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
        }

    @app.post("/api/generate")
    def generate_text(body: GenerationRequest, request: Request):
        origin = request.headers.get("origin")
        if origin and urlsplit(origin).netloc != request.url.netloc:
            raise HTTPException(
                403, "Generation requests must come from this playground."
            )
        run = child_path(root, body.run, directory=True)
        checkpoint = child_path(run, body.checkpoint)
        tokenizer_path = child_path(run, body.tokenizer)
        if checkpoint.suffix != ".pt" or tokenizer_path.suffix != ".json":
            raise HTTPException(400, "Select a .pt checkpoint and a .json tokenizer.")
        if body.device == "cuda" and not torch.cuda.is_available():
            raise HTTPException(
                400,
                "GPU execution is unavailable in this server environment. Select CPU.",
            )
        if not lock.acquire(blocking=False):
            raise HTTPException(
                409, "Another generation is running. Try again when it finishes."
            )
        try:
            started = time.perf_counter()
            key = (
                str(checkpoint),
                checkpoint.stat().st_mtime_ns,
                checkpoint.stat().st_size,
                str(tokenizer_path),
                tokenizer_path.stat().st_mtime_ns,
                tokenizer_path.stat().st_size,
                body.device,
            )
            if key != cache["key"]:
                cache.update(key=None, model=None, tokenizer=None)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                model, tokenizer = load_model(checkpoint, tokenizer_path, body.device)
                cache.update(key=key, model=model, tokenizer=tokenizer)
            model, tokenizer = cache["model"], cache["tokenizer"]
            legacy = model.context_length is None
            context = resolve_context_length(model, body.context_length)
            settings = {
                **body.model_dump(),
                "prompt": body.prompt
                if body.prompt is not None
                else ("<INTRO>" if tokenizer.section_tokens else ""),
                "seed": str(
                    body.seed if body.seed is not None else secrets.randbits(63)
                ),
                "max_new_tokens": body.max_new_tokens
                if body.max_new_tokens is not None
                else (1000 if tokenizer.token_level == "char" else 200),
                "context_length": context,
                "legacy_context": legacy,
                "token_level": tokenizer.token_level,
                "section_tokens": tokenizer.section_tokens,
            }
            text = generate(
                model,
                tokenizer,
                prompt=settings["prompt"],
                max_new_tokens=settings["max_new_tokens"],
                temperature=body.temperature,
                top_k=body.top_k,
                context_length=context,
                seed=int(settings["seed"]),
            )
            return {
                "text": text,
                "settings": settings,
                "model_settings": model.model_config,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "command": cli_command(checkpoint, tokenizer_path, settings),
            }
        except HTTPException:
            raise
        except (
            OSError,
            ValueError,
            RuntimeError,
            KeyError,
            TypeError,
            pickle.UnpicklingError,
            EOFError,
        ) as error:
            raise HTTPException(400, f"Could not generate: {error}") from error
        finally:
            lock.release()

    app.mount(
        "/",
        StaticFiles(directory=Path(__file__).parent / "static", html=True),
        name="frontend",
    )
    return app
