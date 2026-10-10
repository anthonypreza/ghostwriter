"""Browse past runs, rank comparable validation results, and reuse checkpoints."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

from ..console import use_color

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SORT_KEYS = {
    "val-ppl": ("perplexity", False),
    "val-loss": ("loss", False),
    "val-accuracy": ("accuracy", True),
}


def read_json(path, notes):
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as error:
        notes.append(f"Cannot read {path.name}: {error}")
        return {}


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def comparison_group(run_dir, config, tokenizer, notes):
    """Fingerprint evaluation data and vocabulary, without exposing song text."""
    seq_len = config.get("seq_len")
    tokens = tokenizer.get("tokens")
    paths = sorted((run_dir / "processed" / "val").glob("*.txt"))
    if (
        not paths
        or not isinstance(tokens, list)
        or not tokens
        or not isinstance(seq_len, int)
    ):
        notes.append(
            "Comparison group unavailable: missing validation files, vocabulary, or sequence length."
        )
        return None
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"tokens": tokens, "seq_len": seq_len, "evaluation": "nonoverlapping-v1"},
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
    )
    try:
        for path in paths:
            digest.update(json.dumps(path.name).encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    except OSError as error:
        notes.append(f"Cannot fingerprint validation data: {error}")
        return None
    return digest.hexdigest()[:12]


def ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def inspect_run(run_dir):
    run_dir = run_dir.resolve()
    notes = []
    config = read_json(run_dir / "config.json", notes)
    metrics = read_json(run_dir / "metrics.json", notes)
    history = read_json(run_dir / "history.json", notes)
    tokenizer = read_json(run_dir / "tokenizer.json", notes)
    manifest = read_json(run_dir / "processed" / "splits.json", notes)
    model = metrics.get("model") or config.get("model")
    if not isinstance(model, str):
        model = None
    checkpoints = sorted(run_dir.glob("best_*.pt"))
    if not model and len(checkpoints) == 1:
        model = checkpoints[0].stem.removeprefix("best_")
    checkpoint = run_dir / f"best_{model}.pt" if model else None
    settings = config.get("model_settings") or {}
    if not isinstance(settings, dict):
        settings = {}
    # An interrupted run may have saved a checkpoint before updating config.
    if checkpoint and checkpoint.is_file() and not settings:
        try:
            import torch

            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            settings = saved.get("model_config", {})
        except Exception as error:
            notes.append(f"Checkpoint metadata unavailable: {type(error).__name__}")
    evaluation = metrics.get("evaluation", {})
    if not isinstance(evaluation, dict):
        evaluation = {}
    val = evaluation.get("val", {})
    test = evaluation.get("test", {})
    val = val if isinstance(val, dict) else {}
    test = test if isinstance(test, dict) else {}
    complete = all(finite_number(part.get("loss")) for part in (val, test))
    status = "complete" if complete else "incomplete"
    log = run_dir / "pipeline.log"
    if not complete and log.exists():
        try:
            tail = log.read_text(encoding="utf-8")[-16000:]
            if "FAILED:" in tail:
                status = "failed"
            elif "Pipeline interrupted" in tail:
                status = "interrupted"
        except OSError:
            pass
    losses = history.get("val_loss", [])
    losses = losses if isinstance(losses, list) else []
    best_epoch = metrics.get("best_epoch")
    if best_epoch is None:
        valid = [(i + 1, loss) for i, loss in enumerate(losses) if finite_number(loss)]
        best_epoch = min(valid, key=lambda pair: pair[1])[0] if valid else None
    counts = {
        split: len(manifest[split])
        if isinstance(manifest.get(split), list)
        else len(list((run_dir / "processed" / split).glob("*.txt")))
        for split in ("train", "val", "test")
    }
    tokenizer_path = run_dir / "tokenizer.json"
    commands = {}
    if checkpoint and checkpoint.is_file() and tokenizer_path.is_file():
        common = f"--checkpoint {ps_quote(checkpoint)} --tokenizer {ps_quote(tokenizer_path)}"
        commands = {
            "generate": f"python -m ghostwriter.inference.generate {common} --prompt '<INTRO>' --max-new-tokens 200 --temperature 1.1",
            "evaluate": f"python -m ghostwriter.training.eval {common} --data-dir {ps_quote(run_dir / 'processed')} --split test",
        }
    else:
        notes.append("Generation unavailable: checkpoint or tokenizer missing.")
    group = comparison_group(run_dir, config, tokenizer, notes)
    vocabulary = tokenizer.get("tokens", [])
    return {
        "run": run_dir.name,
        "path": str(run_dir),
        "model": model or "unknown",
        "status": status,
        "config": config,
        "model_settings": settings,
        "songs": counts,
        "vocab_size": len(vocabulary) if isinstance(vocabulary, list) else 0,
        "best_epoch": best_epoch,
        "epochs_completed": metrics.get("epochs_completed", len(losses)),
        "elapsed_seconds": metrics.get("elapsed_seconds"),
        "val": val,
        "test": test,
        "group": group,
        "rank": None,
        "commands": commands,
        "notes": notes,
    }


def rank_runs(runs, sort_by):
    metric, descending = SORT_KEYS[sort_by]
    groups = defaultdict(list)
    for run in runs:
        run["rank"] = None
        if (
            run["status"] == "complete"
            and run["group"]
            and finite_number(run["val"].get(metric))
        ):
            groups[run["group"]].append(run)
    for members in groups.values():
        ordered = sorted(
            members,
            key=lambda run: (
                (-1 if descending else 1) * run["val"][metric],
                run["run"],
            ),
        )
        previous, rank = None, 0
        for index, run in enumerate(ordered, 1):
            value = run["val"][metric]
            if value != previous:
                rank = index
            run["rank"], previous = rank, value
    return sorted(
        runs,
        key=lambda run: (
            run["rank"] is None,
            run["group"] or "~",
            run["rank"] or 0,
            run["run"],
        ),
    )


def number(value, percent=False):
    if not finite_number(value):
        return "-"
    return f"{value:.2%}" if percent else f"{value:.3f}"


def table(runs):
    headers = [
        "Run",
        "Model",
        "Status",
        "Group",
        "Rank",
        "Best/epochs",
        "Songs T/V/T",
        "Seq",
        "Vocab",
        "Val loss",
        "Val PPL",
        "Val acc",
        "Test PPL",
    ]
    rows = []
    for run in runs:
        rows.append(
            [
                run["run"],
                run["model"],
                run["status"],
                run["group"] or "-",
                str(run["rank"] or "-"),
                f"{run['best_epoch'] or '-'}/{run['epochs_completed']}",
                "/".join(str(run["songs"][s]) for s in ("train", "val", "test")),
                str(run["config"].get("seq_len", "-")),
                str(run["vocab_size"]),
                number(run["val"].get("loss")),
                number(run["val"].get("perplexity")),
                number(run["val"].get("accuracy"), True),
                number(run["test"].get("perplexity")),
            ]
        )
    return headers, rows


def details(run):
    lines = [
        f"Run: {run['run']} | Model: {run['model']} | Status: {run['status']}",
        f"Directory: {run['path']}",
        "Model settings: "
        + json.dumps(run["model_settings"], ensure_ascii=False, sort_keys=True),
        "Training/data settings: "
        + json.dumps(
            {
                key: value
                for key, value in run["config"].items()
                if key not in ("model_settings", "color", "model_config")
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        f"Elapsed: {number(run['elapsed_seconds'])} seconds",
        "Validation: " + json.dumps(run["val"], sort_keys=True),
        "Test: " + json.dumps(run["test"], sort_keys=True),
    ]
    for note in run["notes"]:
        lines.append("Note: " + note)
    for name, command in run["commands"].items():
        lines.extend([name.capitalize() + ":", command])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--model", help="Filter by model name.")
    parser.add_argument("--run", help="Inspect one exact run directory name.")
    parser.add_argument("--sort", choices=tuple(SORT_KEYS), default="val-ppl")
    parser.add_argument(
        "--top",
        type=int,
        help="Show the top N ranked runs per comparison group; keep incomplete runs visible.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print settings and reuse commands for each displayed run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write a Markdown report; defaults to <runs-dir>/summary.md.",
    )
    parser.add_argument("--color", choices=("auto", "always", "never"), default="auto")
    args = parser.parse_args()
    if args.top is not None and args.top < 1:
        parser.error("--top must be positive.")
    if not args.runs_dir.is_dir():
        parser.error(f"Run directory not found: {args.runs_dir}")
    runs = [
        inspect_run(path)
        for path in sorted(args.runs_dir.iterdir())
        if path.is_dir()
        and (not args.run or path.name == args.run)
        and any(
            (path / name).exists()
            for name in ("config.json", "metrics.json", "history.json", "pipeline.log")
        )
    ]
    if args.model:
        runs = [run for run in runs if run["model"].casefold() == args.model.casefold()]
    if not runs:
        parser.error("No matching runs found.")
    ranked = rank_runs(runs, args.sort)
    shown = [
        run
        for run in ranked
        if args.top is None or run["rank"] is None or run["rank"] <= args.top
    ]
    title = f"Ghostwriter runs | {len(shown)}/{len(runs)} displayed | Ranking: {args.sort} within each group"
    note = (
        "Ranks apply only within groups sharing saved validation content, vocabulary, and evaluation window length. "
        "Group IDs differ when those change; ranks do not establish a winner across groups. "
        "Missing final metrics means incomplete, running, or interrupted; checkpoint presence alone does not prove completion."
    )
    headers, rows = table(shown)
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    colored = use_color(args.color, sys.stdout)
    print(f"\033[1;36m{title}\033[0m" if colored else title)
    print("  ".join(value.ljust(widths[i]) for i, value in enumerate(headers)))
    for row, run in zip(rows, shown):
        line = "  ".join(value.ljust(widths[i]) for i, value in enumerate(row))
        color = (
            "32"
            if run["rank"] == 1
            else "31"
            if run["status"] == "failed"
            else "33"
            if run["status"] != "complete"
            else "0"
        )
        print(f"\033[{color}m{line}\033[0m" if colored else line)
    print("\n" + note)
    if args.details or args.run:
        for run in shown:
            print("\n" + details(run))

    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    markdown = [
        "# Ghostwriter run summary",
        "",
        title,
        "",
        note,
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    markdown.extend(
        "| " + " | ".join(cell(value) for value in row) + " |" for row in rows
    )
    for run in shown:
        markdown.extend(["", f"## {run['run']}", "", "```text", details(run), "```"])
    output = args.output or args.runs_dir / "summary.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"\nFull report with settings and copyable commands: {output.resolve()}")


if __name__ == "__main__":
    main()
