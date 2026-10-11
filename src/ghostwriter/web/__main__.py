import argparse
from pathlib import Path

import uvicorn

from .app import create_app
from ..training.checkpoint import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(
        description="Serve the local Ghostwriter playground."
    )
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")
    print(f"Ghostwriter playground: http://127.0.0.1:{args.port}", flush=True)
    uvicorn.run(create_app(args.runs_dir), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
