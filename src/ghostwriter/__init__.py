"""Instructional next-token language modeling."""


def main() -> None:
    """Run the existing training pipeline from the installed console command."""
    from .pipeline import main as pipeline_main
    pipeline_main()
