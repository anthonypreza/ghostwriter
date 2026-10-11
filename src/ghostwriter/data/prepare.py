import json
import os
import re
from pathlib import Path

from sklearn.model_selection import train_test_split

SECTION_TYPES = [
    ("PRE_CHORUS", r"pre[\s-]?chorus"),
    ("POST_CHORUS", r"post[\s-]?chorus"),
    ("CHORUS", r"chorus"),
    ("VERSE", r"verse"),
    ("BRIDGE", r"bridge"),
    ("INTRO", r"intro"),
    ("OUTRO", r"outro"),
    ("REFRAIN", r"refrain"),
    ("INTERLUDE", r"interlude"),
    ("HOOK", r"hook"),
]


def normalize_section(line: str) -> str | None:
    line = line.strip()

    if not (line.startswith("[") and line.endswith("]")):
        return None

    content = line[1:-1].strip().lower()

    for section, pattern in SECTION_TYPES:
        if re.search(pattern, content):
            return f"<{section}>"

    return None


CURRENT_DIR = os.path.dirname(__file__)
RAW_DIR = Path(f"{CURRENT_DIR}/../../../data/raw")
PROCESSED_DIR = Path(f"{CURRENT_DIR}/../../../data/processed")


def normalize_lyrics(text: str, strip=True) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() if strip else text


def clean_lyrics(text: str, section_tokens=True) -> str:
    text = normalize_lyrics(text)

    cleaned_lines = []

    lines = text.splitlines()
    for index, line in enumerate(lines):
        line = line.strip()

        if not line:
            if not section_tokens or (cleaned_lines and cleaned_lines[-1] != ""):
                cleaned_lines.append("")
            continue

        section = normalize_section(line)
        if line in {f"<{name}>" for name, _ in SECTION_TYPES}:
            section = line

        if not section_tokens and (
            section or (line.startswith("[") and line.endswith("]"))
        ):
            # Removing a label must not join two stanzas. Keep existing blank lines.
            if (
                cleaned_lines
                and cleaned_lines[-1] != ""
                and index + 1 < len(lines)
                and lines[index + 1].strip()
            ):
                cleaned_lines.append("")
            continue

        if section:
            cleaned_lines.append(section)
        elif line.startswith("[") and line.endswith("]"):
            # Unknown annotation: skip for now
            continue
        else:
            cleaned_lines.append(line)

    cleaned = normalize_lyrics("\n".join(cleaned_lines))

    return f"<BOS>\n{cleaned}\n<EOS>"


def prepare_corpus(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR, section_tokens=True):
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    songs = sorted(raw_dir.glob("*.txt"))
    if len(songs) < 6:
        raise ValueError("At least 6 song files are needed for an 80/10/10 split.")

    train_songs, remaining = train_test_split(songs, test_size=0.2, random_state=42)
    val_songs, test_songs = train_test_split(remaining, test_size=0.5, random_state=42)
    splits = {"train": train_songs, "val": val_songs, "test": test_songs}
    # Read and clean every input before changing any previous output.
    cleaned_songs = {
        path.name: clean_lyrics(path.read_text(encoding="utf-8"), section_tokens)
        for path in songs
    }
    manifest = {
        "random_state": 42,
        "section_tokens": section_tokens,
        "max_consecutive_newlines": 2,
    }

    for split, paths in splits.items():
        split_dir = processed_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        filenames = {path.name for path in paths}
        # Remove stale generated files when the input corpus changes.
        for old_path in split_dir.glob("*.txt"):
            if old_path.name not in filenames:
                old_path.unlink()
        manifest[split] = []
        for path in sorted(paths):
            output_path = split_dir / path.name
            output_path.write_text(cleaned_songs[path.name], encoding="utf-8")
            manifest[split].append(output_path.relative_to(processed_dir).as_posix())

    (processed_dir / "splits.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean lyrics and split by song.")
    parser.add_argument(
        "--section-tokens", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    prepare_corpus(section_tokens=args.section_tokens)
