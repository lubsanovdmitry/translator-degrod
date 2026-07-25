from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def resource_differences(repository_root: Path) -> list[str]:
    pairs = (
        (
            repository_root / "prompts",
            repository_root / "src/semantic_telephone/resources/prompts",
            "*.txt",
        ),
        (
            repository_root / "configs",
            repository_root / "src/semantic_telephone/resources/profiles",
            "*.yaml",
        ),
    )
    differences: list[str] = []
    for source, packaged, pattern in pairs:
        source_names = {path.name for path in source.glob(pattern)}
        packaged_names = {path.name for path in packaged.glob(pattern)}
        for name in sorted(source_names | packaged_names):
            source_path = source / name
            packaged_path = packaged / name
            if (
                not source_path.exists()
                or not packaged_path.exists()
                or source_path.read_bytes() != packaged_path.read_bytes()
            ):
                differences.append(name)
    return differences


def sync_resources(repository_root: Path) -> None:
    destinations = (
        (
            repository_root / "prompts",
            repository_root / "src/semantic_telephone/resources/prompts",
            "*.txt",
        ),
        (
            repository_root / "configs",
            repository_root / "src/semantic_telephone/resources/profiles",
            "*.yaml",
        ),
    )
    for source, packaged, pattern in destinations:
        packaged.mkdir(parents=True, exist_ok=True)
        expected = {path.name for path in source.glob(pattern)}
        for stale in packaged.glob(pattern):
            if stale.name not in expected:
                stale.unlink()
        for path in source.glob(pattern):
            shutil.copyfile(path, packaged / path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize packaged prompts and profiles.")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    if args.check:
        differences = resource_differences(root)
        if differences:
            raise SystemExit("packaged resources differ: " + ", ".join(differences))
        return
    sync_resources(root)


if __name__ == "__main__":
    main()
