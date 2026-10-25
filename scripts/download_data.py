"""Download and restore the Loom datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request

REPO = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verified(path: Path, asset: dict) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == asset["bytes"]
        and sha256(path) == asset["sha256"]
    )


def destination(root: Path, name: str) -> Path:
    relative = Path(name)
    target = (root / relative).resolve()
    if relative.is_absolute() or not target.is_relative_to(root):
        raise ValueError(f"Invalid archive path: {name}")
    if relative.parts[0] not in ("data", "external"):
        raise ValueError(f"Unexpected data directory: {name}")
    return target


def extract_archive(archive: Path, root: Path) -> None:
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            target = destination(root, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"Expected a regular dataset file: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as incoming, target.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=8 * 1024 * 1024)


def main() -> None:
    manifest = json.loads((REPO / "datasets/manifest.json").read_text(encoding="utf-8"))
    packages = {item["key"]: item for item in manifest["packages"]}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", nargs="+", choices=list(packages), help="Default: all packages"
    )
    parser.add_argument(
        "--list", action="store_true", help="List packages without downloading"
    )
    parser.add_argument(
        "--root", type=Path, default=REPO, help="Repository-style destination root"
    )
    parser.add_argument("--cache", type=Path, default=REPO / ".dataset-downloads")
    parser.add_argument(
        "--from-local", type=Path, help="Read manually downloaded release assets here"
    )
    args = parser.parse_args()
    selected = args.only or list(packages)
    if args.list:
        for key in selected:
            item = packages[key]
            mib = sum(asset["bytes"] for asset in item["assets"]) / 2**20
            print(f'{key:18s} {mib:9.1f} MiB  {item["title"]}')
        return
    root = args.root.resolve()
    cache = (args.from_local or args.cache).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    for key in selected:
        item = packages[key]
        paths = []
        for asset in item["assets"]:
            path = cache / asset["name"]
            if not verified(path, asset):
                if args.from_local:
                    raise RuntimeError(f"Missing or incomplete release asset: {path}")
                print(f'Downloading {asset["name"]}', flush=True)
                url = (
                    f'https://github.com/{manifest["repository"]}/releases/download/'
                    f'{manifest["release_tag"]}/{asset["name"]}'
                )
                temporary = path.with_name(path.name + ".partial")
                with (
                    urllib.request.urlopen(url, timeout=120) as incoming,
                    temporary.open("wb") as outgoing,
                ):
                    shutil.copyfileobj(incoming, outgoing, length=8 * 1024 * 1024)
                temporary.replace(path)
                if not verified(path, asset):
                    raise RuntimeError(f"Download verification failed: {path}")
            paths.append(path)
        print(f"Restoring {key}", flush=True)
        if item["format"] == "tar.gz":
            extract_archive(paths[0], root)
        elif item["format"] == "split-file":
            target = destination(root, item["output"])
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".partial")
            with temporary.open("wb") as output:
                for path in paths:
                    with path.open("rb") as incoming:
                        shutil.copyfileobj(incoming, output, length=8 * 1024 * 1024)
            if (
                temporary.stat().st_size != item["bytes"]
                or sha256(temporary) != item["sha256"]
            ):
                raise RuntimeError(f"Reassembled file verification failed: {temporary}")
            temporary.replace(target)
        else:
            raise ValueError(f'Unsupported format: {item["format"]}')
        for entry in item["files"]:
            target = destination(root, entry["path"])
            if not target.is_file() or target.stat().st_size != entry["bytes"]:
                raise RuntimeError(
                    f'Restored dataset file is incomplete: {entry["path"]}'
                )
        print(f'  {len(item["files"])} files ready', flush=True)
    print(
        f"Datasets restored under {root}. See datasets/README.md for upstream loader setup."
    )


if __name__ == "__main__":
    main()
