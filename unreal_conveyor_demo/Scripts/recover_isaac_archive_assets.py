#!/usr/bin/env python3
"""Recover named original files from NVIDIA's split Isaac Sim asset archive.

The 6.0.1 complete pack is about 80 GiB. This reader presents its five HTTP
parts as one seekable ZIP and uses byte-range requests, so release preparation
can extract a few server-retired dependencies without downloading the pack.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path


PART_URLS = [
    f"https://downloads.isaacsim.nvidia.com/isaac-sim-assets-complete-6.0.1.{part:03d}.zip"
    for part in range(1, 6)
]


class SplitHttpReader(io.BufferedIOBase):
    def __init__(self, urls: list[str]) -> None:
        self.urls = urls
        self.part_sizes = [self._content_length(url) for url in urls]
        self.boundaries = []
        total = 0
        for size in self.part_sizes:
            total += size
            self.boundaries.append(total)
        self.total_size = total
        self.position = 0

    @staticmethod
    def _content_length(url: str) -> int:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=30) as response:
            return int(response.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.total_size + offset
        else:
            raise ValueError(f"Unsupported seek mode: {whence}")
        if position < 0:
            raise ValueError("Negative seek position")
        self.position = min(position, self.total_size)
        return self.position

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.total_size - self.position
        remaining = min(size, self.total_size - self.position)
        chunks: list[bytes] = []
        while remaining:
            part_index = bisect.bisect_right(self.boundaries, self.position)
            part_start = 0 if part_index == 0 else self.boundaries[part_index - 1]
            offset = self.position - part_start
            count = min(remaining, self.part_sizes[part_index] - offset)
            end = offset + count - 1
            request = urllib.request.Request(
                self.urls[part_index],
                headers={"Range": f"bytes={offset}-{end}"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise OSError(
                        f"Server ignored range request for part {part_index + 1}: HTTP {response.status}"
                    )
                chunk = response.read()
            if len(chunk) != count:
                raise OSError(f"Short range response: expected {count}, received {len(chunk)}")
            chunks.append(chunk)
            self.position += count
            remaining -= count
        return b"".join(chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--name", action="append", dest="names")
    parser.add_argument(
        "--missing-list",
        type=Path,
        help="Newline-separated absolute dependency paths emitted by collect_omniverse_assets.py",
    )
    parser.add_argument(
        "--payload-root",
        type=Path,
        help="Payload root used with --missing-list; recovered files are written to their exact paths below it",
    )
    args = parser.parse_args()
    if args.missing_list:
        if not args.payload_root:
            parser.error("--payload-root is required with --missing-list")
    elif not args.output or not args.names:
        parser.error("either --missing-list/--payload-root or --output with one or more --name is required")
    return args


def choose_member(members: list[str], filename: str) -> str:
    candidates = [member for member in members if member.lower().endswith("/" + filename.lower())]
    if not candidates:
        raise FileNotFoundError(f"{filename} is not present in the official 6.0.1 asset pack")
    packing_table = [member for member in candidates if "/PackingTable/" in member]
    selected = packing_table[0] if packing_table else candidates[0]
    if len(packing_table or candidates) > 1:
        print(f"multiple archive members match {filename}; selecting {selected}", flush=True)
    return selected


def choose_dependency_member(members: list[str], relative_path: Path) -> str:
    """Find the same asset path in the official archive, preserving its destination hierarchy."""

    normalized = relative_path.as_posix()
    marker = "/Assets/"
    marker_index = normalized.find(marker)
    suffix = normalized[marker_index + 1 :] if marker_index >= 0 else normalized
    suffix_lower = suffix.lower()
    exact = [
        member
        for member in members
        if member.lower() == suffix_lower or member.lower().endswith("/" + suffix_lower)
    ]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        exact.sort(key=len)
        print(f"multiple exact archive paths match {suffix}; selecting {exact[0]}", flush=True)
        return exact[0]

    # Some 6.0 URLs were retired after the same source textures had been copied
    # into another official asset subtree. Prefer the candidate with the longest
    # identical path tail; this keeps, for example, PackingTable textures tied to
    # PackingTable rather than accepting an unrelated same-named bitmap.
    filename = relative_path.name
    candidates = [member for member in members if member.lower().endswith("/" + filename.lower())]
    if not candidates:
        raise FileNotFoundError(f"{suffix} is not present in the official 6.0.1 asset pack")
    wanted_parts = [part.lower() for part in Path(suffix).parts]

    def tail_score(member: str) -> tuple[int, int]:
        candidate_parts = [part.lower() for part in Path(member).parts]
        score = 0
        for wanted, candidate in zip(reversed(wanted_parts), reversed(candidate_parts)):
            if wanted != candidate:
                break
            score += 1
        return score, -len(member)

    candidates.sort(key=tail_score, reverse=True)
    selected = candidates[0]
    score = tail_score(selected)[0]
    if score < 2:
        raise FileNotFoundError(
            f"No path-safe archive match for {suffix}; filename-only matches are intentionally rejected"
        )
    print(f"retired URL recovery: {suffix} -> {selected} (matching tail parts={score})", flush=True)
    return selected


def dependency_targets(missing_list: Path, payload_root: Path) -> list[tuple[Path, Path]]:
    payload_root = payload_root.resolve()
    targets: list[tuple[Path, Path]] = []
    for line in missing_list.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        destination = Path(line.strip()).resolve()
        try:
            relative = destination.relative_to(payload_root)
        except ValueError as error:
            raise ValueError(f"Dependency path escapes payload root: {destination}") from error
        targets.append((destination, relative))
    if not targets:
        raise ValueError(f"Missing dependency list is empty: {missing_list}")
    return targets


def main() -> int:
    args = parse_args()
    if args.missing_list:
        targets = dependency_targets(args.missing_list, args.payload_root)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        targets = [(args.output / name, Path(name)) for name in args.names]
    reader = SplitHttpReader(PART_URLS)
    print(f"opened virtual {reader.total_size / (1024**3):.2f} GiB Isaac archive", flush=True)
    with zipfile.ZipFile(reader) as archive:
        members = archive.namelist()
        print(f"archive index contains {len(members)} members", flush=True)
        cached: dict[str, bytes] = {}
        for destination, relative in targets:
            member = (
                choose_dependency_member(members, relative)
                if args.missing_list
                else choose_member(members, relative.name)
            )
            if member not in cached:
                with archive.open(member) as source:
                    cached[member] = source.read()
            payload = cached[member]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            print(
                f"recovered {member} -> {destination} ({len(payload)} bytes, sha256={digest})",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
