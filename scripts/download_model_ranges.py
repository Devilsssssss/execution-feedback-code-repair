from __future__ import annotations

import argparse
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def remote_size(url: str) -> int:
    response = requests.get(
        url,
        headers={"Range": "bytes=0-0", "User-Agent": "repairbench/0.1"},
        timeout=(20, 60),
    )
    response.raise_for_status()
    content_range = response.headers.get("Content-Range", "")
    match = re.search(r"/([0-9]+)$", content_range)
    if not match:
        raise RuntimeError("server did not return a usable Content-Range header")
    return int(match.group(1))


def fetch_chunk(url: str, start: int, end: int, retries: int) -> tuple[int, bytes]:
    headers = {
        "Range": f"bytes={start}-{end}",
        "User-Agent": "repairbench/0.1",
    }
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=(20, 180))
            response.raise_for_status()
            payload = response.content
            expected = end - start + 1
            if len(payload) != expected:
                raise RuntimeError(f"short response: expected {expected}, got {len(payload)}")
            return start, payload
        except Exception:
            if attempt + 1 == retries:
                raise
    raise AssertionError("unreachable")


def sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="download a large ModelScope file by HTTP ranges")
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sha256")
    parser.add_argument("--chunk-mib", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    if args.chunk_mib <= 0 or args.workers <= 0 or args.retries <= 0:
        raise ValueError("chunk size, workers, and retries must be positive")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    total = remote_size(args.url)
    chunk_size = args.chunk_mib * 1024 * 1024
    ranges = [
        (start, min(start + chunk_size, total) - 1)
        for start in range(0, total, chunk_size)
    ]
    with temporary.open("wb") as handle:
        handle.truncate(total)

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_chunk, args.url, start, end, args.retries): (start, end)
            for start, end in ranges
        }
        for future in as_completed(futures):
            start, payload = future.result()
            with temporary.open("r+b") as handle:
                handle.seek(start)
                handle.write(payload)
            completed += 1
            print(f"downloaded {completed}/{len(ranges)} chunks", flush=True)

    actual = sha256(temporary)
    if args.sha256 and actual.lower() != args.sha256.lower():
        raise RuntimeError(f"sha256 mismatch: expected {args.sha256}, got {actual}")
    temporary.replace(output)
    print(f"wrote {output} ({total} bytes), sha256={actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
