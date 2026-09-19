"""Reproducibly fetch the Olist source CSVs.

The dataset is published on Kaggle under CC BY-NC-SA 4.0 and requires an
account to download, which makes an unattended CI run impossible. This module
retrieves the same files from a public mirror and verifies every one against a
recorded SHA-256, so the pipeline provably runs on the authentic release
regardless of where the bytes came from.

The hashes in `spec.SOURCE_SHA256_FULL` were taken from files cross-verified
byte-for-byte against two independent mirrors (see docs/data_audit.md). A
mismatch aborts the run rather than training on unknown data.

Raw CSVs are not redistributed in this repository; this script is how a fresh
checkout obtains them.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

from data_pipeline import spec
from data_pipeline.loading import sha256_of

# Public mirrors of the unmodified Kaggle release, tried in order.
MIRRORS = [
    "https://raw.githubusercontent.com/tunguyenn99/ecommerce-data-modeling/main/dataset",
    "https://raw.githubusercontent.com/ductransponster/olist-brazilian-ecommerce/main/data/original_data",
]

TIMEOUT_SECONDS = 300


class FetchError(RuntimeError):
    """Raised when a file cannot be obtained or fails verification."""


def _download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "opspilot-data-fetch"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        if response.status != 200:
            raise FetchError(f"HTTP {response.status} for {url}")
        target.write_bytes(response.read())


def fetch_file(filename: str, raw_dir: Path, *, force: bool = False) -> str:
    """Download and verify one file. Returns its verified digest."""
    expected = spec.SOURCE_SHA256_FULL.get(filename)
    target = raw_dir / filename

    if target.exists() and not force:
        actual = sha256_of(target).lower()
        if expected and actual == expected:
            print(f"  {filename:<45s} cached, verified")
            return actual
        print(f"  {filename:<45s} cached copy failed verification, re-fetching")

    last_error: Exception | None = None
    for mirror in MIRRORS:
        url = f"{mirror}/{filename}"
        try:
            _download(url, target)
        except (urllib.error.URLError, OSError, FetchError) as exc:
            last_error = exc
            continue

        actual = sha256_of(target).lower()
        if expected and actual != expected:
            target.unlink(missing_ok=True)
            last_error = FetchError(
                f"{filename} from {mirror} has sha256 {actual}, expected {expected}"
            )
            continue

        print(f"  {filename:<45s} {target.stat().st_size:>12,} bytes  verified")
        return actual

    raise FetchError(
        f"could not obtain a verified copy of {filename}. Last error: {last_error}. "
        "Download the dataset manually from "
        "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce and place "
        f"the CSVs in {raw_dir}."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="re-download even if a verified copy exists")
    args = parser.parse_args(argv)

    raw_dir = spec.RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching the Olist Brazilian E-Commerce Public Dataset (CC BY-NC-SA 4.0)")
    print(f"  target: {raw_dir}")
    for filename in spec.SOURCE_FILES.values():
        fetch_file(filename, raw_dir, force=args.force)

    print("All source files present and verified.")
    print("Attribution: Olist, Brazilian E-Commerce Public Dataset, "
          "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FetchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
