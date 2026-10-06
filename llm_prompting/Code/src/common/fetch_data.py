"""Fetch dataset files that cannot be loaded through the datasets library.

SentMix-3L is distributed as a CSV in a GitHub repository rather than on the Hub,
so it is downloaded once and cached under data/. Re-running is a no-op.

Run:  python -m src.common.fetch_data
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.datasets import SENTMIX3L_CSV, SENTMIX3L_URL


def fetch_sentmix3l():
    if SENTMIX3L_CSV.exists():
        print(f"already present: {SENTMIX3L_CSV} ({SENTMIX3L_CSV.stat().st_size:,} bytes)")
        return
    import httpx

    SENTMIX3L_CSV.parent.mkdir(parents=True, exist_ok=True)
    response = httpx.get(SENTMIX3L_URL, timeout=120, follow_redirects=True)
    response.raise_for_status()
    SENTMIX3L_CSV.write_bytes(response.content)
    print(f"downloaded {len(response.content):,} bytes -> {SENTMIX3L_CSV}")


def main():
    print(f"source: {SENTMIX3L_URL}")
    fetch_sentmix3l()


if __name__ == "__main__":
    main()
