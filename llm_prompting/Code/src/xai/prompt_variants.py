"""Generate auditable leave-one-demonstration-out prompt copies."""

import re
from pathlib import Path

from .config import PROJECT_ROOT, XAI_PROMPTS_DIR, ensure_output_dirs

EXAMPLE_BLOCK = re.compile(
    r"(?ms)^Example\s+(\d+):\s*\n.*?(?=^Example\s+\d+:|^Return exactly one label|\Z)"
)


def make_leave_one_out_prompts(source: Path | None = None) -> list[Path]:
    source = source or (
        PROJECT_ROOT / "prompts" / "prompting" / "five_shot_minimal_v1.txt"
    )
    original = source.read_bytes().decode("utf-8")
    matches = list(EXAMPLE_BLOCK.finditer(original))
    if not matches:
        raise ValueError(f"no Example blocks found in {source}")
    ensure_output_dirs()
    XAI_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for match in matches:
        number = int(match.group(1))
        changed = original[:match.start()] + original[match.end():]
        path = XAI_PROMPTS_DIR / f"five_shot_minimal_leave_out_{number:02d}_v1.txt"
        path.write_bytes(changed.encode("utf-8"))
        paths.append(path)
    return paths
