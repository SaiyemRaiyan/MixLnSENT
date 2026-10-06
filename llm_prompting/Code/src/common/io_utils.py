import json
from pathlib import Path


def batch_sentences(data, batch_size, column="Sentence"):
    for start in range(0, len(data), batch_size):
        stop = min(start + batch_size, len(data))
        yield [row[column] for row in data.select(range(start, stop))]


def append_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_records_by_index(records, path, key="index"):
    """Resume helper: return the set of keys already stored in a JSONL file."""
    done = {record[key] for record in load_jsonl(path)}
    return done, records
