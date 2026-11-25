"""DocumentStore as one JSON file per document, readable only by the owner."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class JsonFileStore:
    def __init__(self, directory: Path) -> None:
        self._dir = directory.expanduser()
        self._dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    def load(self, name: str) -> dict[str, Any]:
        path = self._path(name)
        if not path.exists():
            return {}
        document = json.loads(path.read_text())
        if not isinstance(document, dict):
            raise ValueError(f"{path} does not hold a JSON object")
        return document

    def save(self, name: str, document: dict[str, Any]) -> None:
        path = self._path(name)
        # Write a sibling file, then rename over the old one: a crash leaves either version, never half of one.
        fd, temp = tempfile.mkstemp(dir=self._dir, prefix=f".{name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(document, handle, indent=2, sort_keys=True)
            os.chmod(temp, 0o600)
            os.replace(temp, path)
        except BaseException:
            Path(temp).unlink(missing_ok=True)
            raise

    def _path(self, name: str) -> Path:
        if not _NAME.match(name):
            raise ValueError(f"invalid document name: {name!r}")
        return self._dir / f"{name}.json"
