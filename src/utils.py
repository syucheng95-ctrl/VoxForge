"""Pipeline utilities: config loading, path resolution, subprocess helpers.

Upload version: all paths resolved relative to the config file's directory (upload/).
No external ../pipeline or ../stage0_workspace references.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


# ── Where the external workspace source code lives (set by user if needed) ──
# These are the ONLY paths that reference outside upload/.
# If the workspaces are installed as packages or in PYTHONPATH, these can be empty.
WORKSPACE_PATHS = {
    "stage0": "",       # e.g. "../stage0_workspace/stage0_v2/router"
    "stage0_5": "",     # e.g. "../stage0.5_workspace"
    "stage1": "",       # e.g. "../生医工大赛-demo"
    "stage2": "",       # e.g. "../stage2_workspace"
}


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load YAML config, resolve all relative paths to absolute."""
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    upload_dir = config_path.parent.resolve()
    if not (upload_dir / "pipeline").is_dir() and (upload_dir.parent / "pipeline").is_dir():
        upload_dir = upload_dir.parent.resolve()
    config["_upload_dir"] = str(upload_dir)
    config["_config_path"] = str(config_path)

    # Resolve all ./ and ../ paths in config relative to upload_dir
    config = _resolve_string_paths(config, upload_dir)

    return config


def _resolve_string_paths(obj: Any, base_dir: Path) -> Any:
    """Recursively resolve string paths that start with ./ or ../ to absolute."""
    if isinstance(obj, dict):
        return {k: _resolve_string_paths(v, base_dir) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_string_paths(v, base_dir) for v in obj]
    if isinstance(obj, str) and (obj.startswith("./") or obj.startswith("../")):
        return str((base_dir / obj).resolve())
    return obj


def resolve(config: dict[str, Any], key_path: str) -> str:
    """Get a resolved value from config by dot-separated key path."""
    keys = key_path.split(".")
    val = config
    for k in keys:
        val = val[k]
    return str(val)


def run_step(cmd: list[str], step_name: str, env: dict[str, str] | None = None,
             cwd: str | None = None) -> None:
    """Run a subprocess step, print progress, fail on error."""
    print(f"\n{'=' * 60}")
    print(f"  {step_name}")
    print(f"{'=' * 60}")
    if cwd:
        print(f"  CWD: {cwd}")
    print(f"  CMD: {' '.join(cmd)}")
    print()

    t0 = time.time()
    result = subprocess.run(cmd, env=env, cwd=cwd)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  [FAILED] {step_name} (exit code {result.returncode}, {elapsed:.0f}s)")
        sys.exit(result.returncode)

    print(f"\n  [OK] {step_name} ({elapsed:.0f}s)")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]], append: bool = False) -> None:
    """Write a list of dicts to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write a dict to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
