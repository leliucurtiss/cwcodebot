from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, value = line.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[name] = value
    return values


def _expand_env(value: Any, environment: dict[str, str]) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            name = match.group(1)
            default = match.group(2)
            return environment.get(name, default or "")

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v, environment) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v, environment) for v in value]
    return value

@dataclass(frozen=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000

@dataclass(frozen=True)
class MaasConfig:
    base_url: str
    api_key: str
    default_model: str = "devstral"
    timeout_seconds: float = 300.0

@dataclass(frozen=True)
class ContextConfig:
    model_context_max: int = 10240
    safe_input_budget: int = 10240
    output_reserve: int = 16000
    token_char_ratio: float = 3.0
    token_safety_multiplier: float = 1.25
    max_file_bytes: int = 12374
    max_compressed_length: int = 10240  # Add this line to include the new attribute

@dataclass(frozen=True)
class WorkspaceConfig:
    root: Path

@dataclass(frozen=True)
class StorageConfig:
    sqlite_path: Path

@dataclass(frozen=True)
class AppConfig:
    server: ServerConfig
    maas: MaasConfig
    context: ContextConfig
    workspace: WorkspaceConfig
    storage: StorageConfig
    log_level: str = "INFO"


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path or os.getenv("HARNESS_CONFIG", "config.yaml"))
    environment = {**_load_dotenv(config_path.parent / ".env"), **os.environ}
    if not config_path.exists():
        example = Path("config.example.yaml")
        if example.exists():
            config_path = example
        else:
            raise FileNotFoundError("Missing config.yaml and config.example.yaml")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data = _expand_env(raw, environment)

    server = data.get("server", {})
    maas = data.get("maas", {})
    context = data.get("context", {})
    workspace = data.get("workspace", {})
    storage = data.get("storage", {})

    base_url = str(maas.get("base_url", "")).rstrip("/")
    if not base_url:
        raise ValueError("maas.base_url is required")
    api_key = str(maas.get("api_key", "")).strip()
    if len(api_key) >= 2 and api_key[0] == api_key[-1] and api_key[0] in {"'", '"'}:
        api_key = api_key[1:-1].strip()
    if api_key.lower().startswith("bearer "):
        api_key = api_key[7:].strip()

    return AppConfig(
        server=ServerConfig(
            host=str(server.get("host", "127.0.0.1")),
            port=int(server.get("port", 8000)),
        ),
        maas=MaasConfig(
            base_url=base_url,
            api_key=api_key,
            default_model=str(maas.get("default_model", "devstral")),
            timeout_seconds=float(maas.get("timeout_seconds", 300)),
        ),
        context=ContextConfig(
            model_context_max=int(context.get("model_context_max", 128000)),
            safe_input_budget=int(context.get("safe_input_budget", 128000)),
            output_reserve=int(context.get("output_reserve", 16000)),
            token_char_ratio=float(context.get("token_char_ratio", 3.0)),
            token_safety_multiplier=float(context.get("token_safety_multiplier", 1.25)),
            max_file_bytes=int(context.get("max_file_bytes", 64000)),
            max_compressed_length=int(context.get("max_compressed_length", 500000)),  # Add this line to include the new attribute
        ),
        workspace=WorkspaceConfig(
            root=Path(workspace.get("root", ".")).expanduser().resolve(),
        ),
        storage=StorageConfig(
            sqlite_path=Path(storage.get("sqlite_path", "./data/harness.db")).expanduser(),
        ),
        log_level=data.get("log_level", "INFO"),
    )
