from pathlib import Path

from local_ai_harness.config import load_config


def test_config_loads_ignored_dotenv_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        'OPENROUTER_API_KEY="test-key"\n',
        encoding="utf-8",
    )
    (tmp_path / "config.yaml").write_text(
        """
server:
  host: 127.0.0.1
  port: 8000
maas:
  base_url: https://openrouter.ai/api/v1
  api_key: ${OPENROUTER_API_KEY}
workspace:
  root: .
storage:
  sqlite_path: ./data/harness.db
""",
        encoding="utf-8",
    )

    config = load_config(tmp_path / "config.yaml")

    assert config.maas.api_key == "test-key"


def test_process_environment_takes_precedence_over_dotenv(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "process-key")
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=file-key\n", encoding="utf-8")
    (tmp_path / "config.yaml").write_text(
        """
maas:
  base_url: https://openrouter.ai/api/v1
  api_key: ${OPENROUTER_API_KEY}
workspace:
  root: .
storage:
  sqlite_path: ./data/harness.db
""",
        encoding="utf-8",
    )

    config = load_config(tmp_path / "config.yaml")

    assert config.maas.api_key == "process-key"
