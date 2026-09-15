"""What a repository declares about itself: `.mo-eval/config.toml`, read by the runner.

Three lines make a repository mineable, and the fourth is the security-relevant one:

    language = "python"
    setup_command = "poetry install --no-interaction"
    test_command = ".venv/bin/python -m pytest -q"
    forward_env = ["MOMENTO_API_KEY"]

`forward_env` names the environment variables the repository's tests need. Names only. The values
come from the runner's own environment — the CI job's secrets — and are never written here, never
put on a command line, and never sent to the service, which has no field that could carry them.

TOML rather than YAML because the runner is meant to be thin: `tomllib` is in the standard library
and a YAML parser is not, and a thin runner with a dependency is a runner that needs an install step
before it can run one.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from languages import LANGUAGES

CONFIG_PATH = Path(".mo-eval") / "config.toml"

_KNOWN_KEYS = {"language", "setup_command", "test_command", "forward_env", "env", "repo", "worker_image"}


class ConfigError(ValueError):
    """The declaration is missing, malformed, or names something the runner cannot honour."""


@dataclass(frozen=True)
class RunnerConfig:
    """A repository's declaration, validated."""

    language: str
    """A contract name from the language table."""
    test_command: str
    """The allow-list: probes append arguments to this and can never replace it."""
    setup_command: str | None = None
    """What makes a fresh export testable, or `None` for a repository that builds from source."""
    forward_env: tuple[str, ...] = ()
    """Names of environment variables to pass from the runner's environment into setup and probes."""
    env: dict[str, str] = field(default_factory=dict)
    """Static, non-secret environment the tests need (`POETRY_VIRTUALENVS_IN_PROJECT = "true"`)."""
    repo: str | None = None
    """`owner/name` recorded in tasks; derived from the git remote when absent."""
    worker_image: str | None = None
    """An OCI image pinned by digest that can build and test this repository — what a containerized
    agent run executes in. Optional: without it, tasks are emitted for host-native t-suite only."""


def load_config(path: Path) -> RunnerConfig:
    """Read and validate a declaration.

    Raises:
        ConfigError: If the file is absent, is not TOML, omits a required key, names an unknown key
            (a typo like `test_comand` would otherwise silently leave the allow-list empty), names
            a language the table does not know, or gives a value the wrong shape.
    """
    if not path.is_file():
        raise ConfigError(f"no declaration at {path}; a repository needs one to be mined")
    try:
        raw = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as failure:
        raise ConfigError(f"{path} is not valid TOML: {failure}") from failure

    unknown = set(raw) - _KNOWN_KEYS
    if unknown:
        raise ConfigError(f"{path} names unknown key(s) {sorted(unknown)}; known: {sorted(_KNOWN_KEYS)}")
    for required in ("language", "test_command"):
        if not isinstance(raw.get(required), str) or not raw[required].strip():
            raise ConfigError(f"{path} must set `{required}` to a non-empty string")
    if raw["language"] not in LANGUAGES:
        raise ConfigError(f"{path}: language {raw['language']!r} is not one of {sorted(LANGUAGES)}")

    forward = raw.get("forward_env", [])
    if not isinstance(forward, list) or not all(isinstance(name, str) and name for name in forward):
        raise ConfigError(f"{path}: `forward_env` must be a list of environment variable names")
    env = raw.get("env", {})
    if not isinstance(env, dict) or not all(isinstance(v, str) for v in env.values()):
        raise ConfigError(f"{path}: `env` must be a table of string values")
    image = raw.get("worker_image")
    if image is not None and not re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", str(image)):
        raise ConfigError(f"{path}: `worker_image` must be pinned by digest (name@sha256:<64 hex>), got {image!r}")
    setup = raw.get("setup_command")
    if setup is not None and (not isinstance(setup, str) or not setup.strip()):
        raise ConfigError(f"{path}: `setup_command`, when set, must be a non-empty string")

    return RunnerConfig(
        language=raw["language"],
        test_command=raw["test_command"].strip(),
        setup_command=setup.strip() if isinstance(setup, str) else None,
        forward_env=tuple(forward),
        env={str(key): value for key, value in env.items()},
        repo=raw.get("repo"),
        worker_image=image,
    )


def probe_environment(config: RunnerConfig, environ: Mapping[str, str]) -> tuple[dict[str, str], list[str]]:
    """Assemble the environment setup and probes run in, from the declaration and the runner's own.

    Returns:
        The environment to add, and the names in `forward_env` that the runner's environment did not
        have. A missing secret is reported, not ignored: a repository whose conftest reads it at
        import time fails every probe identically, which would otherwise read as a repository with
        no valid tasks.
    """
    forwarded = {name: environ[name] for name in config.forward_env if name in environ}
    missing = [name for name in config.forward_env if name not in environ]
    return {**config.env, **forwarded}, missing
