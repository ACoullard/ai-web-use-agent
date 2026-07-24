from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from evals.models import Fixture


def _resolve_url(raw_url: str, fixture_path: Path) -> str:
    if urlparse(raw_url).scheme:
        return raw_url
    return fixture_path.parent.joinpath(raw_url).resolve().as_uri()


def _fixture_dir(resolved_url: str) -> str:
    return resolved_url.rsplit("/", 1)[0]


def _substitute(value: Any, fixture_dir: str) -> Any:
    if isinstance(value, str):
        return value.replace("{fixture_dir}", fixture_dir)
    if isinstance(value, dict):
        return {key: _substitute(item, fixture_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute(item, fixture_dir) for item in value]
    return value


def load_fixture_file(path: Path) -> Fixture:
    """Load a single fixture YAML file.

    A relative `url` is resolved against this file's own directory (not cwd), so
    fixtures stay portable across checkouts. The literal token `{fixture_dir}` in
    any string leaf of `expected` is substituted with the resolved url's own parent
    directory URI - needed because the agent's file:// answers are checkout-path
    dependent and can't be hardcoded into the fixture. A fixture under a `live/`
    directory is marked live (`is_live=True`)
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    resolved_url = _resolve_url(raw["url"], path)
    fixture_dir = _fixture_dir(resolved_url)
    raw["url"] = resolved_url
    if raw.get("expected") is not None:
        raw["expected"] = _substitute(raw["expected"], fixture_dir)

    raw["is_live"] = "live" in path.parts

    return Fixture.model_validate(raw)


def load_fixtures(root: Path) -> list[Fixture]:
    """Recursively load every *.yaml/*.yml fixture under `root`, sorted by path."""
    paths = sorted({*root.rglob("*.yaml"), *root.rglob("*.yml")})

    fixtures: list[Fixture] = []
    seen: dict[str, Path] = {}
    for path in paths:
        fixture = load_fixture_file(path)
        if fixture.id in seen:
            raise ValueError(f"duplicate fixture id {fixture.id!r}: {seen[fixture.id]} and {path}")
        seen[fixture.id] = path
        fixtures.append(fixture)
    return fixtures


def load_fixture_path(path: Path) -> list[Fixture]:
    """Load one selection path: a single fixture file, or a directory to recurse into."""
    return [load_fixture_file(path)] if path.is_file() else load_fixtures(path)


def load_fixture_paths(paths: list[Path]) -> list[Fixture]:
    """Load and merge multiple selection paths (files and/or directories) - e.g. the
    positional PATHS args of `evals run`. Raises on a duplicate fixture id anywhere in
    the combined set, not just within one directory (each individual `load_fixtures`
    call below only catches duplicates within its own subtree).
    """
    fixtures: list[Fixture] = []
    seen: dict[str, Path] = {}
    for path in paths:
        for fixture in load_fixture_path(path):
            if fixture.id in seen:
                raise ValueError(f"duplicate fixture id {fixture.id!r} found under both {seen[fixture.id]} and {path}")
            seen[fixture.id] = path
            fixtures.append(fixture)
    return fixtures


def filter_fixtures(
    fixtures: list[Fixture],
    *,
    include_live: bool = False,
) -> list[Fixture]:
    """Filter a fixture list: live fixtures are excluded unless `include_live=True`."""
    return [f for f in fixtures if include_live or not f.is_live]
