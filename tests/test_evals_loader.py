import pytest
from pydantic import ValidationError

from evals.loader import filter_fixtures, load_fixture_file, load_fixture_path, load_fixture_paths, load_fixtures
from evals.models import Fixture

_MINIMAL_SCHEMA = {"type": "object", "properties": {"pricing_url": {"type": "string"}}}


def _write_fixture(path, **overrides):
    body = {
        "id": "fixture-id",
        "task": "do the thing",
        "url": "html/home.html",
        "output_schema": _MINIMAL_SCHEMA,
        "grading": "exact_match",
        "expected": {"pricing_url": "{fixture_dir}/pricing.html"},
    }
    body.update(overrides)
    lines = []
    for key, value in body.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            _dump_yaml_block(lines, value, indent=2)
        elif isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {value!r}" if isinstance(value, str) else f"{key}: {value}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _dump_yaml_block(lines, mapping, indent):
    pad = " " * indent
    for key, value in mapping.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            _dump_yaml_block(lines, value, indent + 2)
        else:
            lines.append(f"{pad}{key}: {value!r}" if isinstance(value, str) else f"{pad}{key}: {value}")


def test_relative_url_resolves_against_fixture_directory(tmp_path):
    fixture_dir = tmp_path / "local"
    fixture_dir.mkdir()
    yaml_path = fixture_dir / "pricing-page-link.yaml"
    _write_fixture(yaml_path)

    fixture = load_fixture_file(yaml_path)

    expected_url = (fixture_dir / "html" / "home.html").resolve().as_uri()
    assert fixture.url == expected_url


def test_fixture_dir_substitution_in_expected(tmp_path):
    fixture_dir = tmp_path / "local"
    fixture_dir.mkdir()
    yaml_path = fixture_dir / "pricing-page-link.yaml"
    _write_fixture(yaml_path)

    fixture = load_fixture_file(yaml_path)

    html_dir_uri = (fixture_dir / "html").resolve().as_uri()
    assert fixture.expected == {"pricing_url": f"{html_dir_uri}/pricing.html"}


def test_absolute_url_left_unchanged(tmp_path):
    fixture_dir = tmp_path / "live"
    fixture_dir.mkdir()
    yaml_path = fixture_dir / "live-fixture.yaml"
    _write_fixture(yaml_path, url="https://example.com/page")

    fixture = load_fixture_file(yaml_path)

    assert fixture.url == "https://example.com/page"


def test_local_fixture_not_marked_live(tmp_path):
    fixture_dir = tmp_path / "local"
    fixture_dir.mkdir()
    yaml_path = fixture_dir / "fixture.yaml"
    _write_fixture(yaml_path)

    fixture = load_fixture_file(yaml_path)

    assert fixture.is_live is False


def test_live_directory_marks_fixture_live(tmp_path):
    fixture_dir = tmp_path / "live"
    fixture_dir.mkdir()
    yaml_path = fixture_dir / "fixture.yaml"
    _write_fixture(yaml_path, url="https://example.com/page")

    fixture = load_fixture_file(yaml_path)

    assert fixture.is_live is True


def test_exact_match_requires_expected_when_status_success():
    with pytest.raises(ValidationError, match="require"):
        Fixture(id="x", task="t", url="https://example.com", grading="exact_match")


def test_llm_judge_requires_rubric_when_status_success():
    with pytest.raises(ValidationError, match="rubric"):
        Fixture(id="x", task="t", url="https://example.com", grading="llm_judge")


def test_expected_and_rubric_not_required_for_non_success_expected_status():
    fixture = Fixture(
        id="x",
        task="t",
        url="https://example.com",
        grading="exact_match",
        expected_status="max_steps_exceeded",
    )
    assert fixture.expected is None


def test_load_fixtures_detects_duplicate_ids(tmp_path):
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    _write_fixture(local_dir / "a.yaml", id="dup-id")
    _write_fixture(local_dir / "b.yaml", id="dup-id")

    with pytest.raises(ValueError, match="duplicate"):
        load_fixtures(tmp_path)


def test_filter_fixtures_live_gate():
    live_fixture = Fixture(
        id="live-one",
        task="t",
        url="https://example.com",
        grading="exact_match",
        expected={"x": 1},
        is_live=True,
    )
    local_fixture = Fixture(
        id="local-one",
        task="t",
        url="https://example.com",
        grading="exact_match",
        expected={"x": 1},
    )
    fixtures = [live_fixture, local_fixture]

    # default (no filters) excludes live fixtures entirely.
    assert filter_fixtures(fixtures) == [local_fixture]

    # include_live=True admits them.
    assert filter_fixtures(fixtures, include_live=True) == [live_fixture, local_fixture]


def test_multi_page_fixture_colocated_in_one_folder(tmp_path):
    """A multi-page fixture's YAML and all its HTML pages can live together in their
    own folder - the fixture's url and {fixture_dir} substitution both resolve
    against that same folder, regardless of how deep it's nested under fixtures/.
    """
    fixture_folder = tmp_path / "local" / "multi-step-download"
    fixture_folder.mkdir(parents=True)
    (fixture_folder / "portal.html").write_text("<a href='next.html'>Next</a>", encoding="utf-8")
    (fixture_folder / "next.html").write_text("<a href='download.html'>Download</a>", encoding="utf-8")
    (fixture_folder / "download.html").write_text("the file", encoding="utf-8")
    _write_fixture(
        fixture_folder / "fixture.yaml",
        id="multi-step-download",
        url="portal.html",
        expected={"download_url": "{fixture_dir}/download.html"},
    )

    fixture = load_fixture_file(fixture_folder / "fixture.yaml")

    assert fixture.url == (fixture_folder / "portal.html").resolve().as_uri()
    assert fixture.expected == {"download_url": (fixture_folder / "download.html").resolve().as_uri()}
    assert fixture.is_live is False


def test_load_fixture_path_dispatches_file_vs_directory(tmp_path):
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    _write_fixture(local_dir / "a.yaml", id="fixture-a")
    _write_fixture(local_dir / "b.yaml", id="fixture-b")

    assert [f.id for f in load_fixture_path(local_dir / "a.yaml")] == ["fixture-a"]
    assert sorted(f.id for f in load_fixture_path(local_dir)) == ["fixture-a", "fixture-b"]


def test_load_fixture_paths_merges_multiple_selections(tmp_path):
    local_dir = tmp_path / "local"
    other_dir = tmp_path / "other"
    local_dir.mkdir()
    other_dir.mkdir()
    _write_fixture(local_dir / "a.yaml", id="fixture-a")
    _write_fixture(other_dir / "b.yaml", id="fixture-b")

    fixtures = load_fixture_paths([local_dir / "a.yaml", other_dir])

    assert sorted(f.id for f in fixtures) == ["fixture-a", "fixture-b"]


def test_load_fixture_paths_detects_duplicate_id_across_selections(tmp_path):
    local_dir = tmp_path / "local"
    other_dir = tmp_path / "other"
    local_dir.mkdir()
    other_dir.mkdir()
    _write_fixture(local_dir / "a.yaml", id="dup-id")
    _write_fixture(other_dir / "b.yaml", id="dup-id")

    with pytest.raises(ValueError, match="duplicate"):
        load_fixture_paths([local_dir, other_dir])
