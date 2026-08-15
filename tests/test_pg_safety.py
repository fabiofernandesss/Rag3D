"""Safety gates for the legacy PostgreSQL integration script."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_test_pg():
    path = Path(__file__).with_name("test_pg.py")
    spec = importlib.util.spec_from_file_location("rag3d_legacy_test_pg", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_pgvector_integration():
    path = Path(__file__).with_name("test_pgvector_integration.py")
    spec = importlib.util.spec_from_file_location("rag3d_pgvector_test_guard", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("environment", "safe"),
    [
        ({}, False),
        ({"RAG3D_PG": "postgresql:///production"}, False),
        ({"RAG3D_TEST_PG_DSN": "postgresql:///production"}, False),
        (
            {
                "RAG3D_TEST_PG_DSN": "postgresql:///production",
                "RAG3D_TEST_PG_ALLOW_DESTRUCTIVE": "1",
            },
            False,
        ),
        (
            {
                "RAG3D_TEST_PG_DSN": "postgresql:///production_contest",
                "RAG3D_TEST_PG_ALLOW_DESTRUCTIVE": "1",
            },
            False,
        ),
        (
            {
                "RAG3D_TEST_PG_DSN": "dbname=rag3d_test dbname=production",
                "RAG3D_TEST_PG_ALLOW_DESTRUCTIVE": "1",
            },
            False,
        ),
        (
            {
                "RAG3D_TEST_PG_DSN": "postgresql:///rag3d_v2_test",
                "RAG3D_TEST_PG_ALLOW_DESTRUCTIVE": "1",
            },
            True,
        ),
        (
            {
                "RAG3D_TEST_PG_DSN": "dbname=rag3d_contract_test host=/tmp",
                "RAG3D_TEST_PG_ALLOW_DESTRUCTIVE": "1",
            },
            True,
        ),
    ],
)
def test_legacy_pg_destructive_target_requires_three_independent_guards(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    safe: bool,
) -> None:
    for name in (
        "RAG3D_PG",
        "TRIRAG_PG",
        "RAG3D_TEST_PG_DSN",
        "RAG3D_TEST_PG_ALLOW_DESTRUCTIVE",
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    module = _load_test_pg()

    assert module._safe_test_target() is safe


def test_legacy_pg_make_rag_refuses_unsafe_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG3D_PG", "postgresql://admin:secret@host/production")
    monkeypatch.delenv("RAG3D_TEST_PG_DSN", raising=False)
    monkeypatch.delenv("RAG3D_TEST_PG_ALLOW_DESTRUCTIVE", raising=False)
    module = _load_test_pg()

    with pytest.raises(RuntimeError, match="test-only|refusing") as raised:
        module.make_rag()

    assert "secret" not in str(raised.value)


@pytest.mark.parametrize(
    ("dsn", "safe"),
    [
        ("postgresql:///rag3d_v2_test", True),
        ("dbname=rag3d_test host=/tmp", True),
        ("postgresql:///production", False),
        ("postgresql:///production_contest", False),
        ("dbname=rag3d_test dbname=production", False),
        ("postgresql:///rag3d_test?dbname=production", False),
    ],
)
def test_pgvector_destructive_guard_uses_the_effective_delimited_database_name(
    dsn: str, safe: bool
) -> None:
    module = _load_pgvector_integration()

    assert module._safe_test_database(dsn) is safe
