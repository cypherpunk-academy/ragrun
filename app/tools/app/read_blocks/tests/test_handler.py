"""Tests für read_blocks + document_tree.py — Doppelmatrix-Fixture (ohne Tabellen).

Contract §4.2. Deckt Adressierung per `paragraph_id` und `heading_path`
ab, inkl. Disambiguierung doppelter `###`-Titel unter verschiedenen
`##`-Eltern (Contract §4.3 "Disambiguierung").
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from app.tools.app.read_blocks.handler import run
from app.tools.document_tree import find_by_heading_path, find_paragraph, parse_document_tree
from app.tools.types import ToolContext

FIXTURES_DIR = Path(__file__).parent / "cases"
DOCUMENT_CONTENT = (FIXTURES_DIR / "doppelmatrix_excerpt.md").read_text(encoding="utf-8")
CASES = yaml.safe_load((FIXTURES_DIR / "read_blocks.yaml").read_text(encoding="utf-8"))["cases"]


def _ctx() -> ToolContext:
    return ToolContext(user_id="user-1", linked_document_id="doc-1", document_content=DOCUMENT_CONTENT)


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_read_blocks_cases(case: dict) -> None:
    result = asyncio.run(run(_ctx(), {"addresses": case["addresses"]}))
    assert result.result_key == "document_blocks"
    assert result.payload["blocks"] == case["expected_blocks"]


def test_read_blocks_without_linked_document_returns_empty() -> None:
    ctx = ToolContext(user_id="user-1")
    result = asyncio.run(run(ctx, {"addresses": [{"paragraph_id": "ch1.p2"}]}))
    assert result.payload == {"blocks": []}


def test_parse_document_tree_structure() -> None:
    tree = parse_document_tree(DOCUMENT_CONTENT)
    assert tree.title == "Doppelmatrix — Testausschnitt (ohne Tabellen)"
    assert [s.heading for s in tree.sections] == [
        "## Kapitel 1: Die Grundidee",
        "## Kapitel 2: Die drei Glieder im Einzelnen",
    ]
    kapitel1 = tree.sections[0]
    assert [p.id for p in kapitel1.paragraphs] == ["ch1.p1"]
    assert [c.heading for c in kapitel1.children] == [
        "### Feld 1 — Geist → Recht",
        "### Feld 2 — Geist → Wirtschaft",
    ]
    # Kein Zähler-Reset an "###" (Parität mit documentTree.ts) — IDs bleiben
    # über das ganze Kapitel eindeutig.
    all_ids = [p.id for p in kapitel1.paragraphs]
    for child in kapitel1.children:
        all_ids.extend(p.id for p in child.paragraphs)
    assert len(all_ids) == len(set(all_ids))


def test_find_by_heading_path_disambiguates_duplicate_subheadings() -> None:
    tree = parse_document_tree(DOCUMENT_CONTENT)
    feld1_kapitel1 = find_by_heading_path(
        tree, ["## Kapitel 1: Die Grundidee", "### Feld 1 — Geist → Recht"]
    )
    feld1_kapitel2 = find_by_heading_path(
        tree, ["## Kapitel 2: Die drei Glieder im Einzelnen", "### Feld 1 — Geist → Recht"]
    )
    assert feld1_kapitel1 is not None
    assert feld1_kapitel2 is not None
    assert feld1_kapitel1.paragraphs[0].text != feld1_kapitel2.paragraphs[0].text


def test_find_paragraph_returns_none_for_unknown_id() -> None:
    tree = parse_document_tree(DOCUMENT_CONTENT)
    assert find_paragraph(tree, "ch9.p9") is None
