"""Intent-Konstanten fuer den Assistenten-Chat.

Iteration 1: 5 vereinfachte Intents.
Iteration 3+: feingranulare Intents aus der "Gestrichene Intents"-Tabelle
im Implementierungsplan einführen.
"""
from __future__ import annotations

INTENT_LABELS: list[str] = [
    "begriff_definieren",   # Begriff nachschlagen (mit Lemma-Lookup)
    "quelle_suchen",        # Zitat oder Belegstelle finden
    "erklaerung",           # Erklaerung, Vergleich, Vertiefung, Zusammenfassung
    "skip",                 # Kein Retrieval nötig (Gruss, Meta-Frage, Off-Topic)
    "sonstiges",            # Fallback: breites Retrieval ohne chunk_type-Filter
]

INTENT_CHUNK_TYPE_MAP: dict[str, list[str]] = {
    "begriff_definieren": ["begriff", "explanation"],
    "quelle_suchen":      ["quote", "book", "talk"],
    "erklaerung":         ["book", "talk", "chapter_summary", "secondary_book"],
    "sonstiges":          [],   # leere Liste → kein chunk_type-Filter
}

SKIP_RETRIEVAL_INTENTS: frozenset[str] = frozenset({"skip"})

# ---------------------------------------------------------------------------
# Gestrichene Intents (Iteration 2+ – nur einfuehren wenn Chat-Logs den Bedarf zeigen)
# ---------------------------------------------------------------------------
#
# | Gestrichener Intent | Gemappt auf (Iter. 1) | Grund                          |
# |---------------------|-----------------------|--------------------------------|
# | werk_lokalisieren   | erklaerung            | chunk_types überlappen stark   |
# | zitat_suchen        | quelle_suchen         | Selber Retrieval-Pfad          |
# | vergleich           | erklaerung            | Breites Retrieval deckt ab     |
# | zusammenfassung     | erklaerung            | chapter_summary im Plan        |
# | beleg_pruefung      | quelle_suchen         | Selber Retrieval-Pfad          |
# | follow_up           | sonstiges             | Komplexe Logik erst ab Iter. 3 |
# | hypothetisch        | erklaerung            | Seltener Fall                  |
# | meta_assistent      | skip                  | Teil der Skip-Gruppe           |
# | konversationell     | skip                  | Dto.                           |
# | out_of_scope        | skip                  | Dto.                           |
