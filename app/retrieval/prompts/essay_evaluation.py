"""Prompt builders for essay evaluation (pitch criteria)."""
from __future__ import annotations

from typing import Mapping

from app.retrieval.prompts.essay_completion import load_essay_system_prompt


def build_evaluation_prompt(
    *,
    assistant: str,
    essay_title: str,
    mood_index: int | None,
    mood_name: str | None,
    text: str,
) -> list[Mapping[str, str]]:
    system = load_essay_system_prompt(assistant)
    segment = (
        f"Abschnitt {mood_index} ({mood_name})" if mood_index and mood_name else "Gesamttext"
    )
    user = f"""
Du bist ein strenger Essay-Editor. Bewerte den folgenden Text anhand der Kriterien:

Dichter (Suchender)
- zusammenhaengende, langlebige Argumentation
- tiefere Zusammenhaenge aufdecken, innere Ordnung zeigen
- Ruhe und Distanz zum Tageslaerm wahren

Director (Anbieter)
- Leser gewinnen, Aufmerksamkeit halten
- kurze, abwechslungsreiche Abschnitte, Beispiele, Anekdoten
- klare, nutzbare Botschaft, die sofort wirkt

Lustige Person (Vermittler)
- lebendige Sprache, Alltagserfahrung, leichte Ironie
- Mischung aus Beobachtung, Irrtum, Ueberraschung
- Leser spueren sich selbst, finden eigenes Echo

Struktur (7 Abschnitte, jeder ~300 Token)
1) Fragestellung / Raetsel (Neugier wecken)
2) Wesentliche Erkenntnisse / Richtung / Hauptpunkte
3) Beispiele / Anekdoten / Analogien (erlebbar machen)
4) Zentraler Gedanke in voller Klarheit (Tiefe & Bedeutung)
5) Handhabung im Leben: praktische erste Schritte
6) Weltzusammenhang: oeffentliche Auswirkung / wo sichtbar
7) Rueckschau: Details, die die Essenz der Schritte 1–6 ausmachen

Bewerte: {segment}
Essay-Titel: {essay_title}

Text:
{text}

Gib ausschliesslich JSON zurueck, mit genau diesem Schema:
{{
  "overall_score": 0-10,
  "criteria_scores": {{
    "Dichter": 0-10,
    "Director": 0-10,
    "LustigePerson": 0-10,
    "Struktur": 0-10
  }},
  "issues": ["..."],
  "instruction": "Knappe, umsetzbare Anweisung fuer eine Ueberarbeitung."
}}
""".strip()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
