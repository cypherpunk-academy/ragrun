# Intent-Modell und Abgrenzung (Philo von Freisinn)

> Dieses Dokument beschreibt das Intent-Design fuer den Assistenten-Chat.
> Implementierung: siehe `ASSITANTS_CHAT_PLAN.md`.

---

## 1. Kontext: Philos augmentierte Daten

Der Nutzer schreibt an die fiktive Person "Philo von Freisinn". Philo hat verschiedene
augmentierte Daten, die direkt abgefragt werden koennen:

| Augmentations-Typ | Chunk-Typen in Qdrant | Beispiel-Anfragen |
|-------------------|------------------------|-------------------|
| **Zusammenfassungen** | `chapter_summary`, `talk` (Vortraege) | "In welchem Vortrag/Buch hat Steiner ueber Beduerfnisse gesprochen?" |
| **Zitate** | `quote` | "Hat Steiner gesagt: '<text>'?", "Ich brauche ein Zitat zu Dreigliederung" |
| **Begriffe** | `begriff_list`, `explanation` | "Was ist das Rechtsleben?" |
| **Essays** | `essay` | Gedankenfehler-Essays |
| **Buch-/Vortragstexte** | `book`, `talk` | Volltext |

---

## 2. Feingranulare Intents (Zielbild)

| Intent | Bedeutung | Chunk-Typen | Beispiel |
|--------|-----------|-------------|----------|
| **werk_lokalisieren** | Wo findet sich ein Thema? (Buch, Vortrag, Reihe) | `chapter_summary`, `talk` | "In welchem Vortrag hat Steiner ueber Beduerfnisse gesprochen?" |
| **zitat_suchen** | Konkrete Zitate oder Belegstellen finden | `quote`, `book`, `talk` | "Ich brauche ein Zitat zu Dreigliederung" |
| **zitat_pruefen** | Pruefung eines konkreten Zitats (exakte Formulierung) | `quote` | "Hat Steiner gesagt: '<text>'?" |
| **begriff_definieren** | NUR Definition eines einzelnen Begriffs | `begriff_list`, `explanation` | "Was ist das Rechtsleben?" |
| **erklaerung** | Erklaerung, Vergleich, Vertiefung, Liste/Auflistung | `book`, `talk`, `chapter_summary`, `essay` | "Welches sind die 12 Weltanschauungen?" |
| **skip** | Kein Retrieval noetig | – | "Hallo", "Danke!" |
| **unklar** | Kein klares Intent erkennbar | alle Chunk-Typen, LLM entscheidet | vage, mehrdeutige Fragen |

---

## 3. Strenge Abgrenzungsregeln

### begriff_definieren
- **NUR** wenn der Nutzer explizit nach der *Bedeutung* oder *Definition* eines einzelnen Begriffs fragt.
- **NICHT** bei Auflistungen: "Welches sind die 12 Weltanschauungen?" = `erklaerung`.
- **NICHT** bei "Wo findet sich X?": das ist `werk_lokalisieren`.

### werk_lokalisieren
- **NUR** wenn explizit nach dem *Ort* (Buch, Vortrag, Vortragsreihe) gefragt wird.
- Typische Muster: "in welchem ...", "wo ...", "welches Werk ...".

### zitat_suchen vs. zitat_pruefen
- **zitat_pruefen**: Nutzer zitiert eine konkrete Formulierung und fragt, ob Steiner das gesagt hat.
- **zitat_suchen**: Nutzer moechte ein Zitat zu einem Thema finden (ohne exakte Formulierung).

### Negativbeispiele im Prompt
- "Welches sind die 12 Weltanschauungen?" → `erklaerung` (Liste), NICHT `begriff_definieren`.
- "Was bedeutet Weltanschauung?" → `begriff_definieren`.

### Confidence-Schwelle
- Wenn `confidence < 0.7` → Intent als `unklar` behandeln.

---

## 4. Unklarer Intent: LLM mit Qdrant-Chunks

Wenn kein klares Intent erkennbar ist (`unklar` oder confidence < Schwelle):

1. **Retrieval**: Alle Chunk-Typen, kein Filter.
2. **Query**: `user_message` (ggf. mit Kontext).
3. **Chunks**: An das LLM uebergeben.
4. **LLM**: Entscheidet aus den Chunks, was relevant ist, und antwortet.

---

## 5. Query-Vorbereitung fuer Qdrant

### Aktuell
- `query = state["user_message"]` – direkt, ohne Vorverarbeitung.

### Optionen

| Ansatz | Vorbereitung | Wann sinnvoll |
|--------|--------------|---------------|
| **Raw** | Direkt `user_message` | Kurze, klare Fragen |
| **Query-Expansion** | LLM erzeugt mehrere Varianten | Mehrdeutige oder vage Fragen |
| **HyDE** | LLM erzeugt hypothetische Antwort, diese wird als Query verwendet | Semantische Suche |
| **Intent-spezifisch** | Fuer `begriff_definieren`: nur Lemma; fuer `zitat_pruefen`: nur Zitattext | Wenn klar ist, welcher Teil der Frage wichtig ist |

### Empfehlung
- Bei **klarem Intent**: `user_message` unveraendert; fuer `begriff_definieren` nur Lemma (falls Lemma-Lookup verwendet wird).
- Bei **unklar**: `user_message` + optional `conversation_context` (letzte 1–2 Turns) fuer bessere Embedding-Query.

---

## 6. Sparse vs. Dense vs. Hybrid

| Methode | Staerken | Schwaechen | Einsatz |
|---------|----------|------------|----------|
| **Sparse (BM25)** | Exakte Terme, Fachbegriffe, Zitate, "Hat Steiner gesagt: '...'?" | Synonyme, Semantik | `zitat_pruefen`, `zitat_suchen` |
| **Dense** | Semantik, Paraphrasen, Synonyme | Exakte Terme, seltene Begriffe | `erklaerung`, `begriff_definieren` |
| **Hybrid** | Kombination beider | Etwas hoeherer Aufwand | Standard fuer alle Intents |

### Empfehlung
- **Hybrid** als Standard (wie in `retrievers.py`).
- **Zitat-Pruefung**: Sparse staerker gewichten (exakte Formulierung wichtig).
- **Begriff-Definition**: Dense staerker.
- **Werk-Lokalisierung**: Hybrid (Titel + Thema relevant).

### Parallel-Retrieval bei Zitaten
Bei Zitat-Intents (`zitat_suchen`, `zitat_pruefen`) zwei Qdrant-Suchen **parallel** durchfuehren und Ergebnisse fusionieren:

1. **Quote-Suche**: `chunk_type = "quote"`
2. **Buch-Suche**: `chunk_type IN ("book", "secondary_book")` mit `author = "Rudolf Steiner"`

Beide Suchen mit derselben Query; Ergebnisse per RRF oder Score-Fusion zusammenfuehren. So werden sowohl vorab extrahierte Zitate als auch relevante Stellen im Volltext gefunden.

---

## 7. Query-Laenge (Tokens)

| Laenge | Typ | Vor-/Nachteile |
|--------|-----|----------------|
| **Kurz** (10–50 Tokens) | Kernfrage | Fokussiert, schnell; Kontext kann fehlen |
| **Mittel** (50–150 Tokens) | Frage + Kontext | Guter Kompromiss |
| **Lang** (150–300 Tokens) | Mehr Kontext | Mehr Rauschen, hoehere Kosten |
| **Sehr lang** (>300 Tokens) | Vollstaendiger Kontext | Embedding oft "verwaschen" |

### Empfehlung
- **Standard**: ca. 50–150 Tokens (z.B. `user_message` + optional 1–2 Saetze Kontext).
- **Zitat-Pruefung**: Nur den Zitattext (oft <50 Tokens).
- **Begriff**: Nur Lemma oder kurze Frage.
- **Werk-Lokalisierung**: Frage + Thema (z.B. "Beduerfnisse" + "Vortrag").
- **Unklar**: Frage + optional kurzer Kontext.

**Nicht** pauschal 300-Token-Queries fuer Embedding; eher kurz und praezise.

---

## 8. Zusammenfassung

1. **Intent-Modell**: Feingranular mit `werk_lokalisieren`, `zitat_suchen`, `zitat_pruefen`, `begriff_definieren`, `erklaerung`, `skip`, `unklar`.
2. **Unklar**: Bei Confidence < 0.7 → `unklar` → Retrieval ohne Filter, LLM mit Chunks.
3. **Query**: Bei klarem Intent meist `user_message`; bei `unklar` optional `user_message` + kurzer Kontext.
4. **Retrieval**: Hybrid als Standard; bei Zitat-Pruefung Sparse staerker; bei Begriff-Definition Dense staerker.
5. **Zitate parallel**: Bei `zitat_suchen`/`zitat_pruefen` zwei Suchen parallel: `quote` sowie `book`/`secondary_book` mit author "Rudolf Steiner"; Ergebnisse fusionieren.
6. **Query-Laenge**: 50–150 Tokens fuer Embedding; bei Zitat-Pruefung nur Zitattext; bei Begriff nur Lemma.
