# Typologien: Gruppen, Gesundheit, MCP-Navigation

**Stand:** 2026-10-18  
**Beispiel:** `ragkeep/assistants/philo-von-freisinn/sources/typologies/soziale-gesundheit.yaml`  
**Vorbild:** bisheriger Chunk-Typ `typology`. Die Texte stehen vorher in YAML unter `sources/typologies/`. ragprep erzeugt sie nicht.

Die YAML ist die Inhaltsquelle für dieses Beispiel. `docs/specification.md` im selben Projekt beschreibt eine ältere 12-Felder-Grafik und ist hier nicht maßgeblich.

---

## Ziel

Eine Typologie bleibt `chunk_type: typology`, wird aber ein kleiner Baum aus Chunks statt eines einzelnen Erklärungstexts.

| Ebene | Soziale Gesundheit | Weltanschauungen (eine Ebene weniger) |
|---|---|---|
| Übersicht | die ganze Matrix | „12 Weltanschauungen“ |
| Gruppe | ein Feld (`Geist -> Recht`, …) | — |
| Eintrag | ein Aspekt | eine Weltanschauung |

Dazu ein optionales Attribut `health`: `gesund` oder `krank`. Weltanschauungen und die anderen bestehenden JSONL-Typologien bekommen es nicht.

MCP kann den Baum direkt lesen (Geschwister, andere Felder, Gegenpol) und innerhalb einer Typologie semantisch suchen. `search_corpus` bleibt und findet die neuen Chunks weiter über den App-Typ `concept`.

---

## Was die YAML hergibt

Ohne `exclude` wird jedes Feld gechunkt. In dieser Datei steht `exclude: [7, 8, 9]`, also bleiben die Felder 1–6: je Pol 7 Aspekte, 84 Einträge. Jeder Aspekt hat `schlagwort`, `schlagsatz`, `kurztext`, `text`. Die Felder 7–9 stehen in der Datei und werden nicht gechunkt.

Maß für den Langtext ist das e5-Budget `MAX_EMBED_CHUNK_CHARS` = 1550. Gültig ist er von der Hälfte bis zum Budget, also ab 775 Zeichen bis 1550. Unter den 84 Aspekttexten liegen 71 im Fenster, 4 darunter (kürzester: Zutrauen, 470), 9 darüber (längster: Verführung, 3769). Der Import schreibt erst, wenn keiner mehr außerhalb liegt.

Gesund und krank sind keine zwei Typologien. Sie sind derselbe Baum, am Eintrag durch `health` unterschieden. Die Aspekte eines Feldes stehen sich der Reihenfolge nach gegenüber (Entfaltung ↔ Vorschrift, …). Dieselbe Position im Pol ist das Gegenstück. Der Import leitet daraus `counterpart_chunk_id` ab; in der YAML steht dafür kein eigenes Feld.

---

## Chunk-Modell

Weiterhin ein Chunk pro Knoten, Assistant-Partition (nicht `__shared__`), `chunk_type: typology`.

Neue Metadaten auf `ChunkMetadata` in ragrun und ragprep. Fehlende Felder heißen „trifft nicht zu“, nicht ein dritter Wert.

| Feld | Werte | Wer |
|---|---|---|
| `typology_slug` | `soziale-gesundheit`, `12-weltanschauungen` | jeder Knoten |
| `typology_role` | `overview`, `group`, `member` | jeder Knoten |
| `group_id` | `feld-1` … `feld-6` | Gruppe und ihre Einträge |
| `group_title` | Richtung, z. B. `Geist -> Recht` | Gruppe und ihre Einträge |
| `health` | `gesund`, `krank` | nur Einträge mit Pol |
| `counterpart_chunk_id` | Chunk des Gegenstücks im anderen Pol | nur Einträge mit Pol, deren Gegenstück existiert |
| `short_text` | Kurztext aus der YAML, nicht Teil des Chunk-Texts | jeder Knoten |
| `parent_id` | Chunk der nächsthöheren Ebene | Gruppe → Übersicht, Eintrag → Gruppe oder, ohne Gruppe, → Übersicht |

`segment_index` ist die Position des Aspekts in seinem Pol, ab 0, in der Reihenfolge der YAML. Der Gegenpol ist der Eintrag derselben `group_id` mit dem anderen `health` und demselben `segment_index`. Seine `chunk_id` steht als `counterpart_chunk_id` am Eintrag, auf beiden Seiten. Hat der andere Pol an dieser Position keinen Eintrag, bleibt `counterpart_chunk_id` leer.

`segment_title` ist der suchbare Name: Matrixtitel, Feld-Richtung oder Schlagwort / Weltanschauungsname. `aliases` tragen Schlagsatz, Feldtitel und die geläufigen Namensvarianten. `source_title` bleibt der Typologie-Name.

### IDs

Übersicht behält die bisherige ID, damit der bestehende Weltanschauungs-Chunk an Ort und Stelle weiterlebt:

`buildTypologyChunkId(assistant, name)` → `{assistant}:typology:{name}`

Neu, eigener Namespace-Anteil im Namensstring, nicht ein neuer UUID-Namespace:

- Gruppe: `{assistant}:typology:{slug}:group:{group_id}`
- Eintrag: `{assistant}:typology:{slug}:member:{key}` und, wenn `health` gesetzt ist, `:{health}`

`key` ist das normalisierte Schlagwort.

### source_id

Bisher teilen alle Typologien eines Assistenten eine Listen-`source_id` (`assistant:{name}:typology-list`). Ein Store-Lauf deprecated Orphans nur innerhalb einer `source_id`. Deshalb bekommt jede Typologie eine eigene:

`assistant:{name}:typology:{slug}`

Ein Lauf für Soziale Gesundheit fasst dann nicht die Weltanschauungen an. Die Übersicht der Weltanschauungen wird unter der neuen `source_id` mit derselben `chunk_id` upserted. Die alte Listen-`source_id` wird erst deprecated, wenn jede bisherige Typologie unter ihrer eigenen `source_id` liegt.

### Beispielbaum Soziale Gesundheit

- 1 Übersicht, `segment_title` = `Soziale Gesundheit`, ohne `health`
- 6 Gruppen, `segment_title` = Richtung, ohne `health` (die Gruppe umfasst beide Pole)
- 84 Aspekt-Einträge mit `health`

### Beispielbaum 12 Weltanschauungen

- 1 Übersicht (der heutige Erklärungschunk)
- 12 Einträge aus `list`, ohne `group_id`, ohne `health`, `parent_id` = Übersicht

Dieselbe Zweiteilung gilt für die übrigen Typologien aus `typologies-to-explain.jsonl` (Welträtsel, Triebfedern, Motive, Stimmungen): Übersicht plus ein Eintrag pro bisheriger `list`-Zeile, keine Gruppen, kein `health`. Jede wird eine eigene YAML, im selben Sinn wie `12-weltanschauungen.yaml`.

| Datei | Form |
|---|---|
| `soziale-gesundheit.yaml` | Übersicht, Gruppen, Einträge, `health` |
| `12-weltanschauungen.yaml` | Übersicht, zwölf Einträge |
| `7-weltraetsel-des-materialismus.yaml` | Übersicht, sieben Einträge |
| `4-stufen-der-sittlichen-triebfedern.yaml` | Übersicht, vier Einträge |
| `4-motive-sittlichen-handelns.yaml` | Übersicht, vier Einträge |
| `7-weltanschauungsstimmungen.yaml` | Übersicht, sieben Einträge |

`typologies-to-explain.jsonl` ist danach nicht mehr die Quelle. Die Übersichtstexte, die heute in `vector_chunks` liegen, werden der Langtext der Übersicht. Kurztexte und die Langtexte der Einträge (und bei Sozialer Gesundheit der Gruppen) werden außerhalb von ragprep in die YAML geschrieben, bevor irgendetwas gechunkt wird.

---

## Texte entstehen außerhalb von ragprep

Jeder Knoten, der ein Chunk wird, hat in der YAML einen Kurztext und einen Langtext: die Übersicht, jede Gruppe, jeder Eintrag. ragprep setzt diese Texte nicht zusammen, kürzt sie nicht und füllt keine Lücken. `rag:augment:typologies:explain` und der Endpoint `/graphs/typology-explain` werden entfernt.

## ragprep prüft nur

```
yarn rp rag:augment:typologies:from-yaml <assistant>
```

Liest jede YAML mit einem `typology:`-Block unter `paths.typologies` (`sources/typologies/`). Eine einzelne Datei geht über `--from-yaml <path>`.

Die Prüfung, sonst nichts:

- `typology` ist gesetzt. Übersicht, jede Gruppe und jeder Eintrag haben einen nicht leeren Kurztext und einen Langtext.
- Jeder Langtext, der ein Chunk wird (Übersicht, Gruppe, Eintrag), liegt im Längenfenster.
- `groups.exclude` nennt die Feldnummern, die nicht gechunkt werden. Jede andere Feldnummer ist eingeschlossen. Ein ausgeschlossenes Feld ohne Mitgliederliste ist in Ordnung. Ein eingeschlossenes Feld ohne Mitgliederliste ist ein Fehler. Eine Nummer in `exclude`, die in der Datei nicht vorkommt, ist ein Fehler.
- Fehlt das Frontmatter, bricht der Lauf ab, bevor er Längen prüft.

Besteht die Prüfung, werden die schon dastehenden Texte unverändert zu Chunks. Kurztext wird das Metadatum `short_text`, Langtext wird der Chunk-Text. Kein Kopf davor, kein `trim_to_embedding_budget`. Besteht sie nicht, sammelt der Lauf alle Verstöße und schreibt weder JSONL noch `store-chunks`. Jede Zeile nennt Typologie, Knoten (Feld, Pol, Name), ob Kurz oder Lang, Ist-Länge und die Grenze. Die YAML wird außerhalb von ragprep nachgezogen, danach der Lauf wiederholt.

### Frontmatter

Oben in der Inhalts-YAML. Es benennt die Schlüssel, es beschreibt die Texte nicht neu. Jeder Knoten hat `short` und `text`. Bei einer flachen Typologie fehlen `health` und `groups`; `overview.short` und `overview.text` zeigen auf die Felder der Übersicht, wie in `12-weltanschauungen.yaml`.

```yaml
typology:
  slug: soziale-gesundheit
  title: matrix_titel
  aliases: aliases
  basis: grundlage
  overview:
    short: uebersicht_kurz
    text: uebersicht
  health:
    field: pol
    values: [gesund, krank]
  groups:
    from: felder
    id: feld
    title: richtung
    label: titel
    short: kurz
    text: text
    exclude: [7, 8, 9]
  members:
    from: aspekte
    name: schlagwort
    lead: schlagsatz
    short: kurztext
    text: text
    quotes: zitate
```

| Schlüssel | Gilt im Chunk |
|---|---|
| `aliases` | Suchnamen der Typologie, Metadatum der Übersicht |
| `overview.text`, `groups.text`, `members.text` | Chunk-Text, wörtlich, das Längenfenster gilt hier |
| `overview.short`, `groups.short`, `members.short` | `short_text`, nicht Teil des Chunk-Texts; muss vorhanden sein |
| `members.name` | `segment_title` |
| `members.lead` | `aliases`, neben dem Feldtitel |
| `members.quotes` | `references`, nicht Teil des Chunk-Texts |
| `groups.title` | `group_title` |
| `groups.label` | Feldtitel je Pol, in der Gruppe und in `aliases` |
| `health.field` | `health` am Eintrag |
| `groups.exclude` | diese Feldnummern bleiben in der Datei und werden nicht gechunkt; alle übrigen sind eingeschlossen |

Fehlt `typology`, `aliases` oder `members.text`, bricht der Lauf ab, bevor er Längen prüft.

### Längenfenster

Gezählt wird der Langtext allein, nach Trim der YAML-Faltung. Budget `MAX_EMBED_CHUNK_CHARS` = 1550. Die Hälfte davon ist 775; weniger als das ist zu kurz.

| Länge | Folge |
|---|---|
| unter 775 Zeichen | Fehler `too_short` |
| 775–1550 | übernehmen |
| über 1550 | Fehler `too_long` |

Liegt ein Langtext außerhalb, ist das `too_short` oder `too_long`. Ein fehlender Kurztext ist `missing_short`. Geschrieben wird dann nichts.

### Eintrag (Aspekt)

Chunk-Text ist der Wert von `members.text`. Schlagwort, Schlagsatz, Feld und Pol stehen in den Metadaten, nicht vor dem Langtext.

### Gruppe und Übersicht

Auch sie haben Kurz- und Langtext in der YAML, geschrieben außerhalb von ragprep. Dieselben Fensterregeln. Bei den flachen Typologien gibt es keine Gruppe; die Übersicht ist der heutige Erklärungschunk, sobald er als Langtext im Fenster liegt, plus ein noch zu schreibender Kurztext.

Ausgabe nach bestandener Prüfung: `typologies/chunk-cache/<slug>.jsonl` und `store-chunks` unter der typologieeigenen `source_id`. Danach der bestehende Phase-2-Lauf `rag:embed` für `chunk_type=typology`. Die Übersichts-`chunk_id` der heutigen Typologien bleibt `buildTypologyChunkId(assistant, name)`.

---

## ragrun: Metadaten und Filter

`ChunkMetadata` nimmt die Felder aus der Tabelle oben auf. `health` nur `gesund` | `krank`. `typology_role` nur die drei Rollen. Unbekannte Werte lehnt `from_payload` ab, leere optionale Felder bleiben weg.

`search_corpus` und der Intent `erklaerung` ändern ihre Typ-Listen nicht. Neue Member-Chunks sind `chunk_type: typology` und fallen damit weiter unter den App-Typ `concept` (`begriff` + `typology`) und unter die Hybrid-Suche des Intents.

Der Lemma-Lookup `_lemma_lookup_multi_typology` trifft weiter über `segment_title` und `aliases`. Schlagwort und Weltanschauungsname liegen dort, sobald die Query im Action-Manifest wieder aktiv ist. Das ist nicht Teil dieses Plans; MCP ist die abgefragte Fläche.

---

## MCP

Zwei neue Tools. `search_corpus` bleibt der unscharfe Einstieg über `concept` und kennt die Struktur nicht. „Die anderen elf Weltanschauungen“ und „das kranke Gegenstück“ sind Nachschlagen, keine Ähnlichkeitssuche.

### `list_typology`

Argumente: `name` oder `chunk_id` (eines reicht).

Antwort:

- `typology_slug`, `title`, `has_health`
- `overview`: `chunk_id`, `segment_title`
- `groups`: `group_id`, `group_title`, `chunk_id` (leer bei Weltanschauungen)
- `members`: `segment_title`, `chunk_id`, `group_id`, `health`, `segment_index`, `counterpart_chunk_id`

Wird ein Eintrag per `chunk_id` geöffnet, zusätzlich `focus`:

- dieselbe Gruppe, die anderen Einträge (bei Weltanschauungen: alle anderen Mitglieder)
- die anderen Gruppen
- der Gegenpol über `counterpart_chunk_id`, wenn gesetzt

Gelesen wird aus `rag_chunks` (Assistant-Partition, `chunk_type = typology`, aktiv), nicht aus Qdrant. Die Liste stimmt damit schon nach `store-chunks`, vor dem Embed.

Unbekannter Name: `unknown_typology`.

### `search_typology`

Argumente:

- `query` (Pflicht)
- `typology` (Slug oder Titel, optional; fehlt er, läuft die Suche über alle Typologie-Chunks)
- `group` (`feld-1` oder Richtung)
- `health` (`gesund` | `krank`)
- `role` (`overview` | `group` | `member`)
- `limit` (1–20, Default 8)

Hybrid-Suche mit `chunk_type = typology`, danach derselbe Nachfilter wie heute `filter_snippets_by_typology_source`, erweitert um Slug, Rolle, Gruppe, `health`. Treffer tragen dieselben Strukturfelder wie `list_typology`, plus `score` und `snippet`.

`health` an einer Typologie ohne Pol (Weltanschauungen): `health_not_applicable`. Unbekannte Typologie: `unknown_typology`.

`search_corpus` gibt die neuen Felder mit aus, wenn sie am Chunk stehen (`typology_role`, `group_title`, `health`, `counterpart_chunk_id`). Der Filter `types` wird nicht um `typology` erweitert.

---

## Umsetzung

Die Abschnitte oben sind der Vertrag. Jede Stufe baut darauf und setzt die vorige voraus. Texte werden nur in Stufe 1 geschrieben, und nur außerhalb von ragprep.

### Stufe 1 — YAML

Sechs Dateien unter `sources/typologies/`, Frontmatter wie oben. `12-weltanschauungen.yaml` und `soziale-gesundheit.yaml` liegen schon, die anderen vier kommen aus `typologies-to-explain.jsonl`. Übersichtstexte aus `vector_chunks` werden der Langtext der Übersicht. Kurztexte und die übrigen Langtexte werden in die Dateien geschrieben, nicht erzeugt.

Fertig, wenn jede Datei `typology`, `aliases`, Kurz- und Langtext an Übersicht, Gruppe und Eintrag trägt, und `soziale-gesundheit.yaml` `exclude: [7, 8, 9]` hat. Die Längen müssen noch nicht stimmen. `typologies-to-explain.jsonl` wird nicht mehr gelesen.

### Stufe 2 — Chunk-Modell

`ChunkMetadata` in ragrun und ragprep nimmt `typology_slug`, `typology_role`, `group_id`, `group_title`, `health`, `counterpart_chunk_id`, `short_text` auf. IDs und `source_id` wie oben. Store lehnt unbekannte Werte für Rolle und `health` ab. Noch keine neuen Chunks.

Fertig, wenn ein Chunk mit diesen Feldern angenommen wird und einer ohne `health` ebenfalls.

### Stufe 3 — Erklärungsweg entfernen

`rag:augment:typologies:explain` und `/graphs/typology-explain` werden gelöscht, samt Prompt, Chain und Service. Nichts ruft sie mehr auf.

### Stufe 4 — Prüfung und Chunks

`rag:augment:typologies:from-yaml` liest die YAML-Dateien und prüft nur. Bei Verstoß Fehlerliste, keine Ausgabe. Für Soziale Gesundheit ist der erste Lauf die Liste: 4 Langtexte unter 775 Zeichen, 9 über 1550, dazu fehlende Kurz- und Langtexte von Übersicht und Gruppen. Die Dateien werden außerhalb von ragprep nachgezogen, der Lauf wiederholt.

Besteht jede Datei, schreibt das Kommando die Chunks unverändert, `store-chunks` unter der `source_id` der Typologie, danach `rag:embed` für `chunk_type=typology`. Die alte Listen-`source_id` wird deprecated, sobald alle sechs unter der eigenen liegen.

Fertig, wenn die Übersicht der zwölf Weltanschauungen ihre bisherige `chunk_id` behält, die Aspekte derselben Position auf beiden Polen denselben `segment_index` haben und über `counterpart_chunk_id` wechselseitig aufeinander zeigen, und die Felder 7–9 keine Chunks sind.

### Stufe 5 — MCP

`list_typology` und `search_typology` wie oben. `search_corpus` bleibt, gibt die neuen Felder mit aus und bekommt `typology` nicht als Filter. Testplan `tests/mcp/TESTPLAN_MCP.md`: Geschwister einer Weltanschauung, Aspekte eines Feldes, Gegenpol über `counterpart_chunk_id`, `health` an einer Weltanschauung weist ab.

---

## Nicht in diesem Schritt

- In ragprep Texte erzeugen, kürzen, auffüllen oder aus einer Übersicht auf Einträge verteilen
- `health` an Begriffe, Zitate oder Buch-Chunks
- Die auskommentierte Lemma-Query im Action-Manifest
- Die YAML wieder aus dem Projektordner lesen; sie liegt unter `sources/typologies/`
