# Typologien: Gruppen, Gesundheit, MCP-Navigation

**Stand:** 2026-10-19  
**Beispiel:** `ragkeep/assistants/philo-von-freisinn/sources/typologies/soziale-gesundheit.yaml`  
**Vorbild:** bisheriger Chunk-Typ `typology`. Die Texte stehen vorher in YAML unter `sources/typologies/`. ragprep erzeugt sie nicht.

Die YAML ist die Inhaltsquelle für dieses Beispiel. `docs/specification.md` im selben Projekt beschreibt eine ältere 12-Felder-Grafik und ist hier nicht maßgeblich.

---

## Ziel

Eine Typologie bleibt `chunk_type: typology`, wird aber ein kleiner Baum aus Chunks statt eines einzelnen Erklärungstexts.

| Ebene | Soziale Gesundheit | Weltanschauungen (eine Ebene weniger) |
|---|---|---|
| Übersicht | die ganze Matrix | „12 Weltanschauungen“ |
| Gruppe | ein Feld je Pol (`Geist -> Recht`, gesund bzw. krank) | — |
| Eintrag | ein Aspekt | eine Weltanschauung |

Dazu ein optionales Attribut `health`: `gesund` oder `krank`. Weltanschauungen und die anderen bestehenden JSONL-Typologien bekommen es nicht.

MCP kann den Baum direkt lesen (Geschwister, andere Felder, Gegenpol) und innerhalb einer Typologie semantisch suchen. `search_corpus` bleibt und findet die neuen Chunks weiter über den App-Typ `concept`.

---

## Was die YAML hergibt

Ohne `exclude` wird jedes Feld gechunkt. In dieser Datei steht `exclude: [7, 8, 9]`, also bleiben die Felder 1–6: je Pol 7 Aspekte, 84 Einträge. Jeder Aspekt hat `schlagwort`, `schlagsatz`, `kurztext`, `text`. Die Felder 7–9 stehen in der Datei und werden nicht gechunkt.

Maß für den Langtext ist das e5-Budget `MAX_EMBED_CHUNK_CHARS` = 1550. Gültig ist er von der Hälfte bis zum Budget, also ab 775 Zeichen bis 1550. Alle 84 Aspekttexte liegen im Fenster (Stand 19.10.2026). Der Import schreibt erst, wenn kein Langtext mehr außerhalb liegt.

Gesund und krank sind keine zwei Typologien. Sie sind derselbe Baum, an Gruppe und Eintrag durch `health` unterschieden. Jedes Feld hat in der YAML einen Eintrag je Pol mit eigenem `titel`, `kurz` und `text`; daraus werden zwei Gruppen-Chunks mit derselben `group_id`. Die Aspekte eines Feldes stehen sich der Reihenfolge nach gegenüber (Entfaltung ↔ Vorschrift, …). Dieselbe Position im Pol ist das Gegenstück. Der Import leitet daraus `counterpart_chunk_id` ab; in der YAML steht dafür kein eigenes Feld.

---

## Chunk-Modell

Weiterhin ein Chunk pro Knoten, Assistant-Partition (nicht `__shared__`), `chunk_type: typology`.

Neue Metadaten auf `ChunkMetadata` in ragrun und ragprep. Fehlende Felder heißen „trifft nicht zu“, nicht ein dritter Wert.

| Feld | Werte | Wer |
|---|---|---|
| `typology_slug` | `soziale-gesundheit`, `12-weltanschauungen` | jeder Knoten |
| `typology_role` | `overview`, `group`, `member` | jeder Knoten |
| `group_id` | `feld-1` … `feld-6`, für beide Pole gleich | Gruppe und ihre Einträge |
| `group_title` | Richtung, z. B. `Geist -> Recht`, für beide Pole gleich | Gruppe und ihre Einträge |
| `health` | `gesund`, `krank` | Gruppen und Einträge mit Pol |
| `counterpart_chunk_id` | Chunk des Gegenstücks im anderen Pol | Gruppen und Einträge mit Pol, deren Gegenstück existiert |
| `short_text` | Kurztext aus der YAML, nicht Teil des Chunk-Texts | jeder Knoten |
| `parent_id` | Chunk der nächsthöheren Ebene | Gruppe → Übersicht, Eintrag → Gruppe desselben Pols oder, ohne Gruppe, → Übersicht |

`segment_index` ist die Position des Aspekts in seinem Pol, ab 0, in der Reihenfolge der YAML. Der Gegenpol ist der Eintrag derselben `group_id` mit dem anderen `health` und demselben `segment_index`. Seine `chunk_id` steht als `counterpart_chunk_id` am Eintrag, auf beiden Seiten. Hat der andere Pol an dieser Position keinen Eintrag, bleibt `counterpart_chunk_id` leer. Für Gruppen gilt dasselbe ohne Position: Gegenstück ist die Gruppe derselben `group_id` mit dem anderen `health`.

`segment_title` ist der suchbare Name: Matrixtitel, Feldtitel des Pols (`titel`) oder Schlagwort / Weltanschauungsname. `aliases` tragen Schlagsatz, Feldtitel und die geläufigen Namensvarianten. `source_title` bleibt der Typologie-Name.

### IDs

Übersicht behält die bisherige ID, damit der bestehende Weltanschauungs-Chunk an Ort und Stelle weiterlebt:

`buildTypologyChunkId(assistant, name)` → `{assistant}:typology:{name}`

Neu, eigener Namespace-Anteil im Namensstring, nicht ein neuer UUID-Namespace:

- Gruppe: `{assistant}:typology:{slug}:group:{group_id}` und, wenn `health` gesetzt ist, `:{health}`
- Eintrag: `{assistant}:typology:{slug}:member:{key}` und, wenn `health` gesetzt ist, `:{health}`

`key` ist das normalisierte Schlagwort, bei den flachen Typologien der normalisierte `name` des Eintrags.

### source_id

Bisher teilen alle Typologien eines Assistenten eine Listen-`source_id` (`assistant:{name}:typology-list`). Ein Store-Lauf deprecated Orphans nur innerhalb einer `source_id`. Deshalb bekommt jede Typologie eine eigene:

`assistant:{name}:typology:{slug}`

Ein Lauf für Soziale Gesundheit fasst dann nicht die Weltanschauungen an. Die Übersicht der Weltanschauungen wird unter der neuen `source_id` mit derselben `chunk_id` upserted. Die alte Listen-`source_id` wird erst deprecated, wenn jede bisherige Typologie unter ihrer eigenen `source_id` liegt.

### Beispielbaum Soziale Gesundheit

- 1 Übersicht, `segment_title` = `Soziale Gesundheit`, ohne `health`
- 12 Gruppen (6 Felder × 2 Pole), `segment_title` = Feldtitel des Pols, `group_title` = Richtung, mit `health`
- 84 Aspekt-Einträge mit `health`, `parent_id` = Gruppe desselben Feldes und Pols

### Beispielbaum 12 Weltanschauungen

- 1 Übersicht (der heutige Erklärungschunk)
- 12 Einträge aus `list`, ohne `group_id`, ohne `health`, `parent_id` = Übersicht

Dieselbe Zweiteilung gilt für alle übrigen Typologien: Übersicht plus ein Eintrag pro Mitglied, keine Gruppen, kein `health`. Vier davon stammen aus `typologies-to-explain.jsonl` (Welträtsel, Triebfedern, Motive, Stimmungen), vier sind neu (Sinne, Epochen, Freiheiten, Temperamente). Ihre Übersicht hat noch keinen Chunk; ihre `chunk_id` ist trotzdem `buildTypologyChunkId(assistant, name)`.

| Datei | Form | Herkunft |
|---|---|---|
| `soziale-gesundheit.yaml` | Übersicht, 12 Gruppen, 84 Einträge, `health` | YAML |
| `12-weltanschauungen.yaml` | Übersicht, 12 Einträge | JSONL |
| `7-weltraetsel-des-materialismus.yaml` | Übersicht, 7 Einträge | JSONL |
| `4-stufen-der-sittlichen-triebfedern.yaml` | Übersicht, 4 Einträge | JSONL |
| `4-motive-sittlichen-handelns.yaml` | Übersicht, 4 Einträge | JSONL |
| `7-weltanschauungsstimmungen.yaml` | Übersicht, 7 Einträge | JSONL |
| `12-sinne.yaml` | Übersicht, 12 Einträge | neu |
| `4-epochen-der-gedankenentwicklung.yaml` | Übersicht, 4 Einträge | neu |
| `4-freiheiten-freier-software.yaml` | Übersicht, 4 Einträge | neu |
| `4-temperamente.yaml` | Übersicht, 4 Einträge | neu |

Zusammen 10 Übersichten, 12 Gruppen, 142 Einträge, also 164 Chunks.

`typologies-to-explain.jsonl` ist danach nicht mehr die Quelle, und der Listeneintrag `typologies: [typologies.jsonl]` im `assistant-manifest.yaml` wird nicht mehr gelesen. Die Übersichtstexte, die heute in `vector_chunks` liegen, werden der Langtext der Übersicht. Kurztexte und die Langtexte der Einträge (und bei Sozialer Gesundheit der Gruppen) werden außerhalb von ragprep in die YAML geschrieben, bevor irgendetwas gechunkt wird.

---

## Texte entstehen außerhalb von ragprep

Jeder Knoten, der ein Chunk wird, hat in der YAML einen Kurztext und einen Langtext: die Übersicht, jede Gruppe, jeder Eintrag. ragprep setzt diese Texte nicht zusammen, kürzt sie nicht und füllt keine Lücken. `rag:augment:typologies:explain` und der Endpoint `/graphs/typology-explain` werden entfernt.

## ragprep prüft nur

```
yarn rp rag:augment:typologies:import <assistant>
```

Liest jede YAML mit einem `typology:`-Block unter `paths.typologies` (`sources/typologies/`). Eine einzelne Datei geht über `--file <path>`.

Die Prüfung, sonst nichts:

- `typology` ist gesetzt. Übersicht, jede Gruppe und jeder Eintrag haben einen nicht leeren Kurztext und einen Langtext.
- Jeder Langtext, der ein Chunk wird (Übersicht, Gruppe, Eintrag), liegt im Längenfenster.
- `groups.exclude` nennt die Feldnummern, die nicht gechunkt werden. Jede andere Feldnummer ist eingeschlossen. Ein ausgeschlossenes Feld ohne Mitgliederliste ist in Ordnung. Ein eingeschlossenes Feld ohne Mitgliederliste ist ein Fehler. Eine Nummer in `exclude`, die in der Datei nicht vorkommt, ist ein Fehler.
- Fehlt das Frontmatter, bricht der Lauf ab, bevor er Längen prüft.

Besteht die Prüfung, werden die schon dastehenden Texte unverändert zu Chunks. Kurztext wird das Metadatum `short_text`, Langtext wird der Chunk-Text. Kein Kopf davor, kein `trim_to_embedding_budget`. Besteht sie nicht, sammelt der Lauf alle Verstöße und schreibt weder JSONL noch `store-chunks`. Jede Zeile nennt Typologie, Knoten (Feld, Pol, Name), ob Kurz oder Lang, Ist-Länge und die Grenze. Die YAML wird außerhalb von ragprep nachgezogen, danach der Lauf wiederholt.

### Frontmatter

Oben in der Inhalts-YAML. Es benennt die Schlüssel, es beschreibt die Texte nicht neu. Jeder Knoten hat `short` und `text`. Bei einer flachen Typologie fehlen `health` und `groups`; `overview.short`, `overview.text` und `overview.references` zeigen auf die Felder der Übersicht, wie in `12-weltanschauungen.yaml`.

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
| `aliases` | Suchnamen der Typologie, Metadatum der Übersicht; optional |
| `overview.text`, `groups.text`, `members.text` | Chunk-Text, wörtlich, das Längenfenster gilt hier |
| `overview.short`, `groups.short`, `members.short` | `short_text`, nicht Teil des Chunk-Texts; muss vorhanden sein |
| `members.name` | `segment_title` |
| `members.lead` | `aliases`, neben dem Feldtitel |
| `members.quotes` | `references`, nicht Teil des Chunk-Texts |
| `overview.references` | `references` der Übersicht (`chunk_id`, `relevance`, `description`), nicht Teil des Chunk-Texts |
| `groups.title` | `group_title` |
| `groups.label` | Feldtitel je Pol: `segment_title` der Gruppe, in `aliases` ihrer Einträge |
| `groups.short`, `groups.text` | je Pol-Eintrag des Feldes, also zwei Texte je Feld |
| `health.field` | `health` an Gruppe und Eintrag |
| `groups.exclude` | diese Feldnummern bleiben in der Datei und werden nicht gechunkt; alle übrigen sind eingeschlossen |

Fehlt `typology` oder `members.text`, bricht der Lauf ab, bevor er Längen prüft. `aliases` ist optional; fehlt der Schlüssel oder ist die Liste leer, trägt die Übersicht keine Typologie-Aliases.

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

Die Übersichts-`chunk_id` der heutigen Typologien bleibt `buildTypologyChunkId(assistant, name)`.

### Ablauf: von der YAML nach Supabase und Qdrant

ragprep schreibt weder in Supabase noch in Qdrant direkt. Beides geht über ragrun: erst `store-chunks` (Supabase), dann `embed-chunks` (Qdrant). Die Endpoints gibt es schon; `rag:embed` und `pushChunksToRagrun` benutzen sie.

1. **Prüfen.** Alle YAML-Dateien werden gelesen und geprüft. Ein Verstoß in irgendeiner Datei bricht den ganzen Lauf ab; es wird nichts geschrieben.
2. **Chunks bauen.** Pro Typologie Übersicht, Gruppen, Einträge mit `buildChunkOutput` und den IDs aus Stufe 2. `counterpart_chunk_id` wird erst gesetzt, wenn alle Einträge eines Feldes ihre `chunk_id` haben.
3. **Lokal ablegen.** `<paths.generated>/chunk-cache/typologies/<slug>.jsonl` im Assistenten (heute `.generated/`, nicht in Git), eine Zeile pro Chunk. Das ist die Spur des Laufs, nicht die Quelle.
4. **Supabase.** Pro Typologie ein `POST /rag/store-chunks` mit `collection_name` = `rag-collection` des Assistenten. ragrun upsertet nach `rag_chunks` (Assistant-Partition, Schlüssel `rag_partition` + `chunk_id`). Ändert sich der `content_hash`, wird `embedded_at` zurückgesetzt; sonst bleibt der Vektor gültig. Danach deprecated ragrun innerhalb der `source_id` dieser Typologie alle aktiven Zeilen, die im Lauf nicht vorkamen. Andere Typologien bleiben unberührt.
5. **Alte Listen-`source_id`.** Erst wenn alle zehn Typologien unter eigener `source_id` gespeichert sind, deprecated der Lauf die Zeilen unter `assistant:{name}:typology-list` (`POST /rag/deprecate-chunk-ids`). Die Übersichten der fünf alten Typologien liegen dann schon mit derselben `chunk_id` unter der neuen `source_id`; deprecated werden nur Zeilen, die keiner neuen `chunk_id` entsprechen.
6. **Qdrant.** Ein `POST /rag/embed-chunks` mit `collection_name`, `chunk_types: [typology]`, `source_ids` = die zehn neuen `source_id`s, `shared_source_ids: []` und `prefix_passage` aus dem `active-embedder` des Manifests (`e5` → `passage: `). ragrun lädt die aktiven Zeilen aus `rag_chunks`, holt die Vektoren beim Embedder, upsertet sie in die Qdrant-Collection des Assistenten, spiegelt sie in `vector_chunks` und setzt `embedded_at`. Danach löscht ragrun Vektoren dieser `(source_id, chunk_type)`-Paare, die in `rag_chunks` nicht mehr aktiv sind.
7. **Bericht.** Pro Typologie: Chunks gespeichert, deprecated, eingebettet.

`--dry-run` hört nach Schritt 3 auf. `--skip-embed` hört nach Schritt 5 auf; eingebettet wird dann später mit `rag:embed <assistant> --type typology`.

#### Embedder auf Tuxai

Eingebettet wird mit dem Embedder auf Tuxai (Pulse), in die Qdrant-Collection auf Tuxai. ragprep wählt dafür nur den ragrun, an den es sendet; welcher Embedder rechnet, entscheidet dieser ragrun über `RAGRUN_EMBEDDINGS_BASE_URL` und `RAGRUN_QDRANT_URL`.

- Regelweg: `RAGPREP_ENV=production yarn rp rag:augment:typologies:import philo-von-freisinn`. ragprep sendet an ragrun auf Railway (`.env.production`). Dieser ragrun liest Embedder und Qdrant auf Tuxai über den Cloudflare-Tunnel (`embed.ragxxx.com`, `qdrant.ragxxx.com`) und schreibt `rag_chunks` in das Supabase-Projekt von production.
- Mit lokalem ragrun: in `ragrun/.env` `RAGRUN_EMBEDDINGS_PROVIDER=http`, `RAGRUN_EMBEDDINGS_BASE_URL` und `RAGRUN_QDRANT_URL` auf Tuxai (Tunnel oder Tailnet), dazu `RAGRUN_QDRANT_API_KEY`. Mit den heutigen Werten (`localhost:8001`, `localhost:6333`) landen die Vektoren nicht auf Tuxai.

Vor Schritt 4 fragt der Lauf `GET /healthz` beim Ziel-ragrun ab (an der Wurzel, nicht unter `/api/v1`). Die Antwort nennt den `endpoint` von Qdrant und Embedder. Der Lauf gibt beide Hosts aus und bricht ab, wenn der Embedder auf `localhost` zeigt oder nicht `healthy` ist. Mit `--allow-local-embedder` lässt sich das für Tests übergehen.

---

## ragrun: Metadaten und Filter

`ChunkMetadata` nimmt die Felder aus der Tabelle oben auf. `health` nur `gesund` | `krank`. `typology_role` nur die drei Rollen. Unbekannte Werte lehnt `from_payload` ab, leere optionale Felder bleiben weg.

`search_corpus` und der Intent `erklaerung` ändern ihre Typ-Listen nicht. Neue Member-Chunks sind `chunk_type: typology` und fallen damit weiter unter den App-Typ `concept` (`begriff` + `typology`) und unter die Hybrid-Suche des Intents.

Der Lemma-Lookup `_lemma_lookup_multi_typology` trifft weiter über `segment_title` und `aliases`. Schlagwort und Weltanschauungsname liegen dort, sobald die Query im Action-Manifest wieder aktiv ist. Das ist nicht Teil dieses Plans; MCP ist die abgefragte Fläche.

---

## MCP

Zwei neue Tools. `search_corpus` bleibt der unscharfe Einstieg über `concept` und kennt die Struktur nicht. „Die anderen elf Weltanschauungen“ und „das kranke Gegenstück“ sind Nachschlagen, keine Ähnlichkeitssuche.

### `list_typologies`

Keine Argumente. Liefert alle Typologien mit aktivem Übersichts-Chunk aus `rag_chunks` (Assistant-Partition): `count`, `typologies[]` mit `typology_slug`, `title`, `overview_chunk_id`, `has_health`, `group_count`, `member_count`, `short_text`. Keine Vektorsuche — vollständiges Verzeichnis (derzeit zehn YAML-Dateien).

### `list_typology`

Argumente: `name` oder `chunk_id` (eines reicht).

Antwort:

- `typology_slug`, `title`, `has_health`
- `overview`: `chunk_id`, `segment_title`
- `groups`: `group_id`, `group_title`, `segment_title`, `health`, `chunk_id`, `counterpart_chunk_id` (leer bei Weltanschauungen)
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

Zehn Dateien unter `sources/typologies/`, Frontmatter wie oben (Tabelle im Abschnitt „Beispielbaum 12 Weltanschauungen“). Fünf stammen aus `typologies-to-explain.jsonl` bzw. lagen schon, vier sind neu, dazu `soziale-gesundheit.yaml`. Übersichtstexte aus `vector_chunks` werden der Langtext der Übersicht. Kurztexte und die übrigen Langtexte werden in die Dateien geschrieben, nicht erzeugt.

Fertig, wenn jede Datei `typology`, Kurz- und Langtext an Übersicht, Gruppe und Eintrag trägt, und `soziale-gesundheit.yaml` `exclude: [7, 8, 9]` hat. Die Längen müssen noch nicht stimmen. `typologies-to-explain.jsonl` wird nicht mehr gelesen.

### Stufe 2 — Chunk-Modell

`ChunkMetadata` in ragrun und ragprep nimmt `typology_slug`, `typology_role`, `group_id`, `group_title`, `health`, `counterpart_chunk_id`, `short_text` auf. IDs und `source_id` wie oben. Store lehnt unbekannte Werte für Rolle und `health` ab. Noch keine neuen Chunks.

Fertig, wenn ein Chunk mit diesen Feldern angenommen wird und einer ohne `health` ebenfalls.

### Stufe 3 — Erklärungsweg entfernen

`rag:augment:typologies:explain` und `/graphs/typology-explain` werden gelöscht, samt Prompt, Chain und Service. Nichts ruft sie mehr auf.

### Stufe 4 — Prüfung und Chunks

`rag:augment:typologies:import` liest die YAML-Dateien und prüft. Bei Verstoß Fehlerliste, keine Ausgabe.

Stand 19.10.2026: Alle zehn Dateien bestehen die Prüfung. Jede Übersicht, jede Gruppe und jeder Eintrag hat Kurz- und Langtext, jeder Langtext liegt im Fenster, die beiden Pole jedes Feldes haben je 7 Aspekte, `exclude: [7, 8, 9]` trifft vorhandene Felder. Nachgezogen wurden dafür außerhalb von ragprep:

- Soziale Gesundheit: Kurz- und Langtext der Übersicht und der 12 Gruppen; die Felder 7–9 bleiben unverändert
- Kurztexte der Übersicht bei Weltanschauungen, Triebfedern, Motiven, Stimmungen, Welträtseln
- Fünf Übersichten gekürzt (Sinne, Epochen, Freiheiten, Temperamente, Welträtsel), dazu die Einträge „Begriffliche Intuition“ und „Ursprung von Kraft und Materie“

Bei Soziale Gesundheit stehen Kurz- und Langtext der Gruppe an jedem Pol-Eintrag eines Feldes (`kurz`, `text` neben `titel`), also zwölf Paare.

Die Dateien werden außerhalb von ragprep nachgezogen, der Lauf wiederholt. Besteht jede Datei, läuft der Ablauf aus „Ablauf: von der YAML nach Supabase und Qdrant“ bis zum Embed auf Tuxai. Die alte Listen-`source_id` wird deprecated, sobald alle zehn unter der eigenen liegen.

`chunk-philo-all.sh` ruft danach `rag:augment:typologies:import` an der Stelle auf, an der bis Stufe 3 `typologies:explain` stand. `remediationHints.ts` schlägt bei fehlenden Typologie-Chunks wieder einen Befehl vor, diesen.

Fertig, wenn die Übersicht der zwölf Weltanschauungen ihre bisherige `chunk_id` behält, die Aspekte derselben Position auf beiden Polen denselben `segment_index` haben und über `counterpart_chunk_id` wechselseitig aufeinander zeigen, die beiden Gruppen eines Feldes ebenso, jeder Aspekt auf die Gruppe seines Pols zeigt, und die Felder 7–9 keine Chunks sind.

### Stufe 5 — MCP

`list_typologies`, `list_typology` und `search_typology` in `app/mcp_server/typology_tools.py`, registriert in `server.py`. `search_corpus` gibt Typology-Metadaten mit aus (`typology_role`, `group_title`, `health`, `counterpart_chunk_id`, …); `types` ohne `typology`. Unit-Tests: `tests/unit/test_typology_mcp_tools.py`. Live-Checks: `tests/mcp/TESTPLAN_MCP.md` (Typologie-Abschnitt).

---

## Nicht in diesem Schritt

- In ragprep Texte erzeugen, kürzen, auffüllen oder aus einer Übersicht auf Einträge verteilen
- `health` an Begriffe, Zitate oder Buch-Chunks
- Die auskommentierte Lemma-Query im Action-Manifest
- Die YAML wieder aus dem Projektordner lesen; sie liegt unter `sources/typologies/`
