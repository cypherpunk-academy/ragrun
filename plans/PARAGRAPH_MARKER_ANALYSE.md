# Analyse: Absatz-Marker-Pipeline

## Ziel

Lückenlose Dokumentation, wie `N|`-Absatz-Marker von Phase5 bis zur App-Anzeige
fließen — und wo Defekte sind.

Grundsatz (bestätigt durch Code):
> Die App nummeriert **niemals selbstständig**. Die Nummer kommt immer aus der DB.
> Die App fügt nur das Styling hinzu.

---

## 1. Phase5 → ragprep: Marker-Format

**Format in Phase5-MD-Dateien:**
```
1| In diesen vier Vorträgen wollen wir uns beschäftigen...
2| Im Jahre 1914 war es möglich...
```

Regex: `/^\d{1,4}\|(?:\s|$)/` (in `paragraphBoundaries.ts`)

**parseBookStructure.ts:**
- `ParsedParagraph.index` = 1-basiert, pro Kapitel (reset bei jedem Kapitel)
- `ParsedParagraph.numberPrefix` = geparster `N`-Wert
- `paragraph.text` enthält den `N|`-Präfix **weiterhin** (wird nicht entfernt)

---

## 2. Zwei getrennte Datenpfade ab ragprep

Nach dem Parsen gibt es **zwei unabhängige Datenwege**:

### Pfad A: rag_chunks → vector_chunks (Chunk-Text)

**Rollen:**
- `rag_chunks` = **Single Source of Truth** für alle Chunks (Postgres)
- `vector_chunks` = **Abbild von Qdrant** (Postgres-Spiegel) — enthält ausschließlich
  aktive Chunks; keine deprecated oder veralteten Einträge
- Qdrant = Vektorindex; Payload identisch mit `vector_chunks`

`rag:embed` hält `vector_chunks` und Qdrant synchron mit `rag_chunks`:
stale Chunk-IDs werden aus Qdrant/`vector_chunks` gelöscht, neue werden eingefügt,
geänderte werden re-embedded oder payload-only aktualisiert.

**Chunk-Text aus `step3BuildChunks.ts → buildChunks()`:**
```typescript
const chunkText = assembleTextForRanges(chapter, spec.sentenceRanges)
    .replace(/<[^>]+>/g, '')   // HTML-Tags entfernen
    .replace(/\u00AD/g, '')    // Soft-Hyphens
    .trim();
//  N|-Präfix bleibt ABSICHTLICH im Text (Navigation + Anzeige)
```

`metadata.paragraph` = `spec.paragraphNumbers[0]` (erster Absatz-Index, 1-basiert pro Kapitel)

Gespeichert identisch in:
- `rag_chunks.text` — mit `N|` und `<q>`/`<i>` drin
- `vector_chunks.text` — Spiegel, immer aktuell
- Qdrant payload `.text` — Spiegel, immer aktuell

**Verwendung des `N|`-Markers:** Navigation in `rag.py` via Regex `(^|\n\n){para}\|`.

**Für Embeddings** (`_prepare_embedding_text()` in `ingestion_service.py`):
```python
def _prepare_embedding_text(cls, text: str) -> str:
    """Normalize chunk text for embedding only — stored text unchanged."""
    stripped = cls._strip_markup(text)        # <q>/<i>: Tags weg, Innentext bleibt
    stripped = cls._PARA_MARKER_RE.sub("", stripped)  # N|-Marker weg (behoben)
    return cls._SOFT_HYPHEN_RE.sub("", stripped)
```

### Pfad B: rag_paragraphs (Absatz-Text)

`supabaseParagraphWriter.ts → writeParagraphsAndMapping()`:
```typescript
const rawNoPrefix = paragraph.text.replace(/^\d+\|\s*/, '');  // N| wird entfernt!
const { text_raw, annotations } = normalizeParagraphForDb(rawNoPrefix, anchorMap);
```

Gespeichert in:
- `rag_paragraphs.text_raw` — **ohne** `N|`, mit `<q>`/`<i>` als `annotations` JSONB
- `rag_paragraphs.paragraph_number` = `paragraph.index` (die Zahl aus dem `N|`)

**Verwendung:** Lese-Ansicht der App (synced via WatermelonDB).

`app_paragraph_chunk`: Mapping `paragraph_id (UUID) → chunk_id` (geschrieben von ragprep).

---

## 3. <q> und <i> Tags in der Datenbank

| Ort | `<q>`/`<i>` Tags | `N|`-Marker |
|-----|-----------------|-------------|
| `rag_chunks.text` | **vorhanden** (original) | **vorhanden** |
| `vector_chunks.text` | **vorhanden** | **vorhanden** |
| Embedding-Input | **entfernt** (Innentext bleibt) | **entfernt** — behoben |
| `rag_paragraphs.text_raw` | in `.annotations` JSONB konvertiert | **entfernt** |
| App ReadScreen render | via `ParagraphRenderer` + annotations | via `item.paragraphNumber` |

---

## 4. App-seitige Anzeige (ragapp)

### Wie die App Absätze zusammenstellt

**Datensource:** WatermelonDB lokale SQLite-DB, synced vom ragrun `/app/sync/pull` Endpoint.

**Abfrage (`ParagraphRepository.observeBySource`):**
```typescript
collection.query(
  Q.where('source_id', sourceId),
  Q.where('deprecated_at', null),    // nur aktive Absätze
  Q.sortBy('segment_index', Q.asc),
  Q.sortBy('paragraph_number', Q.asc),
).observe();
```

**ReadScreen:** Zeigt alle aktiven `paragraphs` für die aktuelle `sourceId`, gefiltert auf
den aktuellen `currentSegmentIndex` (Kapitel), sortiert nach `paragraph_number ASC`.
Kein eigenes Nummerieren — Reihenfolge und Nummer kommen vollständig aus der DB.

**Rendered per Absatz (`ReadScreen.tsx:416-418`):**
```tsx
<Text style={textStyles.readingParagraphNumber}>
  {item.paragraphNumber}{'| '}    // Zahl aus DB, App fügt nur Styling hinzu
</Text>
<ParagraphRenderer text={item.textRaw} annotations={item.annotations} />
```

- `item.paragraphNumber` ← `rag_paragraphs.paragraph_number` ← `N|`-Zahl aus Phase5
- `item.textRaw` ← `rag_paragraphs.text_raw` (ohne `N|`, ohne HTML-Tags)
- `item.annotations` ← `rag_paragraphs.annotations` JSONB (Kursiv, Fremdzitate, Seitenverweise)

### Duplikat-Problem: Absatz 1 erscheint zweimal

**Designprinzip:** Die App soll Kapitel und Vorträge **ausschließlich** aus `rag_paragraphs`
zusammenbauen. `rag_paragraphs` ist die einzige zulässige Quelle für den Lesetext.

**Symptom:** Absatz 1 erscheint beim Scrollen ein zweites Mal (mit Leerzeile davor).
In `rag_paragraphs` auf dem Server existiert der Absatz nur **einmal** — die App zeigt ihn
trotzdem doppelt. Die App baut also gerade nicht ausschließlich aus `rag_paragraphs`.

**Lokaler Datenpfad:**
```
rag_paragraphs (Server)
  → /app/sync/pull (WatermelonDB-Sync-Protokoll)
    → WatermelonDB lokale SQLite (assets/seed/db-snapshot.json als Startwert)
      → ParagraphRepository.observeBySource(sourceId)
        → ReadScreen
```

**Ursache:** Der lokale WatermelonDB-Cache enthält zwei Rows für Absatz 1, weil:

1. Das Seed-File (`assets/seed/db-snapshot.json`, generiert mit `npm run seed:fetch`)
   wurde zu einem Zeitpunkt erstellt, als `rag_paragraphs` noch **zwei aktive Rows**
   für Absatz 1 enthielt (Bug im ParagraphWriter: `deprecateOrphanParagraphs` nicht aufgerufen)
2. Der Server hat seitdem die alte Row deprecated → nur noch eine aktive Row
3. Die Deprecation (als `updated_at`-Änderung) müsste via Sync übertragen werden
4. Wenn der Sync die Deprecation nicht korrekt als `updated` oder `deleted` liefert,
   bleibt die alte Row im lokalen WatermelonDB — mit `deprecated_at = null`
5. Ergebnis: zwei lokale Rows mit `deprecated_at = null` → beide erscheinen im ReadScreen

**Wurzel des Problems — zwei Ebenen:**

| Ebene | Problem | Fix |
|-------|---------|-----|
| Server (`supabaseParagraphWriter`) | `deprecateOrphanParagraphs` wird nicht immer aufgerufen → Server hat zeitweise zwei aktive Rows | Immer aufrufen — unabhängig von `verify`-Option |
| App (Seed + Sync) | Seed enthält den Stand mit zwei Rows; Sync liefert Deprecation ggf. nicht vollständig | Sync-Endpoint prüfen ob deprecated Rows korrekt als `updated` oder `deleted` geliefert werden |

**Sofortdiagnose:**
```sql
-- Server: sollte genau 1 Row ergeben
SELECT id, segment_slug, paragraph_number, deprecated_at
FROM rag_paragraphs
WHERE source_id = '<lecture-source-id>'
ORDER BY paragraph_number, created_at;
```

Wenn der Server korrekt ist (1 aktive Row), liegt das Duplikat im lokalen WatermelonDB-Cache
und verschwindet nach einem sauberen Re-Sync (App-Daten löschen + neu synchronisieren).

### Such-Ansicht (SearchScreen / searchHitCard.ts)

```typescript
export function chunkPreviewText(r: SearchResult): string {
  return (r.text ?? r.snippet ?? '').trim();
}
// result.text = vector_chunks.text = Chunk-Text MIT N|-Marker!
// -> "1| Wenn wir heute darangehen..." erscheint im Snippet
```

**BUG:** Suchergebnis-Karten zeigen den rohen Chunk-Text inkl. `N|`-Prefix im
Body-Snippet. Der Body-Text sollte wie in der Lese-Ansicht den `N|`-Marker entfernt haben.

### Navigation

- Lese-Ansicht: navigiert via `paragraph_id` (UUID) — korrekt aus DB
- Suche: navigiert via `result.paragraph_id` — kommt aus `app_paragraph_chunk`-Join (ragrun)

---

## 5. Defekte und Fixes

### BUG #1: `N|`-Marker in Embedding-Text (ragrun) — BEHOBEN

**Ursache:** `_prepare_embedding_text()` in `ingestion_service.py` entfernte `<q>`/`<i>`, aber
nicht `N|`-Marker. Die Marker störten die semantische Qualität des Embeddings.

**Fix (bereits angewendet in `ingestion_service.py`):**
```python
_PARA_MARKER_RE = re.compile(r"(?m)^\d{1,4}\|\s?")

@classmethod
def _prepare_embedding_text(cls, text: str) -> str:
    stripped = cls._strip_markup(text)
    stripped = cls._PARA_MARKER_RE.sub("", stripped)   # N|-Marker entfernen
    return cls._SOFT_HYPHEN_RE.sub("", stripped)
```

Scope: Nur Embedding-Normalisierung — gespeicherter Text in DB bleibt unverändert.
Pending: Server-Neustart + Re-embed aller betroffenen Quellen nötig.

### BUG #2: `N|`-Marker in Such-Snippets (ragapp)

**Ursache:** `chunkPreviewText()` in `searchHitCard.ts` gibt `result.text` unbereinigt zurück.
`result.text` ist der rohe Chunk-Text aus `vector_chunks` inkl. `N|`.

**Fix in `searchHitCard.ts`:**
```typescript
const PARA_MARKER_RE = /^(\d{1,4})\|\s?/gm;

export function chunkPreviewText(r: SearchResult): string {
  const raw = (r.text ?? r.snippet ?? '').trim();
  return raw.replace(PARA_MARKER_RE, '');
}
```

Oder alternativ: ragrun API gibt `result.text` bereits bereinigt zurück (server-seitig besser
kontrollierbar, dann muss die App nichts wissen).

### BUG #3: Falsches Vortragsdatum in Such-Karte (chunk_vortrag)

**Symptom:** Karte zeigt `"DORNACH, 11. OKTOBER 1921"` (= ERSTER VORTRAG) als Überschrift,
Untertitel zeigt korrekt `"(ZWEITER VORTRAG Dornach, 12. Oktober 1921)"`.

**Erklärung der Felder:**
- `headlineLarge` kommt aus `result.lecture_date` → `formatMetaDate("1921-10-11")` → falsch
- `subHeadSmall` kommt aus `result.segment_title` → `"ZWEITER VORTRAG Dornach, 12. Oktober 1921"` → korrekt

**Hypothese:** `metadata.lecture_date` in `rag_chunks` enthält `"1921-10-11"` statt `"1921-10-12"`.
Das passiert in `persistLectureChunks` (ragprep), wo der Katalogeintrag für den ZWEITEN
VORTRAG dem ERSTEN zugeordnet wird (oder umgekehrt).

**Zu prüfen (DB-Query):**
```sql
SELECT
  metadata->>'lecture_date'  AS date,
  metadata->>'segment_title' AS segment,
  metadata->>'lecture_id'    AS lecture_id,
  chunk_id
FROM rag_chunks
WHERE source_id LIKE '%339%'
  AND metadata->>'chunk_type' IN ('book','secondary_book')
ORDER BY metadata->>'segment_title'
LIMIT 20;
```

**Fix:** In ragprep's Lecture-Zuweisung prüfen, ob `usedLectureIndices` oder
`findLectureEntryByDatumAndBook` den falschen Katalogeintrag zurückgibt.
Nach Fix: `rag:chunk` + `rag:embed` für GA 339 nötig.

### BUG #4: Doppelter Absatz in Lese-Ansicht — fehlende Orphan-Deprecation im ParagraphWriter

**Symptom:** Absatz 1 (und ggf. weitere) erscheinen im ReadScreen doppelt.

**Ursache:** Bug in `supabaseParagraphWriter.ts`. `deprecateOrphanParagraphs` wird nur
dann aufgerufen, wenn die `verify`-Option übergeben wird — also nicht immer.

Wenn beim Re-Ingest der `segment_slug` sich ändert (Titel-Normalisierung o.ä.), findet
`planParagraphIngestForParsedBook` die alten DB-Rows unter dem alten Slug nicht (sucht
nur nach dem neuen Slug). Neue Rows werden angelegt. Alte Rows bleiben aktiv.
Beide erscheinen in `paragraphClausesForSource` — Duplikat im ReadScreen.

**Betroffene Funktion:** `writeParagraphRowsAndMappings` in `supabaseParagraphWriter.ts`

```typescript
// Aktuell: deprecateOrphanParagraphs wird nur im verify-Block aufgerufen
if (options?.verify) {
    // ...
    const orphansDeprecated = await deprecateOrphanParagraphs(client, sourceId, activeIds);
    // ...
}
```

**Fix:** `deprecateOrphanParagraphs` muss **immer** aufgerufen werden — unabhängig von
`verify`. Nach dem Upsert aller neuen/aktualisierten Rows sind die aktiven IDs bekannt;
alle anderen Rows derselben `source_id` müssen deprecatiert werden:

```typescript
// Direkt nach dem Paragraph-Upsert, vor dem Commit:
for (const [sourceId, activeIds] of activeIdsBySource) {
    await deprecateOrphanParagraphs(client, sourceId, activeIds);
}
```

**Datei:** `ragprep/src/lib/supabaseParagraphWriter.ts` — Funktion `writeParagraphRowsAndMappings`

**DB-Diagnose:**
```sql
SELECT id, segment_slug, paragraph_number, created_at, deprecated_at
FROM rag_paragraphs
WHERE source_id = '<lecture-source-id>'
ORDER BY paragraph_number, created_at;
```

---

## 6. Betroffene Dateien

| Datei | Thema |
|-------|-------|
| `ragrun/app/ingestion/services/ingestion_service.py` | BUG #1: `_prepare_embedding_text()` |
| `ragapp/src/shared/lib/searchHitCard.ts` | BUG #2: `chunkPreviewText()` |
| `ragrun/app/ingestion/services/ingestion_service.py` | BUG #1: behoben — `_PARA_MARKER_RE` + `_prepare_embedding_text()` |
| `ragapp/src/shared/lib/searchHitCard.ts` | BUG #2: `chunkPreviewText()` |
| `ragprep/src/cli/commands/ragEmbed/index.ts` | BUG #3: `persistLectureChunks` |
| `ragprep/src/lib/supabaseParagraphWriter.ts` | BUG #4: `deprecateOrphanParagraphs` |
| `ragprep/src/cli/commands/ragChunk/step3BuildChunks.ts` | Referenz: `paragraph` korrekt gesetzt |
| `ragapp/src/features/read/ReadScreen.tsx:417` | Referenz: styled `N|`-Prefix korrekt |
| `ragapp/src/data/repositories/paragraphQuery.ts` | Referenz: Abfrage nach `source_id + deprecated_at IS NULL` |

---

## 7. Empfohlene Reihenfolge

1. **BUG #4** (DB: doppelte Absätze, kurzfristig SQL, dauerhaft in `supabaseParagraphWriter.ts`)
2. **BUG #2** (ragapp: `chunkPreviewText()` — sofort sichtbar, kein Re-Ingest)
3. **BUG #1** bereits gefixt — Server-Neustart + Re-embed ausstehend
4. **BUG #3** (nach DB-Query bestätigen, dann `rag:chunk` + `rag:embed` für GA 339)
