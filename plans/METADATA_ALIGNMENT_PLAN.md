# Metadata-Alignment Plan: ragprep ↔ ragrun

**Erstellt:** 2025-12-05  
**Status:** INITIAL ANALYSIS

## Zusammenfassung

Dieser Plan dokumentiert die Diskrepanzen zwischen den Metadaten-Definitionen in `ragrun` (Qdrant-basiertes Backend) und `ragprep` (TypeScript Chunking-Pipeline) sowie den vorhandenen Daten in `ragkeep`.

**Haupterkenntnis:** Die ragprep-Schema-Definition ist **unvollständig** und muss erweitert werden, um mit ragrun kompatibel zu sein. Die vorhandenen Daten in ragkeep verwenden jedoch **bereits zusätzliche Felder**, die weder in ragprep noch in ragrun definiert sind.

---

## 1. Schema-Vergleich

### 1.1 ragrun: ChunkMetadata (Python/Pydantic)

**Datei:** `ragrun/ragrun/models.py`

**Pflichtfelder:**
- `source_id`: str
- `chunk_id`: str
- `chunk_type`: str (validiert gegen CHUNK_TYPE_ENUM)
- `content_hash`: str (SHA-256)
- `created_at`: datetime
- `updated_at`: datetime
- `language`: str (2-5 Zeichen, ISO-Code)

**Optionale Felder:**
- `author`: Optional[str]
- `source_title`: Optional[str]
- `source_index`: Optional[int] (≥0)
- `segment_id`: Optional[str]
- `segment_title`: Optional[str]
- `segment_index`: Optional[int] (≥0)
- `parent_id`: Optional[str]
- `worldview`: Optional[str] (z.B. "Idealismus", "Realismus")
- `importance`: int (1-10, default: 5)
- `text`: Optional[str]
- `source_type`: Optional[str]
- `tags`: List[str] (default: [])

**Erlaubte chunk_type-Werte (CHUNK_TYPE_ENUM):**
```python
"book", "secondary_book", "chapter_summary", "begriff_list", 
"talk", "talk_summary", "essay", "essay_summary", "quote", 
"explanation", "explanation_summary", "typology"
```

---

### 1.2 ragprep: ChunkOutput.metadata (TypeScript)

**Datei:** `ragprep/src/types/ragSchemas.ts`

**Aktuelle Definition:**
```typescript
metadata: {
    author: string;              // REQUIRED
    source_id: string;           // REQUIRED
    source_title: string;        // REQUIRED
    segment_id: string | null;
    segment_title: string;       // REQUIRED
    segment_index: number | null;
    parent_id: string | null;
    chunk_id: string;            // REQUIRED
    chunk_type: ChunkType;       // REQUIRED
    importance: Importance;      // REQUIRED (1-10 enum)
}
```

**ChunkType enum in ragprep:**
```typescript
enum ChunkType {
    BOOK = 'book',
    SECONDARY_BOOK = 'secondary_book',
    CHAPTER_SUMMARY = 'chapter_summary',
    CONCEPT = 'concept',           // ⚠️ nicht in ragrun
    QUESTION = 'question',         // ⚠️ nicht in ragrun
    ESSAY = 'essay',
    ESSAY_SUMMARY = 'essay_summary',
    QUOTE = 'quote',
    EXPLANATION = 'explanation',
    ORDER = 'order',               // ⚠️ nicht in ragrun
}
```

---

### 1.3 Diskrepanzen

#### Felder in ragrun, aber NICHT in ragprep:
1. ❌ `source_index: Optional[int]` 
2. ❌ `worldview: Optional[str]`
3. ❌ `content_hash: str` (Pflichtfeld!)
4. ❌ `created_at: datetime` (Pflichtfeld!)
5. ❌ `updated_at: datetime` (Pflichtfeld!)
6. ❌ `source_type: Optional[str]`
7. ❌ `language: str` (Pflichtfeld!)
8. ❌ `tags: List[str]`

#### Unterschiedliche Optionalität:
- `author`: in ragprep REQUIRED, in ragrun Optional
- `source_title`: in ragprep REQUIRED, in ragrun Optional
- `segment_title`: in ragprep REQUIRED, in ragrun Optional

#### Chunk-Types nicht in ragrun:
- `concept` (ragprep) → vermutlich `begriff_list` in ragrun
- `question` (ragprep) → fehlt in ragrun
- `order` (ragprep) → fehlt in ragrun

#### Chunk-Types nicht in ragprep:
- `begriff_list` → vermutlich `concept`
- `talk`, `talk_summary` → fehlen
- `explanation_summary` → fehlt
- `typology` → fehlt

---

## 2. Vorhandene Daten-Analyse

### 2.1 Book-Chunks (chunks.jsonl)

**Beispiel:** `ragkeep/books/Rudolf_Steiner#Die_Philosophie_der_Freiheit#4/results/rag-chunks/chunks.jsonl`

**Struktur:**
```json
{
  "text": "...",
  "metadata": {
    "chunk_id": "4b8e4c2a-3f1b-4d2e-9c4a-8e4f4b2c3a1d__1-vorrede-zur-neuausgabe__1__pna__ea3c2904c0",
    "chunk_index": 0,           // ⚠️ nicht in ragprep/ragrun
    "book_id": "4b8e4c2a-...",  // ⚠️ nicht in ragprep/ragrun
    "author": "Rudolf Steiner",
    "book_index": 4,            // ⚠️ nicht in ragprep/ragrun
    "book_title": "Die Philosophie der Freiheit",
    "book_subtitle": "Grundzüge einer modernen Weltanschauung",  // ⚠️ nicht in ragprep/ragrun
    "chapter_level_1": "VORREDE ZUR NEUAUSGABE",  // ⚠️ nicht in ragprep/ragrun
    "chapter_level_2": null,    // ⚠️ nicht in ragprep/ragrun
    "chapter_level_3": null,    // ⚠️ nicht in ragprep/ragrun
    "paragraph_numbers": [1],   // ⚠️ nicht in ragprep/ragrun
    "paragraph_page": null,     // ⚠️ nicht in ragprep/ragrun
    "content_length": 1737,     // ⚠️ nicht in ragprep/ragrun
    "chunk_type": "book",
    "created_at": "2025-10-22T10:38:35.313Z",
    "importance": 9
  }
}
```

**Fehlende Pflichtfelder (laut ragrun):**
- ❌ `source_id` (wird als `book_id` gespeichert?)
- ❌ `source_title` (wird als `book_title` gespeichert?)
- ❌ `content_hash`
- ❌ `updated_at`
- ❌ `language`

**Zusätzliche Felder (nicht in ragprep/ragrun):**
- `chunk_index`
- `book_id` (könnte `source_id` sein)
- `book_index` (könnte `source_index` sein)
- `book_title` (könnte `source_title` sein)
- `book_subtitle`
- `chapter_level_1`, `chapter_level_2`, `chapter_level_3`
- `paragraph_numbers`
- `paragraph_page`
- `content_length`

---

### 2.2 Assistant-Chunks (concepts.jsonl, summaries.jsonl)

**Beispiel:** `ragkeep/assistants/philo-von-freisinn/concepts/concepts.jsonl`

**Struktur:**
```json
{
  "concept": "Denken",
  "augmentId": "25b02706-9dc4-48a2-8a55-b1c813aa37d9",
  "examples": ["..."],
  "significance": 9.357142857142858,
  "explanation": "..."
}
```

**Beispiel:** `ragkeep/assistants/philo-von-freisinn/summaries/chapters.jsonl`

**Struktur:**
```json
{
  "assistant": "philo-von-freisinn",
  "augmentKind": "summaries",
  "augmentId": "57b11402-f23d-4a62-a3e9-4f5f033df16d",
  "bookDir": "Rudolf_Steiner#Die_Philosophie_der_Freiheit#4",
  "chapterIndex": 1,
  "chapterId": "vorrede-zur-neuausgabe",
  "tokensTarget": 280,
  "summary": "...",
  "previousSummariesContextChars": 0,
  "createdAt": "2025-10-19T12:52:22.804Z",
  "model": "deepseek-chat"
}
```

**Analyse:**
- Diese Dateien verwenden ein **komplett anderes Schema**
- Sie sind **nicht chunk-basiert**, sondern konzept- bzw. zusammenfassungs-basiert
- Sie müssen **transformiert** werden, um als Chunks in ragrun hochgeladen zu werden

---

## 3. ragUpload-Analyse

**Datei:** `ragprep/src/cli/commands/ragUpload/index.ts`

**Aktuelles Verhalten:**
1. Liest `chunks.jsonl` aus `<bookDir>/results/rag-chunks/`
2. Parst jede Zeile als JSON
3. Filtert optional nach `chunkIds` oder `chapter_level_X`
4. Sendet Chunks als JSONL-String an `/rag/upload-chunks`
5. Verwendet `assistant-manifest.yaml` für Collection-Name

**Probleme:**
- ❌ Keine Validierung gegen ragrun-Schema
- ❌ Keine Transformation von Feldnamen (z.B. `book_id` → `source_id`)
- ❌ Keine Ergänzung fehlender Pflichtfelder
- ❌ Keine Unterstützung für Assistant-Chunks (concepts, summaries)

---

## 4. Handlungsempfehlungen

### 4.1 ragprep/src/types/ragSchemas.ts - ÄNDERN

**Ziel:** Schema an ragrun anpassen

```typescript
export type ChunkMetadata = {
    // === Core Identification ===
    source_id: string;                // REQUIRED: Stable source identifier
    chunk_id: string;                 // REQUIRED: Unique chunk identifier
    chunk_type: ChunkType;            // REQUIRED: Type enum
    
    // === Source Info ===
    author?: string;                  // Optional: Primary author/creator
    source_title?: string;            // Optional: Human-readable title
    source_index?: number;            // Optional: Index within source (≥0)
    source_type?: string;             // Optional: Container type (book, essay, etc.)
    
    // === Segment Info ===
    segment_id?: string;              // Optional: Logical segment identifier
    segment_title?: string;           // Optional: Segment title/label
    segment_index?: number;           // Optional: Index within segment (≥0)
    
    // === Relationships ===
    parent_id?: string;               // Optional: Reference to original artifact
    
    // === Classification ===
    worldview?: string;               // Optional: Assistant affinity (Idealismus, etc.)
    importance: number;               // REQUIRED: 1-10, default 5
    tags?: string[];                  // Optional: Free-form tags
    
    // === Technical ===
    content_hash: string;             // REQUIRED: SHA-256 of canonical text
    language: string;                 // REQUIRED: ISO language code (2-5 chars)
    created_at: string;               // REQUIRED: ISO timestamp
    updated_at: string;               // REQUIRED: ISO timestamp
};

export enum ChunkType {
    BOOK = 'book',
    SECONDARY_BOOK = 'secondary_book',
    CHAPTER_SUMMARY = 'chapter_summary',
    BEGRIFF_LIST = 'begriff_list',   // NEW (renamed from CONCEPT)
    TALK = 'talk',                    // NEW
    TALK_SUMMARY = 'talk_summary',    // NEW
    ESSAY = 'essay',
    ESSAY_SUMMARY = 'essay_summary',
    QUOTE = 'quote',
    EXPLANATION = 'explanation',
    EXPLANATION_SUMMARY = 'explanation_summary',  // NEW
    TYPOLOGY = 'typology',            // NEW
    // Legacy support (deprecated):
    CONCEPT = 'concept',              // Map to begriff_list
    QUESTION = 'question',            // Keep for backward compat
    ORDER = 'order',                  // Keep for backward compat
}

export type ChunkOutput = {
    text: string;
    metadata: ChunkMetadata;
};
```

**Migration:**
- Alte `ChunkOutput.metadata` erweitern
- `Importance` enum durch `number` (1-10) ersetzen
- Default-Werte hinzufügen (z.B. `language: 'de'`)

---

### 4.2 ragUpload - ERWEITERN

**Neue Features:**

1. **Schema-Validierung:**
   ```typescript
   import { ChunkMetadata } from '@/types/ragSchemas';
   
   function validateChunk(chunk: any): ChunkMetadata {
       // Validate required fields
       // Transform legacy field names
       // Add missing fields with defaults
       // Return validated metadata
   }
   ```

2. **Feld-Mapping:**
   ```typescript
   const fieldMap = {
       'book_id': 'source_id',
       'book_title': 'source_title',
       'book_index': 'source_index',
   };
   ```

3. **Fehlende Felder ergänzen:**
   ```typescript
   const defaults = {
       language: 'de',
       updated_at: chunk.created_at || new Date().toISOString(),
       content_hash: computeHash(chunk.text),
       tags: [],
   };
   ```

4. **Assistant-Chunk-Support:**
   - Erkenne `.jsonl`-Dateien in `assistants/*/concepts/`, `assistants/*/summaries/`
   - Transformiere zu Standard-Chunks
   - Setze `chunk_type` entsprechend (`begriff_list`, `chapter_summary`)

---

### 4.3 Vorhandene Daten - NICHT ÄNDERN

**Wichtig:** Die bestehenden `chunks.jsonl`-Dateien in `ragkeep/books/` sollen **nicht modifiziert** werden.

**Grund:**
- Sie sind bereits korrekt für den aktuellen Stand von ragprep
- Die Transformation erfolgt **zur Laufzeit** in `ragUpload`

**Hinweis für fehlende Felder:**

| Feld | Status | Hinweis |
|------|--------|---------|
| `source_id` | ⚠️ fehlt | Kann aus `book_id` abgeleitet werden |
| `source_title` | ⚠️ fehlt | Kann aus `book_title` abgeleitet werden |
| `source_index` | ⚠️ fehlt | Kann aus `book_index` abgeleitet werden |
| `content_hash` | ❌ fehlt | Muss zur Laufzeit berechnet werden: `sha256(text)` |
| `updated_at` | ❌ fehlt | Kann auf `created_at` gesetzt werden |
| `language` | ❌ fehlt | Default: `'de'` (alle Daten sind Deutsch) |
| `worldview` | ℹ️ optional | Kann aus `assistant-manifest.yaml` abgeleitet werden |
| `tags` | ℹ️ optional | Default: `[]` |
| `source_type` | ℹ️ optional | Default: `'book'` |

**Nicht-gemappte Felder (bleiben in Daten, werden von ragrun ignoriert):**
- `chunk_index`
- `book_subtitle`
- `chapter_level_1`, `chapter_level_2`, `chapter_level_3`
- `paragraph_numbers`, `paragraph_page`
- `content_length`

Diese Felder können in der Zukunft als `tags` oder in einem erweiterten Schema genutzt werden.

---

## 5. Assistant-Chunks: Transformation

### 5.1 concepts.jsonl → Chunks

**Mapping:**
```typescript
{
  text: concept.explanation,
  metadata: {
    source_id: `assistant:${assistant}:concepts`,
    chunk_id: concept.augmentId,
    chunk_type: 'begriff_list',
    author: assistant,
    source_title: `Konzepte von ${assistant}`,
    segment_id: concept.concept,
    segment_title: concept.concept,
    importance: Math.round(concept.significance),
    content_hash: sha256(concept.explanation),
    language: 'de',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    tags: ['concept', 'assistant-generated'],
  }
}
```

### 5.2 summaries.jsonl → Chunks

**Mapping:**
```typescript
{
  text: summary.summary,
  metadata: {
    source_id: summary.bookDir,
    chunk_id: summary.augmentId,
    chunk_type: 'chapter_summary',
    author: 'Rudolf Steiner',  // from bookDir
    source_title: extractTitle(summary.bookDir),
    segment_id: summary.chapterId,
    segment_title: extractChapterTitle(summary.summary),
    segment_index: summary.chapterIndex,
    parent_id: summary.bookDir,
    worldview: getWorldview(summary.assistant),
    importance: 7,  // summaries sind wichtig
    content_hash: sha256(summary.summary),
    language: 'de',
    created_at: summary.createdAt,
    updated_at: summary.createdAt,
    tags: ['summary', 'assistant-generated', summary.assistant],
  }
}
```

---

## 6. Implementierungs-Reihenfolge

### Phase 1: Schema-Update (ragprep)
1. ✅ `ragprep/src/types/ragSchemas.ts` erweitern
2. ✅ `ChunkMetadata` mit allen ragrun-Feldern definieren
3. ✅ `ChunkType` enum um fehlende Typen ergänzen
4. ✅ Bestehende `ChunkOutput`-Nutzung prüfen

### Phase 2: ragUpload erweitern
1. ✅ Schema-Validierung implementieren
2. ✅ Feld-Mapping (legacy → new) implementieren
3. ✅ Fehlende Pflichtfelder ergänzen (defaults + computed)
4. ✅ Hash-Berechnung (`crypto.createHash('sha256')`)
5. ✅ Tests schreiben

### Phase 3: Assistant-Chunks-Support
1. ✅ Erkenne Assistant-JSONL-Dateien
2. ✅ Transformiere `concepts.jsonl` → Chunks
3. ✅ Transformiere `summaries.jsonl` → Chunks
4. ✅ Neuer Command: `rag:upload:assistant`?

### Phase 4: Dokumentation
1. ✅ README aktualisieren
2. ✅ Migration-Guide schreiben
3. ✅ Beispiele für neue Chunk-Typen

---

## 7. Offene Fragen

### 7.1 Chunk-Type Mapping

**Frage:** Soll `concept` (ragprep) dauerhaft als Alias für `begriff_list` (ragrun) bestehen bleiben?

**Empfehlung:** Ja, für Rückwärtskompatibilität. In ragUpload wird `concept` automatisch zu `begriff_list` transformiert.

---

### 7.2 Question & Order Types

**Frage:** Was passiert mit `question` und `order` aus ragprep?

**Optionen:**
1. **In ragrun hinzufügen** (empfohlen)
2. **Auf bestehende Types mappen** (z.B. `question` → `explanation`)
3. **Ablehnen beim Upload** (breaking change)

**Empfehlung:** Option 1 – ragrun erweitern:
```python
CHUNK_TYPE_ENUM = (
    ...,
    "question",
    "order",
)
```

---

### 7.3 Worldview-Ableitung

**Frage:** Wie wird `worldview` für Book-Chunks bestimmt?

**Optionen:**
1. Aus `assistant-manifest.yaml` lesen
2. Als CLI-Parameter übergeben (`--worldview Idealismus`)
3. Leer lassen (optional field)

**Empfehlung:** Option 2 + 3 – optional Parameter, sonst leer.

---

### 7.4 Historische Daten

**Frage:** Sollen alte Chunks re-generiert werden?

**Antwort:** **NEIN.** Die Daten bleiben unverändert. Transformation erfolgt zur Laufzeit in ragUpload.

**Ausnahme:** Wenn `content_hash` oder andere Pflichtfelder fehlen, werden sie beim Upload berechnet, aber die Quelldatei bleibt unangetastet.

---

## 8. Nächste Schritte

1. **Bestätigung einholen:** User-Feedback zu diesem Plan
2. **ragprep Schema erweitern** (Phase 1)
3. **ragUpload refactoren** (Phase 2)
4. **Testing:** Uploads gegen ragrun/qdrant testen
5. **Assistant-Support** (Phase 3)

---

## Anhang

### A.1 Beispiel: Validierte Chunk-Transformation

**Input (chunks.jsonl):**
```json
{
  "text": "Zwei Wurzelfragen...",
  "metadata": {
    "chunk_id": "4b8e4c2a-...__1-vorrede-zur-neuausgabe__1__pna__ea3c2904c0",
    "chunk_index": 0,
    "book_id": "4b8e4c2a-3f1b-4d2e-9c4a-8e4f4b2c3a1d",
    "author": "Rudolf Steiner",
    "book_index": 4,
    "book_title": "Die Philosophie der Freiheit",
    "chunk_type": "book",
    "created_at": "2025-10-22T10:38:35.313Z",
    "importance": 9
  }
}
```

**Output (nach Transformation):**
```json
{
  "text": "Zwei Wurzelfragen...",
  "metadata": {
    "source_id": "4b8e4c2a-3f1b-4d2e-9c4a-8e4f4b2c3a1d",
    "chunk_id": "4b8e4c2a-...__1-vorrede-zur-neuausgabe__1__pna__ea3c2904c0",
    "chunk_type": "book",
    "author": "Rudolf Steiner",
    "source_title": "Die Philosophie der Freiheit",
    "source_index": 4,
    "segment_title": "VORREDE ZUR NEUAUSGABE",
    "importance": 9,
    "content_hash": "3a92db7d57c8f4b9e2a1...",
    "language": "de",
    "created_at": "2025-10-22T10:38:35.313Z",
    "updated_at": "2025-10-22T10:38:35.313Z",
    "tags": []
  }
}
```

---

**Ende des Plans**


