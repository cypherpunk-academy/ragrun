## ESSAY_LANGCHAIN — `essay:create` (ragprep → ragrun LangGraph/LangChain → ragprep)

Ziel: Ein Essay **in 7 Stufen** generieren lassen. Jede Stufe nutzt
- **Pitch-Abschnitt** (aus `pitch.json`)
- **Dense Retrieval** in Qdrant aus **primary_books** (= `chunk_type: book`) mit **k=5**
- **Prompt pro Stufe** (`step1_...prompt` … `step7_...prompt`)
- baut auf dem vorherigen Text (`essay_draft`) auf

Nach Stufe 7 liefert `ragrun` den finalen Essay-Text an `ragprep` zurück.

---

## CLI (ragprep): `essay:create`

### Aufruf

- `rp essay:create <assistant> <pitch_json_path>`

Beispiel:
- `rp essay:create philo-von-freisinn @ragkeep/assistants/philo-von-freisinn/essays/ich-kann-gegen-andere-gewinnen/pitch.json`

### Parameter-Semantik

- **assistant**: Assistant-Name (wie bei anderen Commands via `resolveAssistantInteractive`).
- **pitch_json_path**: Pfad zur `pitch.json` (JSON Array).
  - Aus dem Pfad wird **nur der Essay-Name/Slug** verwendet: Parent-Directory, z.B. `ich-kann-gegen-andere-gewinnen`.
  - Der Inhalt der `pitch.json` wird gelesen und an `ragrun` geschickt (ragrun muss die Datei nicht lokal haben).

### Output / Side Effects (ragprep)

- ruft `ragrun` HTTP API auf und gibt den finalen Text auf stdout aus
- optional (empfohlen): schreibt den Output in
  - `ragkeep/assistants/<assistant>/essays/<slug>/draft.md` (oder `essay.md`)

---

## CLI (ragprep): `essay:finetune`

Ziel: Einen bereits generierten Essay anhand einer **Änderungsanweisung** überarbeiten.

### Aufruf

- `rp essay:finetune <assistant> <essay_slug> --instruction "<text>"`

Optional:
- `--instruction-file <path>` statt `--instruction`
- `--in <path>` (Input-Datei setzen; default: `assistants/<assistant>/essays/<slug>/essay.md`)
- `--out <path>` (Output-Datei setzen; default: `assistants/<assistant>/essays/<slug>/essay.finetuned.md`)

---

## HTTP API (ragrun)

### Endpoint

Für den Start (philo-von-freisinn):
- `POST /api/v1/agent/philo-von-freisinn/graphs/essay-create`
- `POST /api/v1/agent/philo-von-freisinn/graphs/essay-finetune`

Später erweiterbar auf andere Assistants.

### Request (JSON)

```json
{
  "assistant": "philo-von-freisinn",
  "essay_slug": "ich-kann-gegen-andere-gewinnen",
  "essay_title": "ich-kann-gegen-andere-gewinnen",
  "pitch_steps": ["... 7 strings ..."],
  "k": 5,
  "verbose": false
}
```

Regeln:
- `pitch_steps`: exakt 7 Einträge; wenn `pitch.json` mehr Einträge enthält, nimmt ragprep **die ersten 7**.
- `k`: default 5
- `essay_title`: initial identisch zu `essay_slug` (kann später erweitert werden).

### Response (JSON)

```json
{
  "assistant": "philo-von-freisinn",
  "essay_slug": "ich-kann-gegen-andere-gewinnen",
  "essay_title": "ich-kann-gegen-andere-gewinnen",
  "final_text": "...",
  "steps": [
    { "step": 1, "prompt_file": "step1_okkultismus_draft.prompt" },
    { "step": 2, "prompt_file": "step2_transzendentalismus_draft.prompt" }
  ]
}
```

`steps[]` ist optional/kurz (Debug/Trace), `final_text` ist der Primärwert.

---

## LangGraph/LangChain Pipeline (ragrun)

### State

- `assistant`: string
- `collection`: string (Qdrant collection)
  - Default: aus `assistant-manifest.yaml` → `rag-collection`
  - Fallback: `assistant`
- `essay_slug`: string
- `essay_title`: string
- `pitch_steps`: list[str] (len=7)
- `k`: int (default 5)
- `step_index`: int (1..7)
- `essay_draft`: string (akkumuliert)

### Loop über 7 Schritte

Für `i = 1..7`:

1. **Retrieval Query**
   - `query = pitch_steps[i-1]`
2. **Dense Retrieval (Qdrant)**
   - `k = 5` (Startwert)
   - Filter: `chunk_type == "book"` (primary)
   - keine worldview-filter
3. **Context bauen**
   - join der Top-K chunk texts (z.B. max_chars=12000)
4. **Prompt laden**
   - System: `assistants_root/<assistant>/prompts/essays/instruction.prompt`
   - User: pro Step ein Template, z.B.
     - `.../prompts/essays/step1_okkultismus_draft.prompt`
     - `.../prompts/essays/step2_transzendentalismus_draft.prompt`
     - …
5. **Template-Variablen**
   - `{essay_title}` = essay_title
   - `{text}` = pitch_steps[i-1]
   - `{context}` = Retrieval-Context (nur primary_books chunks)
   - `{essay_draft}` = bisheriger Draft (leer bei step1)
6. **LLM Call**
   - DeepSeek Chat (oder configured chat model)
   - `max_tokens` so wählen, dass 200–300 Token Output pro Step möglich sind (z.B. 450–600)
   - Retry/Completion-Schutz: Satzende erzwingen (wie in bestehenden Graphen)
7. **Akkumulation**
   - `essay_draft = essay_draft + "\n\n" + step_text`

Nach Step 7:
- `final_text = essay_draft`
- zurück an API caller

---

## Finetune Pipeline (ragrun)

### Endpoint

- `POST /api/v1/agent/philo-von-freisinn/graphs/essay-finetune`

### Request (JSON)

```json
{
  "assistant": "philo-von-freisinn",
  "essay_slug": "ich-kann-gegen-andere-gewinnen",
  "essay_title": "ich-kann-gegen-andere-gewinnen",
  "essay_text": "...",
  "instruction": "Bitte kürze um ~20% und mache die Praxis-Sektion konkreter.",
  "k": 5,
  "verbose": false
}
```

### Response (JSON)

```json
{
  "assistant": "philo-von-freisinn",
  "essay_slug": "ich-kann-gegen-andere-gewinnen",
  "essay_title": "ich-kann-gegen-andere-gewinnen",
  "revised_text": "..."
}
```

### Graph/Chain Logik

- Dense Retrieval (Qdrant) aus primary_books (`chunk_type=book`) mit Query = `{instruction}` und `k=5`
- Prompt:
  - System: `assistants_root/<assistant>/prompts/essays/instruction.prompt`
  - User: `assistants_root/<assistant>/prompts/essays/finetune.prompt`
- Variablen:
  - `{essay_title}`, `{instruction}`, `{context}`, `{essay_text}`
- Output:
  - `revised_text` = kompletter überarbeiteter Essay (kein Change-Log)

---

## Prompt-Spezifikation (ragkeep)

Alle Step-Prompts müssen **zusätzlich** den Retrieval-Kontext akzeptieren:
- `{context}` muss im Template vorkommen (und als Grundlage erlaubt sein).

Finetune-Prompt:
- Datei: `prompts/essays/finetune.prompt`
- Muss `{essay_title}`, `{instruction}`, `{context}`, `{essay_text}` akzeptieren.

Konvention:
- Step 1: erzeugt Einleitungsteil (200–300 Token)
- Step 2..7: schreiben jeweils den nächsten Abschnitt als Fortsetzung von `{essay_draft}`

Wichtig:
- Keine Quellen erfinden.
- Kontext ist **nur** Unterstützung/Anregung; keine direkten Zitate erfinden.

---

## Qdrant Details

- Collection: aus Assistant (`rag-collection`) z.B. `philo-von-freisinn`
- Dense Retrieval:
  - `vector = embed(pitch_step_text)`
  - `POST /collections/<collection>/points/search`
  - Filter: `chunk_type == "book"`

---

## Fehlerfälle / Validierung

- pitch_steps != 7 → 400
- assistant nicht supported / prompt files fehlen → 400/502 mit klarer message
- Qdrant leer/keine Hits → trotzdem weiterlaufen, aber `{context}` leer (oder optional warn in verbose)
- LLM leer/unvollständig → Retry + completion pass (bestehendes Muster)
