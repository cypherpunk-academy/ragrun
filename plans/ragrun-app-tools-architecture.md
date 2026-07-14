# ragrun — App-Chat-Tools (Architektur)

**Status:** Entwurf  
**Bezug:** [filo-implementation-plan.md](../../ragapp/plans/filo-implementation-plan.md) (Umsetzungsreihenfolge) · **[filo-arbeitstext-contract.md](../../ragapp/plans/filo-arbeitstext-contract.md)** (Verträge ragapp↔ragrun) · [filo-chat-ui-design.md](../../ragapp/plans/filo-chat-ui-design.md) (UX) · [ASSISTANTS_CHAT_PLAN_V2.md](./ASSISTANTS_CHAT_PLAN_V2.md) · ragprep `src/cli/commands/`  
**Ziel:** Einheitliche, erweiterbare Tool-Architektur für den App-Chat (`POST /app/chat/stream`), analog zu ragprep-Befehlen: **ein Verzeichnis pro Tool**, zentrale Registry, klare Grenze zwischen Server- und Client-Ausführung.

---

## 1. Problem

Heute existieren in ragrun **drei parallele Chat-Pfade** — Ziel Filo: **App-Chat an `assistant_chat_graph` andocken** (§7.1 [filo-chat-ui-design.md](../../ragapp/plans/filo-chat-ui-design.md)), App-Tools ergänzen RAG **nicht** ersetzen:

| Pfad | Datei | RAG / Tools |
|---|---|---|
| Action-Prompt (ragprep CLI) | `action_prompt_service.py` | YAML-Manifeste in `ragrun-personalities/` |
| LangGraph SSE (Agent) | `assistant_chat_graph.py` | **Voller RAG** — Intent, Retrieval, Citations; Checkpoint `thread_id` |
| App-Chat (ragapp) | `app_chat_service.py` → **`app_chat_stream_service.py`** | **Migration:** Graph-Kern wie Agent + App-Adapter → `rag_talks`/`rag_turns`/`rag_references`; **App-Tools** (Arbeitstexte) zusätzlich |

Der abgespeckte Ist-Pfad (`app_search`×4 in `send_app_chat()`) wird **nicht** weiter ausgebaut.

Für Filo brauchen wir **App-Tools** (`create_document`, `update_document`, …) **zusätzlich** zum RAG-Graph — nicht als Ersatz für Retrieval. Die Tool-Registry darf nicht in `app_chat_service.py` monolithisch wachsen; sie hängt am **App-Adapter** (`app_chat_stream_service.py`).

**Vorbild ragprep:**

```
src/cli/commands/<commandName>/
  index.ts       # registerX(program)
  action.ts      # Implementierung
src/cli/index.ts # registerCommands()
```

**Ziel ragrun (App-Tools):**

```
app/tools/app/<tool_id>/
  tool-manifest.yaml
  handler.py       # optional bei execution: server
  schema.json      # LLM function-calling / structured output
app/tools/registry.py
app/tools/index.py # register_app_tools()
```

---

## 2. Grundprinzipien

### 2.1 Zwei Ausführungsorte

| `execution` | Wer schreibt persistente Daten? | Beispiel |
|---|---|---|
| **`client`** | ragapp (WatermelonDB → Supabase-Sync) | `create_document`, `update_document` |
| **`server`** | ragrun (Postgres `rag_*`) | `summarize_talk` (bereits API), später `compress_talk` |

**Regel:** Alles in `app_notes` / nutzer-lokalem Korpus-Spiegel → **`execution: client`**. ragrun liefert nur strukturierte **Vorschläge** im SSE-`done`-Event; ragapp materialisiert (lokal-first, siehe `NOTIZEN_ANALYSE.md`).

### 2.2 Tool ≠ Personality-Action

| Familie | Ort | Zweck | Discovery |
|---|---|---|---|
| **Personality-Actions** | `ragrun-personalities/<id>/` | RAG-Prompts für Assistenten-Chat (ragprep `rag:chat`) | `list_actions()` |
| **Helper-Actions** | `app/retrieval/helper-actions/` | Kurzaktionen ohne Retrieval | `list_actions()` |
| **App-Tools** | `app/tools/app/<id>/` | Filo-Chat in ragapp (`/app/chat/stream`) | `list_app_tools()` |

App-Tools werden **nicht** in die Action-Prompt-Dropdown-Liste von ragprep gemischt. Sie hängen am App-Chat-Graph / Streaming-Endpunkt.

### 2.3 Automatische Client-Materialisierung

Wenn ein Tool `execution: client` hat und im `done`-Event ein Vorschlag mitkommt, wendet ragapp den Patch **automatisch** an — kein Bestätigungs-Chip (Entscheidung Filo-Design, Juli 2026).

### 2.4 Arbeitsdokumente: Baum statt Blob (MVP-Entscheidung, Juli 2026)

**Keine Markdown-Tabellen im MVP.** Arbeitsdokumente nutzen nur `#` / `##` / `###` plus Absätze und Listen. Tabellen (z. B. in der Doppelmatrix-Vorlage) werden vor Filo-Nutzung in `###`-Unterkapitel mit nummerierten Listen konvertiert — oder der Editor zeigt einen Hinweis „Tabellen noch nicht unterstützt".

Statt Rohtext-Patches (`replace_all`, Tabellenzeilen) arbeitet das System mit einem **Document Tree**:

- **Outline** (Überschriften + Absatz-Previews) geht **immer** mit dem Chat-Request — wenige hundert Tokens statt ganzes Werk.
- **`read_blocks`** (optional): voller Text einzelner Absätze/Abschnitte nach Adresse.
- **`update_document`**: gezielte Änderungen an **Überschrift**, **Kapitel/Abschnitt** oder **einzelnem Absatz** per `paragraph_id`.

DeepSeek braucht die Intelligenz; das Tool-Protokoll liefert **Adressen**, die der Client deterministisch patcht (`parse → patch → serialize`).

---

## 3. Verzeichnisstruktur

```
app/tools/
├── __init__.py
├── registry.py              # AppToolRegistry, discover, get_schema_bundle
├── types.py                 # ToolContext, ToolResult, ToolManifest
├── index.py                 # register_all_tools() — beim App-Start
│
├── app/                     # App-Chat-Tools (Filo / ragapp)
│   ├── create_document/
│   │   ├── tool-manifest.yaml
│   │   ├── schema.json      # OpenAI-style function parameters
│   │   └── handler.py       # build_suggestion(ctx, args) -> dict
│   ├── update_document/
│   │   ├── tool-manifest.yaml
│   │   ├── schema.json
│   │   └── handler.py
│   ├── read_blocks/         # optional: volle Absätze/Abschnitte lesen
│   │   ├── tool-manifest.yaml
│   │   ├── schema.json
│   │   └── handler.py
│   ├── create_post_draft/   # später: Modus „Post schreiben"
│   └── search_corpus/       # später: explizite Korpus-Suche als Tool
│
└── README.md                # Konventionen (dieses Dokument, gekürzt)
```

**Spiegel in ragapp** (nur Client-Materialisierung):

```
ragapp/src/data/lib/
├── documentTree.ts          # parseDocument, serializeDocument, buildOutline
ragapp/src/data/tools/
├── index.ts                 # dispatchToolEffects(doneEvent)
├── applyDocumentUpdate.ts   # patch auf DocumentTree
├── materializeDocument.ts
└── types.ts                 # Spiegel der SSE-Payload-Typen
```

---

## 4. Tool-Manifest (`tool-manifest.yaml`)

Analog zu `action-manifest.yaml`, aber für App-Tools:

```yaml
id: update_document
label: Arbeitsdokument aktualisieren
description: >
  Ändert ein verknüpftes Arbeitsdokument gezielt: einzelner Absatz,
  Abschnitt unter einer Überschrift, oder Überschrift umbenennen.
  Keine Tabellen — nur # / ## / ### und Absätze.
category: app-document
execution: client

availability:
  requires_linked_document: true
  modes: [chat, nachdenken]

result_key: suggested_document_update

tests:
  - cases/update_paragraph.yaml
  - cases/update_section.yaml
```

### Pflichtfelder

| Feld | Typ | Bedeutung |
|---|---|---|
| `id` | string | Verzeichnisname, snake_case |
| `execution` | `client` \| `server` | Wo persistiert wird |
| `result_key` | string | Schlüssel im SSE-`done`-Payload |
| `schema.json` | Datei | Parameter-Schema fürs Modell |

---

## 5. Handler-Vertrag (`handler.py`)

Jedes Tool mit Logik exportiert:

```python
# app/tools/app/update_document/handler.py
from app.tools.types import ToolContext, ToolResult

TOOL_ID = "update_document"

async def run(ctx: ToolContext, args: dict) -> ToolResult:
    """
    ctx: user_id, talk_id, mode, linked_document_id, document_content,
         messages, personality, ...
    args: vom Modell (gemäß schema.json)
    """
    return ToolResult(
        result_key="suggested_document_update",
        payload={
            "document_id": ctx.linked_document_id,
            "operation": args["operation"],  # update_paragraph | update_section | …
            "paragraph_id": args.get("paragraph_id"),
            "heading_path": args.get("heading_path"),  # z. B. ["## Kapitel 1", "### Feld 1"]
            "content": args["content"],
            "summary_for_chat": "Absatz zur Pathologie in Kapitel 1 überarbeitet.",
        },
    )
```

**Kein DB-Zugriff** in `execution: client`-Handlern auf `app_notes`.

Registry lädt beim Start alle `app/tools/app/*/tool-manifest.yaml` und bindet optional `handler.py`.

---

## 5b. Document Tree (Arbeitsdokumente)

**Single Source of Truth:** [filo-arbeitstext-contract.md](../../ragapp/plans/filo-arbeitstext-contract.md) §2 (Parsing, Adressen, Limits, Outline-Schema) und §3 (`linked_document_content`).

**ragrun:** liest `document_outline` + `linked_document_content` aus dem Request-Body in `ToolContext`; schreibt **nicht** in `app_notes`.

**ragapp:** `documentTree.ts` — parse, `buildOutline`, serialize, patch.

**Härtetest Doppelmatrix** (Referenzfälle): Contract §2 + Filo-Plan §5.3; Fixture ohne Tabellen in T1-Tests.

---

## 6. Registry (`app/tools/registry.py`)

Siehe auch §7. Discovery: Glob `app/tools/app/*/tool-manifest.yaml`. Registrierung: `app/main.py` → `app.state.app_tool_registry`.

---

## 7. Einbindung in `/app/chat/stream`

**Ablauf, Request-Body, SSE-`done`-Format:** [filo-arbeitstext-contract.md](../../ragapp/plans/filo-arbeitstext-contract.md) §3, §5–6.

**ragrun-spezifisch:** `app_chat_stream_service.py` orchestriert Graph-Lauf, Tool-Loop (`AppToolRegistry`), Persistenz. Registry-API:

```python
class AppToolRegistry:
    def list_tools(self, *, mode: str, linked_document_id: str | None) -> list[ToolSpec]: ...
    def schemas_for_llm(self, available: list[ToolSpec]) -> list[dict]: ...
    async def invoke(self, tool_id: str, ctx: ToolContext, args: dict) -> ToolResult: ...
```

`ToolContext` enthält u. a. `linked_document_id`, `document_content` (= `linked_document_content` aus Request), `mode`, `talk_id`.

**Discovery:** Glob `app/tools/app/*/tool-manifest.yaml`. **Registrierung:** `app/main.py` → `app.state.app_tool_registry`.

---

## 8. Erste App-Tools (MVP Filo)

Tool-IDs, Verfügbarkeit, Args/Payloads, Operationen: **[filo-arbeitstext-contract.md](../../ragapp/plans/filo-arbeitstext-contract.md) §4**.

Implementierung: je ein Verzeichnis unter `app/tools/app/<tool_id>/` mit `tool-manifest.yaml`, `schema.json`, `handler.py`.

---

## 9. Geplante weitere Tools (nach MVP)

| Tool-ID | execution | Beschreibung |
|---|---|---|
| `create_post_draft` | client | Modus „Post schreiben" — Facebook-Kommentar-Stil (post-MVP; im MVP per Chat/Arbeitsdokument) |
| `compress_talk` | server | Verdichtung alter Turns → `compressed_up_to_turn_index` |
| `search_corpus` | server | Explizite Hybrid-Suche, Ergebnis in Turn-Metadaten |
| `cite_paragraph` | client | Verweis in Arbeitsdokument einfügen |
| `split_document` | client | Ein Dokument in zwei teilen |
| `replace_table_row` | client | Tabellen-Support (post-MVP) |

Jedes Tool = neues Verzeichnis unter `app/tools/app/`, kein Edit in Monolithen.

---

## 10. Verhältnis zu Personality-Actions (Migration, optional)

Langfristig können Personality-Actions dasselbe Muster übernehmen:

```
app/tools/personality/<action_id>/   # oder weiter extern in ragrun-personalities/
```

`action_prompt_service.py` wird dann **Orchestrator** (Retrieval + Prompt-Fill), nicht Spezialfall-Sammlung. **Nicht Teil des Filo-MVP** — nur gleiche Konvention.

---

## 11. Tests

Pro Tool-Verzeichnis:

```
app/tools/app/update_document/
  tests/
    cases/
      update_section.yaml    # args + input markdown + expected patch
    test_handler.py
```

CI: `pytest app/tools/app/*/tests/` — analog ragprep `tests/cli/commands/`.

---

## 12. Phasenplan (ragrun-Seite)

**Mapping zu Filo-Phasen:** [filo-arbeitstext-contract.md](../../ragapp/plans/filo-arbeitstext-contract.md) §7.

### Phase T0 — Gerüst
- [ ] `app/tools/types.py`, `registry.py`, `index.py`
- [ ] README in `app/tools/`
- [ ] Startup-Registrierung in `main.py`

### Phase T1 — MVP-Tools
- [ ] `documentTree.ts` in ragapp (parse, outline, serialize, patch)
- [ ] `create_document/`
- [ ] `read_blocks/`
- [ ] `update_document/` (Operationen § 8)
- [ ] Unit-Tests mit Doppelmatrix-Fixture (**ohne** Tabellen, konvertierte Version)

### Phase T2 — Stream-Integration
- [ ] `app_chat_stream_service.py` (neu oder Erweiterung)
- [ ] `POST /app/chat/stream` mit Tool-Loop + `tool_results` im `done`-Event (Contract §5–6)
- [ ] System-Prompt-Baustein „Arbeitsdokument-Editor" wenn `linked_document_id` gesetzt

(Pin/Cleanup: Filo Phase C — nicht Teil des Tools-Plans.)

### Phase T3 — Weitere Tools
- [ ] `compress_talk` (server)
- [ ] `create_post_draft` (post-MVP)

---

## 13. Offene Entscheidungen

1. **Tool-Loop:** **MVP: max. 2 Runden** — Contract §1.
2. **LLM-API:** DeepSeek function calling vs. structured JSON — Contract §8.1; vor T2 klären.
3. **paragraph_id-Stabilität:** Positions-IDs nach jedem Patch neu — akzeptiert, weil Outline jedes Mal frisch mitgeschickt wird.
4. **Naming:** Verzeichnis `app/tools/app/` — **festgelegt**.
5. **Personality-Actions refactoren:** erst nach App-Tool-Muster bewährt.
6. **Tabellen:** bewusst post-MVP; Doppelmatrix-Vorlage in ragkeep bleibt mit Tabellen, App-Arbeitskopie ohne.
7. **Dokument-Maximalgröße:** **50 000 Zeichen** — [filo-arbeitstext-contract.md](../../ragapp/plans/filo-arbeitstext-contract.md) §2.3.
8. **Chat-Retention:** Pin + 7-Tage-Cleanup — Filo-Plan §2, Phase C (nicht Tools).

---

## 14. Referenzen

- **Contract (ragapp↔ragrun):** [filo-arbeitstext-contract.md](../../ragapp/plans/filo-arbeitstext-contract.md)
- ragprep Registry: `ragprep/src/cli/index.ts`
- Personality-Actions: `ragrun-personalities/*/action-manifest.yaml`
- App-Chat heute: `app/services/app_chat_service.py`, `app/api/app_api.py`
- Filo UX: `ragapp/plans/filo-chat-ui-design.md` §5–6
- Architekturgrenze `app_notes`: `ragapp/plans/NOTIZEN_ANALYSE.md` §2
