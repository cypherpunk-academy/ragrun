# Implementierungsplan: Assistenten-Chat (Philo von Freisinn)

> Dieser Plan basiert auf der Analyse in `ASSISTANT_CHAT_ANALYSE_PHILO_VON_FREISINN.md`.
> Architekturentscheidung: **LangGraph (Variante B)** von Anfang an.
> Referenz-Spike: `spike/` (lauffaehig, alle Kernfragen beantwortet).

---

## Voraussetzungen

```bash
# In requirements.txt hinzufuegen (ragrun):
langgraph>=0.2
langgraph-checkpoint-postgres>=2.0
langchain-openai>=0.2
langchain-core>=0.3
sse-starlette>=2.0
```

Alle anderen Abhaengigkeiten (`qdrant-client`, `sqlalchemy`, `asyncpg`, `fastapi`) sind
bereits vorhanden.

---

## Iteration 1 – Funktionaler Chat (Ziel: E2E-Chat ohne UI)

**Ergebnis:** `curl`-testbarer SSE-Endpunkt mit Intent-Klassifikation, Qdrant-Retrieval
und Philo-Persona. History laeuft per `MemorySaver` im RAM.

---

### Schritt 1: `intents.py` anlegen

**Datei:** `app/retrieval/graphs/intents.py` *(neu)*

Alle Intent-Konstanten an einem Ort. Keine Logik.

**Iteration 1 verwendet 5 Intents** – robuster, weniger Fehlklassifikationen.
Die feingranularen Intents kommen erst in Iteration 2, wenn echte Chat-Logs zeigen
wo die Qualität schwächelt (siehe "Gestrichene Intents" am Ende von Schritt 1).

```python
INTENT_LABELS: list[str] = [
    "begriff_definieren",   # Begriff nachschlagen (mit Lemma-Lookup, siehe Schritt 3e)
    "quelle_suchen",        # Zitat oder Belegstelle finden
    "erklaerung",           # Breite Erklärung, Vergleich, Vertiefung, Zusammenfassung
    "skip",                 # Kein Retrieval nötig (Gruss, Meta-Frage, Off-Topic)
    "sonstiges",            # Fallback: breites Retrieval ohne chunk_type-Filter
]

INTENT_CHUNK_TYPE_MAP: dict[str, list[str]] = {
    "begriff_definieren": ["begriff_list", "explanation"],
    "quelle_suchen":      ["quote", "book", "talk"],
    "erklaerung":         ["book", "talk", "chapter_summary", "essay", "secondary_book"],
    "sonstiges":          [],   # leere Liste → kein chunk_type-Filter im Retrieval
}

SKIP_RETRIEVAL_INTENTS: frozenset[str] = frozenset({"skip"})
```

**Gestrichene Intents (Iteration 2+ – erst einführen wenn Chat-Logs zeigen dass
die Unterscheidung nötig ist):**

| Gestrichener Intent | Gemappt auf (Iter. 1) | Grund fuer spaeteren Einstieg |
|---|---|---|
| `werk_lokalisieren` | `erklaerung` | chunk_types überlappen stark |
| `zitat_suchen` | `quelle_suchen` | Selber Retrieval-Pfad |
| `vergleich` | `erklaerung` | Breites Retrieval deckt das ab |
| `zusammenfassung` | `erklaerung` | `chapter_summary`-Filter schon in `erklaerung` |
| `beleg_pruefung` | `quelle_suchen` | Selber Retrieval-Pfad |
| `follow_up` | `sonstiges` | Komplexe Logik (Vorturn-Referenz) erst ab Iter. 3 |
| `hypothetisch` | `erklaerung` | Seltener Fall, deckt sich mit `erklaerung` |
| `meta_assistent` | `skip` | Teil der Skip-Gruppe |
| `konversationell` | `skip` | Dto. |
| `out_of_scope` | `skip` | Dto. |

**Test:** `python -c "from app.retrieval.graphs.intents import INTENT_LABELS; print(INTENT_LABELS)"` → kein Import-Fehler.

---

### Schritt 2: Intent-Classify-Prompt anlegen

**Datei:** `app/retrieval/prompts/intent_classify.prompt` *(neu)*

Der Prompt muss explizit verlangen, dass die Antwort ein JSON-Objekt ist (DeepSeek
`json_mode`-Anforderung – bewaehrt im Spike).

Inhalt:

```
Du bist ein Intent-Klassifikator fuer einen philosophischen Assistenten (Rudolf Steiner / Anthroposophie).
Klassifiziere die folgende Nutzeranfrage in GENAU einen der erlaubten Intents.

Erlaubte Intent-Labels und ihre Bedeutung:
- "begriff_definieren"  → Der Nutzer fragt nach der Bedeutung oder Definition eines Begriffs
- "quelle_suchen"       → Der Nutzer sucht ein konkretes Zitat oder eine Belegstelle
- "erklaerung"          → Der Nutzer moechte etwas erklaert, vertieft, verglichen oder zusammengefasst haben
- "skip"                → Gruss, Dank, Meta-Frage ueber den Assistenten, oder komplett themenfremde Anfrage
- "sonstiges"           → Passt in keine der obigen Kategorien

Antworte ausschliesslich als JSON-Objekt mit diesen Feldern:
- "intent": string (exakt eines der fuenf Labels oben)
- "confidence": float (0.0 bis 1.0)
- "reasoning": string (1-2 Saetze)
- "lemma": string (nur bei intent="begriff_definieren": der gesuchte Begriff als Lemma, sonst "")

Kontext der bisherigen Unterhaltung (falls vorhanden):
{conversation_context}

Nutzeranfrage:
{user_message}
```

**Verwendung im Node:** Keine `{intent_labels}`-Interpolation mehr nötig – die Labels
sind direkt im Prompt erklärt (stabiler, weniger Prompt-Engineering).
`conversation_context` sind die letzten 3 HumanMessage-Texte.
`lemma` ersetzt das bisherige zweistufige Extrahieren des Begriffs im Node (einfacher).

---

### Schritt 3: `assistant_chat_graph.py` anlegen

**Datei:** `app/retrieval/graphs/assistant_chat_graph.py` *(neu)*

Diese Datei enthaelt alles: State, Nodes, Edges, Graph-Aufbau.
Grobe Struktur (vollstaendig in dieser Reihenfolge implementieren):

#### 3a. Imports und LLM-Factory

```python
import json
from contextlib import asynccontextmanager
from typing import Annotated, Any
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from app.config import settings
from app.retrieval.graphs.intents import (
    INTENT_CHUNK_TYPE_MAP,
    INTENT_LABELS,
    SKIP_RETRIEVAL_INTENTS,
)
from app.retrieval.prompts.philo_von_freisinn import load_system_prompt
from app.retrieval.utils.retrievers import (
    hybrid_retrieve,
    build_context,
    payload_filter,
)

def _make_llm(streaming: bool = False) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.deepseek_chat_model,
        openai_api_key=settings.deepseek_api_key,
        openai_api_base=f"{str(settings.deepseek_base_url).rstrip('/')}/",
        temperature=0.3,
        max_tokens=800,
        streaming=streaming,
    )
```

#### 3b. ChatState (TypedDict)

```python
class ChatState(TypedDict):
    assistant_slug: str
    collection_name: str

    user_message: str
    messages: Annotated[list[BaseMessage], add_messages]

    intent: str
    intent_confidence: float
    extracted_lemma: str        # fuer begriff_definieren
    lemma_found: bool

    retrieval_plan: list[str]   # chunk_types
    context_text: str
    context_refs: list[str]     # chunk_ids
    retrieval_mode: str
    sufficiency: str            # "high" / "medium" / "low" / "insufficient"
    retry_count: int

    citations: list[dict]
    final_response: str
    confidence_score: float
```

#### 3c. IntentResult Pydantic-Modell

```python
class IntentResult(BaseModel):
    intent: str
    confidence: float
    reasoning: str
    lemma: str = ""   # nur bei "begriff_definieren" befuellt; sonst leer
```

`lemma` kommt direkt aus dem Prompt (Schritt 2) – kein nachträgliches Heuristik-Parsing nötig.

#### 3d. Node 1: `classify_intent`

- Laedt `intent_classify.prompt` und interpoliert `conversation_context` (letzte 3 Human-Turns)
- Ruft `_make_llm().with_structured_output(IntentResult, method="json_mode")` auf
- Returns: `{"intent": result.intent, "intent_confidence": result.confidence, "extracted_lemma": result.lemma}`

#### 3e. Node 1b: `lemma_lookup`

Wird nur aufgerufen wenn `intent == "begriff_definieren"`.

**Wichtig:** `segment_title` in der DB ist immer ein normalisiertes Kleinbuchstaben-Lemma
ohne Artikel (z.B. `"ich"`, `"seele"`, `"geist"`). Der LLM liefert aber ggf. `"das Ich"`
oder `"Ich-Begriff"`. Daher muss das Lemma vor dem Query normalisiert werden.

**Normalisierungsfunktion** (in `assistant_chat_graph.py` definieren, kein Import nötig):

```python
import re

_ARTICLE_PREFIX = re.compile(
    r'^(der|die|das|ein|eine|des|dem|den|einem|einer)\s+',
    re.IGNORECASE,
)

def _normalize_lemma(raw: str) -> str:
    """
    Normalisiert ein rohes LLM-Lemma auf das DB-Format.

    Beispiele:
        "das Ich"           → "ich"
        "Ich-Begriff"       → "ich-begriff"
        "menschliche Seele" → "seele"   (letztes Wort = Kern-Nomen)
        "Geist"             → "geist"
    """
    cleaned = _ARTICLE_PREFIX.sub("", raw.strip()).lower()
    parts = cleaned.split()
    return parts[-1] if parts else cleaned
```

**Prompt-Hinweis (Schritt 2, `lemma`-Feld):** Der Prompt fordert bereits ein normalisiertes
Lemma – `_normalize_lemma` ist die defensive Absicherung falls der LLM trotzdem einen
Artikel mitliefert.

**Node-Implementierung:**

```python
async def lemma_lookup(state: ChatState, config: RunnableConfig) -> dict:
    raw_lemma = state.get("extracted_lemma", "")
    lemma = _normalize_lemma(raw_lemma)
    if not lemma:
        return {"lemma_found": False}

    async with get_session() as session:
        row = await session.execute(
            text(
                "SELECT chunk_id FROM rag_chunks "
                "WHERE collection = :col "
                "  AND metadata->>'segment_title' = :lemma "
                "  AND chunk_type = 'begriff_list' "
                "LIMIT 1"
            ),
            {"col": state["collection_name"], "lemma": lemma},
        )
        found = row.fetchone() is not None

    return {"lemma_found": found}
```

Hinweis: Kein `ILIKE` mehr nötig – nach Normalisierung sind beide Seiten Kleinbuchstaben,
ein einfacher `=`-Vergleich ist effizienter und eindeutiger.

#### 3f. Node 2: `route_retrieval_plan`

Reine Python-Funktion (kein LLM):

```python
async def route_retrieval_plan(state: ChatState, config: RunnableConfig) -> dict:
    plan = INTENT_CHUNK_TYPE_MAP.get(state["intent"], [])
    # "sonstiges" oder unbekannter Intent → leere Liste = kein chunk_type-Filter
    return {"retrieval_plan": plan}
```

#### 3g. Node 3: `retrieve_chunks`

- Ruft `hybrid_retrieve(query, collection, k=10)` auf (aus `retrievers.py`)
- Wendet `payload_filter(chunk_types=state["retrieval_plan"])` an –
  bei leerem `retrieval_plan` (Intent `sonstiges`) wird **kein Filter** angewendet
- Falls weniger als 2 Treffer: Widen-Retry ohne chunk_type-Filter (wie `_retrieve_with_widen`)
- Ruft `build_context(chunks)` auf → `context_text`, `context_refs`
- Berechnet `sufficiency` aus Anzahl und Score der Treffer:
  - >= 3 Treffer mit Score > 0.7 → "high"
  - 1-2 Treffer oder Score 0.5-0.7 → "medium"
  - Treffer aber Score < 0.5 → "low"
  - 0 Treffer → "insufficient"
- Returns: `{"context_text", "context_refs", "retrieval_mode", "sufficiency"}`

#### 3h. Node 4: `compose_answer`

```python
async def compose_answer(state: ChatState, config: RunnableConfig) -> dict:
    persona = load_system_prompt()
    llm = _make_llm(streaming=True)
    messages_in = [
        SystemMessage(persona),
        SystemMessage(f"Quellen-Kontext:\n{state['context_text']}"),
        *state["messages"][-6:],       # letzte 3 Turns (6 Messages: H+A)
        HumanMessage(state["user_message"]),
    ]
    response = ""
    async for chunk in llm.astream(messages_in, config):
        response += chunk.content
    return {"messages": [AIMessage(content=response)]}
```

#### 3i. Node 5: `attach_citations`

- Laed Chunk-Metadaten per SQL: `SELECT chunk_id, metadata FROM rag_chunks WHERE chunk_id = ANY(:ids)`
- Extraiert `source_title`, `segment_title`, `segment_index`, `lecture_date` aus `metadata` JSONB
- Returns: `{"citations": [...]}`

#### 3j. Node 6: `finalize`

```python
async def finalize(state: ChatState, config: RunnableConfig) -> dict:
    last_ai = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
        None,
    )
    response = last_ai.content if last_ai else "Keine Antwort verfuegbar."

    if state.get("sufficiency") == "insufficient":
        response = (
            "Zu diesem Thema finde ich in meinen Quellen keinen ausreichenden Beleg. "
            + response
        )
    return {
        "final_response": response,
        "confidence_score": state.get("intent_confidence", 0.0),
    }
```

#### 3k. Edges und Routing-Funktionen

```python
def route_after_intent(state: ChatState) -> str:
    if state["intent"] in SKIP_RETRIEVAL_INTENTS:
        return "finalize"
    if state["intent"] == "begriff_definieren":
        return "lemma_lookup"
    return "route_retrieval_plan"

def route_after_lemma(state: ChatState) -> str:
    if state["lemma_found"]:
        return "retrieve_chunks"
    return "route_retrieval_plan"   # konzept_explain via breiteres Retrieval

def route_after_retrieval(state: ChatState) -> str:
    if state["sufficiency"] == "insufficient" and state.get("retry_count", 0) < 2:
        return "retrieve_chunks"    # Widen-Retry (retry_count wird in retrieve_chunks erhoehen)
    return "compose_answer"
```

#### 3l. Graph-Aufbau

```python
def build_chat_graph(checkpointer=None):
    builder = StateGraph(ChatState)

    builder.add_node("classify_intent",     classify_intent)
    builder.add_node("lemma_lookup",        lemma_lookup)
    builder.add_node("route_retrieval_plan",route_retrieval_plan)
    builder.add_node("retrieve_chunks",     retrieve_chunks)
    builder.add_node("compose_answer",      compose_answer)
    builder.add_node("attach_citations",    attach_citations)
    builder.add_node("finalize",            finalize)

    builder.set_entry_point("classify_intent")

    builder.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {"finalize": "finalize", "lemma_lookup": "lemma_lookup",
         "route_retrieval_plan": "route_retrieval_plan"},
    )
    builder.add_conditional_edges(
        "lemma_lookup",
        route_after_lemma,
        {"retrieve_chunks": "retrieve_chunks",
         "route_retrieval_plan": "route_retrieval_plan"},
    )
    builder.add_edge("route_retrieval_plan", "retrieve_chunks")
    builder.add_conditional_edges(
        "retrieve_chunks",
        route_after_retrieval,
        {"retrieve_chunks": "retrieve_chunks", "compose_answer": "compose_answer"},
    )
    builder.add_edge("compose_answer", "attach_citations")
    builder.add_edge("attach_citations", "finalize")
    builder.add_edge("finalize", END)

    graph = builder.compile(checkpointer=checkpointer or MemorySaver())

    # Docstring mit Mermaid-Diagramm:
    # print(graph.get_graph().draw_mermaid())
    return graph
```

**Mermaid-Diagramm:** Einmalig `graph.get_graph().draw_mermaid()` ausfuehren und als
Docstring oben in die Datei einfuegen.

**Test Schritt 3:**

```bash
python -c "
from app.retrieval.graphs.assistant_chat_graph import build_chat_graph
g = build_chat_graph()
print(g.get_graph().draw_mermaid())
"
```

Erwartung: Kein Import-Fehler, Mermaid-Text wird ausgegeben.

---

### Schritt 4: `app/api/chat.py` anlegen

**Datei:** `app/api/chat.py` *(neu)*

```python
import json
import uuid
from fastapi import APIRouter, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

router = APIRouter(prefix="/api/v1/agent", tags=["chat"])

class ChatRequest(BaseModel):
    thread_id: str | None = None
    user_message: str

class ChatResponse(BaseModel):
    thread_id: str
    response: str
    citations: list[dict]
    confidence_score: float
    intent: str
    sufficiency: str

@router.post("/{assistant_slug}/chat/stream")
async def chat_stream(
    assistant_slug: str,
    body: ChatRequest,
    request: Request,
):
    """SSE-Streaming Endpunkt. Sendet Token-Events, dann Citations, dann Done."""
    graph = request.app.state.chat_graph
    thread_id = body.thread_id or str(uuid.uuid4())
    collection_name = f"{assistant_slug}-de"   # Konvention; ggf. aus DB laden

    initial_state = {
        "assistant_slug": assistant_slug,
        "collection_name": collection_name,
        "user_message": body.user_message,
        "messages": [],
        "retry_count": 0,
        "lemma_found": False,
        "extracted_lemma": "",
    }
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator():
        yield {"data": json.dumps({"type": "thread_id", "thread_id": thread_id})}
        async for event in graph.astream_events(initial_state, config, version="v2"):
            if event["event"] == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                if token:
                    yield {"data": json.dumps({"type": "token", "content": token})}
            elif event["event"] == "on_chain_end" and event["name"] == "finalize":
                output = event["data"].get("output", {})
                yield {
                    "data": json.dumps({
                        "type": "done",
                        "citations": output.get("citations", []),
                        "confidence_score": output.get("confidence_score", 0.0),
                        "intent": output.get("intent", ""),
                        "sufficiency": output.get("sufficiency", ""),
                    })
                }

    return EventSourceResponse(event_generator())

@router.get("/{assistant_slug}/chat/thread/{thread_id}")
async def get_thread(assistant_slug: str, thread_id: str, request: Request):
    """Liest den gespeicherten Thread-State (History)."""
    graph = request.app.state.chat_graph
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    messages = [
        {"role": "human" if isinstance(m, HumanMessage) else "assistant",
         "content": m.content}
        for m in snapshot.values.get("messages", [])
    ]
    return {"thread_id": thread_id, "messages": messages}
```

---

### Schritt 5: `main.py` – Graph im lifespan registrieren

**Datei:** `app/main.py` *(modifizieren)*

Im bestehenden `lifespan`-Context-Manager ergaenzen:

```python
from langgraph.checkpoint.memory import MemorySaver
from app.retrieval.graphs.assistant_chat_graph import build_chat_graph
from app.api.chat import router as chat_router

# Im lifespan:
checkpointer = MemorySaver()   # Iter. 1: RAM; Iter. 2: AsyncPostgresSaver
app.state.chat_graph = build_chat_graph(checkpointer=checkpointer)

# Router einbinden (ausserhalb lifespan):
app.include_router(chat_router)
```

---

### Schritt 6: Manueller E2E-Test

```bash
# Server starten
uvicorn app.main:app --reload

# Thread starten (SSE-Stream)
curl -N -X POST http://localhost:8000/api/v1/agent/philo-von-freisinn/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"user_message": "Was ist der Begriff Ich im Sinne Steiners?"}'

# Erwartete Events:
# data: {"type":"thread_id","thread_id":"..."}
# data: {"type":"token","content":"Das"}
# data: {"type":"token","content":" Ich"}
# ...
# data: {"type":"done","citations":[...],"intent":"begriff_definieren",...}

# Follow-up im selben Thread (thread_id aus erstem Response)
curl -N -X POST http://localhost:8000/api/v1/agent/philo-von-freisinn/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"<uuid-aus-oben>","user_message":"Erklaere das genauer."}'
```

Pruefkriterien:
- [ ] Token-Events kommen als SSE-Stream
- [ ] `intent` im `done`-Event ist sinnvoll (nicht `out_of_scope`)
- [ ] `citations` enthalten `source_title`
- [ ] Follow-up-Frage beantwortet "das" korrekt (Pronomen-Aufloesung via History)

---

### Schritt 7: Event-Logging einklinken

In `compose_answer` oder `finalize` den `GraphEventRecorder` verwenden:

```python
from app.retrieval.services.graph_event_recorder import GraphEventRecorder

# Am Ende von finalize:
recorder = GraphEventRecorder()
await recorder.record_event(
    event_type="chat_turn",
    metadata={
        "assistant_slug": state["assistant_slug"],
        "thread_id": config["configurable"]["thread_id"],
        "intent": state["intent"],
        "sufficiency": state["sufficiency"],
        "chunk_ids": state["context_refs"],
    },
    content={
        "user_message": state["user_message"],
        "response": state["final_response"],
        "citations": state["citations"],
    },
)
```

Zweck: Automatische Sammlung von Intent-Labels fuer spaetere Classifier-Trainingsdaten
(siehe Analyse Abschnitt 4 und 11).

---

## Iteration 2 – Qualitaet und Persistence (Ziel: Produktionsreifer Kern)

**Ergebnis:** Grounding-Check, persistente Threads via Postgres, Confidence-Score,
defensiver Antwortmodus.

---

### Schritt 8: `AsyncPostgresSaver` aktivieren

**Datei:** `app/main.py` *(modifizieren)*

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpointer = AsyncPostgresSaver.from_conn_string(str(settings.postgres_dsn))
    await checkpointer.setup()   # legt LangGraph-eigene Tabellen an (idempotent)
    app.state.chat_graph = build_chat_graph(checkpointer=checkpointer)
    yield
```

`checkpointer.setup()` legt Tabellen `checkpoints`, `checkpoint_blobs`,
`checkpoint_migrations` an – kein Alembic-Conflict.

**Test:** Server neu starten, `\dt` in Postgres pruefen → neue Tabellen sichtbar.
Nach einem Chat-Turn: `SELECT * FROM checkpoints LIMIT 5;`

---

### Schritt 9: `citation_service.py` anlegen

**Datei:** `app/retrieval/services/citation_service.py` *(neu)*

```python
from sqlalchemy import text
from app.db.session import get_session   # bestehende Session-Factory

async def resolve_citations(chunk_ids: list[str]) -> list[dict]:
    """Laedt Quellen-Metadaten fuer eine Liste von chunk_ids."""
    if not chunk_ids:
        return []
    async with get_session() as session:
        rows = await session.execute(
            text("SELECT chunk_id, metadata FROM rag_chunks WHERE chunk_id = ANY(:ids)"),
            {"ids": chunk_ids},
        )
        result = []
        for row in rows:
            meta = row.metadata or {}
            result.append({
                "chunk_id":      row.chunk_id,
                "source_title":  meta.get("source_title", ""),
                "segment_title": meta.get("segment_title", ""),
                "segment_index": meta.get("segment_index"),
                "lecture_date":  meta.get("lecture_date", ""),
            })
        return result
```

In `attach_citations`-Node einbinden (ersetzt direkte SQL dort).

---

### Schritt 10: `verify_grounding`-Node hinzufuegen

**Datei:** `app/retrieval/graphs/assistant_chat_graph.py` *(modifizieren)*

```python
class GroundingResult(BaseModel):
    is_grounded: bool
    confidence: float    # 0.0 – 1.0
    reasoning: str

async def verify_grounding(state: ChatState, config: RunnableConfig) -> dict:
    last_ai = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
        None,
    )
    if not last_ai:
        return {"confidence_score": 0.0}

    llm = _make_llm().with_structured_output(GroundingResult, method="json_mode")
    result = await llm.ainvoke([
        SystemMessage(
            "Pruefe ob die Antwort ausschliesslich aus dem Kontext ableitbar ist. "
            "Antworte als JSON mit is_grounded (bool), confidence (float), reasoning (str)."
        ),
        HumanMessage(
            f"Kontext:\n{state['context_text']}\n\nAntwort:\n{last_ai.content}"
        ),
    ], config)
    return {
        "confidence_score": result.confidence,
        "retry_count": state.get("retry_count", 0) + (0 if result.is_grounded else 1),
    }

def route_after_grounding(state: ChatState) -> str:
    if state["confidence_score"] < 0.4 and state.get("retry_count", 0) < 2:
        return "retrieve_chunks"
    return "attach_citations"
```

Graph-Anpassung: `compose_answer → verify_grounding` (statt direkt `attach_citations`),
dann konditionaler Edge von `verify_grounding`.

---

### Schritt 11: `sufficiency.py` extrahieren (optional, bei Komplexitaet)

**Datei:** `app/retrieval/utils/sufficiency.py` *(neu)*

Wenn die Sufficiency-Berechnung in `retrieve_chunks` zu umfangreich wird,
als eigenstaendige Funktion extrahieren:

```python
def compute_sufficiency(chunks: list[dict]) -> str:
    """Berechnet sufficiency aus Anzahl und Score der Retrieval-Treffer."""
    if not chunks:
        return "insufficient"
    scores = [c.get("score", 0.0) for c in chunks]
    high_quality = sum(1 for s in scores if s > 0.7)
    if high_quality >= 3:
        return "high"
    if len(chunks) >= 2 or any(s > 0.5 for s in scores):
        return "medium"
    return "low"
```

---

## Iteration 3 – Robustheit (Ziel: Multi-Intent, Memory, Quote-Ergaenzung)

---

### Schritt 12: Feingranulare Intents + Multi-Intent-Routing

**Voraussetzung:** Echte Chat-Logs zeigen wo die groben 5 Intents aus Iter 1
nicht ausreichen (z.B. `erklaerung` liefert bei Zusammenfassungs-Anfragen
die falschen chunk_types).

**12a. `intents.py` erweitern** – gestrichene Intents aus Schritt 1 einfuehren
(selektiv, nur wenn Logs den Bedarf belegen):

```python
# Aus der "Gestrichene Intents"-Tabelle in Schritt 1 holen:
# werk_lokalisieren, zitat_suchen, vergleich, zusammenfassung,
# beleg_pruefung, follow_up, hypothetisch, meta_assistent
```

**12b. Multi-Intent**: `IntentResult` um `secondary_intent` erweitern:

```python
class IntentResult(BaseModel):
    intent: str
    secondary_intent: str | None = None
    confidence: float
    reasoning: str
    lemma: str = ""

# In route_retrieval_plan:
primary = INTENT_CHUNK_TYPE_MAP.get(state["intent"], [])
secondary = INTENT_CHUNK_TYPE_MAP.get(state.get("secondary_intent") or "", [])
combined = list(dict.fromkeys(primary + secondary))   # Reihenfolge erhalten, Duplikate weg
return {"retrieval_plan": combined}
```

---

### Schritt 13: `memory_service.py` anlegen

**Datei:** `app/retrieval/services/memory_service.py` *(neu)*

```python
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

MAX_TURNS_BEFORE_SUMMARY = 12   # 12 Messages = 6 Turns

async def maybe_summarize_thread(
    messages: list[BaseMessage],
    llm: ChatOpenAI,
) -> list[BaseMessage]:
    """
    Verdichtet alte Turns zu einem Summary-SystemMessage wenn die History
    ueber MAX_TURNS_BEFORE_SUMMARY Messages angewachsen ist.
    """
    if len(messages) <= MAX_TURNS_BEFORE_SUMMARY:
        return messages

    to_summarize = messages[:-6]    # alles ausser den letzten 3 Turns behalten
    recent = messages[-6:]

    summary_text = await llm.ainvoke([
        HumanMessage(
            "Fasse die folgenden Gespraechsabschnitte in 3-5 Saetzen zusammen. "
            "Behalte erwaehnte Konzepte, offene Fragen und wichtige Erkenntnisse:\n\n"
            + "\n".join(f"{m.__class__.__name__}: {m.content}" for m in to_summarize)
        )
    ])

    from langchain_core.messages import SystemMessage
    return [SystemMessage(f"[Gespraechs-Zusammenfassung]: {summary_text.content}")] + recent
```

In `compose_answer` einbinden: `messages_in = await maybe_summarize_thread(state["messages"], llm)`

---

### Schritt 14: Quote-Ergaenzungs-Retrieval

Nach `compose_answer`, vor `attach_citations`: Optionalen `enrich_with_quotes`-Node
einfuegen.

```python
async def enrich_with_quotes(state: ChatState, config: RunnableConfig) -> dict:
    """Ergaenzt den Kontext bei Bedarf um passende Quote-Chunks."""
    if "quote" in state.get("retrieval_plan", []):
        return {}   # schon im Plan enthalten

    quotes = await hybrid_retrieve(
        query=state["user_message"],
        collection=state["collection_name"],
        k=3,
        filters=payload_filter(chunk_types=["quote"]),
    )
    if quotes:
        extra = build_context(quotes)
        return {
            "context_text": state["context_text"] + "\n\nErgaenzende Zitate:\n" + extra,
            "context_refs": state["context_refs"] + [q["id"] for q in quotes],
        }
    return {}
```

---

### Schritt 15: Lokaler Embedding-Classifier (Intent)

**Datei:** `app/retrieval/services/local_intent_classifier.py` *(neu)*

Erst anlegen wenn >= 50 annotierte Chat-Logs via `event_content` vorliegen.
Voraussetzung: `sentence-transformers` in `requirements.txt`.

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

MODEL_NAME = "intfloat/multilingual-e5-small"   # ~115MB, M1-optimiert

class LocalIntentClassifier:
    def __init__(self, model_path: str):
        self._encoder = SentenceTransformer(MODEL_NAME, device="mps")
        self._clf: LogisticRegression = joblib.load(model_path)

    def predict(self, text: str) -> tuple[str, float]:
        emb = self._encoder.encode([text])
        label = self._clf.predict(emb)[0]
        prob = self._clf.predict_proba(emb).max()
        return label, float(prob)
```

In `classify_intent`-Node: zuerst `LocalIntentClassifier.predict()` versuchen;
nur wenn `confidence < 0.7` auf DeepSeek-API-Call fallen lassen (Hybrid-Modus).

---

## Iteration 4 – Optimierung (Ziel: Metriken, Kosten, Classifier-Tuning)

---

### Schritt 16: Metriken aus `event_content` extrahieren

**SQL-Abfragen fuer Monitoring (in `app/shared/sql_commands.py` ergaenzen):**

```sql
-- Intent-Verteilung der letzten 7 Tage
SELECT
  metadata->>'intent'    AS intent,
  COUNT(*)               AS total,
  AVG((metadata->>'intent_confidence')::float) AS avg_conf
FROM event_metadata
WHERE event_type = 'chat_turn'
  AND created_at > NOW() - INTERVAL '7 days'
GROUP BY intent
ORDER BY total DESC;

-- Sufficiency-Rate
SELECT
  metadata->>'sufficiency' AS sufficiency,
  COUNT(*) AS total
FROM event_metadata
WHERE event_type = 'chat_turn'
GROUP BY sufficiency;

-- Trainings-Export fuer Classifier
SELECT
  content->>'user_message'   AS text,
  metadata->>'intent'        AS label
FROM event_metadata em
JOIN event_content ec ON em.id = ec.event_metadata_id
WHERE event_type = 'chat_turn'
  AND metadata->>'intent_confirmed' = 'true'   -- nur manuell bestaetigt
ORDER BY em.created_at DESC;
```

---

### Schritt 17: Classifier-Training

```bash
# Trainingsdaten aus Postgres exportieren
python scripts/export_chat_training_data.py --output data/intent_training.jsonl

# Classifier trainieren
python scripts/train_intent_classifier.py \
  --data data/intent_training.jsonl \
  --output models/intent_clf.joblib
```

Scripte anlegen in `scripts/` (einmalige Hilfswerkzeuge, nicht in `app/`).

---

### Schritt 18: Latenz-Profiling

LangGraph gibt pro Node die Dauer aus. Im `GraphEventRecorder` beim `on_chain_end`-Event
die `elapsed_ms` je Node mitloggen:

```python
if event["event"] == "on_chain_end":
    node_name = event.get("name", "")
    run_id   = event.get("run_id", "")
    elapsed  = event["data"].get("elapsed_ms", 0)
    # In event_metadata.metadata: {"node": node_name, "elapsed_ms": elapsed}
```

---

## Dateistruktur nach Iteration 1 (Gesamtueberblick)

```
app/
├── api/
│   ├── chat.py                      ← NEU (Schritt 4)
│   └── rag.py                       ← unveraendert
├── retrieval/
│   ├── graphs/
│   │   ├── intents.py               ← NEU (Schritt 1)
│   │   ├── assistant_chat_graph.py  ← NEU (Schritt 3)
│   │   └── concept_explain_worldviews.py ← unveraendert
│   ├── prompts/
│   │   ├── intent_classify.prompt   ← NEU (Schritt 2)
│   │   └── philo_von_freisinn.py    ← unveraendert (wiederverwendet)
│   ├── services/
│   │   ├── citation_service.py      ← NEU (Schritt 9, Iter. 2)
│   │   ├── memory_service.py        ← NEU (Schritt 13, Iter. 3)
│   │   ├── event_recorder.py        ← unveraendert
│   │   └── graph_event_recorder.py  ← unveraendert
│   └── utils/
│       ├── retrievers.py            ← unveraendert (wiederverwendet)
│       └── sufficiency.py           ← NEU optional (Schritt 11, Iter. 2)
├── db/
│   └── tables.py                    ← unveraendert
└── main.py                          ← modifiziert (Schritt 5)
```

---

## Checkliste je Iteration

### Iteration 1
- [ ] Schritt 1: `intents.py` + Test
- [ ] Schritt 2: `intent_classify.prompt`
- [ ] Schritt 3: `assistant_chat_graph.py` (alle Nodes + Graph-Aufbau)
- [ ] Schritt 4: `app/api/chat.py` (SSE-Endpunkt)
- [ ] Schritt 5: `main.py` – Graph-Registrierung + Router
- [ ] Schritt 6: E2E-Test per curl (Token-Stream + Citations + Follow-up)
- [ ] Schritt 7: Event-Logging via `GraphEventRecorder`

### Iteration 2
- [ ] Schritt 8: `AsyncPostgresSaver` aktivieren
- [ ] Schritt 9: `citation_service.py`
- [ ] Schritt 10: `verify_grounding`-Node + konditionaler Edge
- [ ] Schritt 11: `sufficiency.py` (bei Bedarf)

### Iteration 3
- [ ] Schritt 12: Feingranulare Intents (aus "Gestrichene Intents"-Tabelle, Schritt 1) + Multi-Intent
- [ ] Schritt 13: `memory_service.py` + `maybe_summarize_thread`
- [ ] Schritt 14: `enrich_with_quotes`-Node
- [ ] Schritt 15: `local_intent_classifier.py` (erst ab 50 Trainingsbeispiele)

### Iteration 4
- [ ] Schritt 16: Metriken-SQL in `sql_commands.py`
- [ ] Schritt 17: Classifier-Training-Skripte
- [ ] Schritt 18: Latenz-Logging im Event-Recorder

---

## Hinweise zur Nutzung mit Cursor Composer

Dieser Plan ist so detailliert, dass er von Composer Schritt fuer Schritt abgearbeitet
werden kann. Empfohlenes Vorgehen:

1. **Einen Schritt pro Composer-Session**: Schritt-Nummer angeben, Dateipfad nennen.
   Beispiel: *"Implementiere Schritt 3d (Node 1: classify_intent) in
   `assistant_chat_graph.py` gemaess ASSITANTS_CHAT_PLAN.md"*

2. **Nach jedem Schritt testen**: Die Tests in den Schritten sind ausfuehrbar –
   immer erst testen, bevor der naechste Schritt beginnt.

3. **Kritische Stellen** – hier besonders pruefen:
   - `_make_llm().with_structured_output(..., method="json_mode")` – `method` **muss**
     angegeben werden (DeepSeek-spezifisch, im Spike bestaetigt).
   - SSE-Event-Filter in `chat.py`: nur `on_chat_model_stream`-Events weiterleiten,
     nicht alle `on_chain_*`-Events (sonst Flut).
   - `add_messages`-Reducer: `messages` im State **niemals direkt ueberschreiben**,
     immer als Liste zurueckgeben (`return {"messages": [AIMessage(...)]}`).
   - `AsyncPostgresSaver.setup()` muss im lifespan **vor** `build_chat_graph()` aufgerufen
     werden.
