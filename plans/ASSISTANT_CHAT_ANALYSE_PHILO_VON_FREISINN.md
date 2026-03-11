# Analyse: Assistenten-Chat fuer "Philo von Freisinn"

Ziel dieses Dokuments: Eine erste technische Analyse fuer einen DeepSeek-aehnlichen Chat in der Assistenten-Seite (Startfall: "Philo von Freisinn") und ein sofort nutzbarer Prompt fuer Cursor.ai mit Claude Sonnet 4.6.

## Analyse Version 1.0 (codebase-basiert)

---

## 1. Zielbild

Auf der Assistenten-Seite (z.B. "Philo von Freisinn") erscheint neben dem `Statistik`-Button ein
`Chat`-Button. Ein Klick oeffnet eine Thread-Ansicht, in der User-Nachrichten und Assistenten-
Antworten chronologisch dargestellt werden – dem UX-Muster von https://chat.deepseek.com/
nachempfunden.

**Kernprinzipien:**
- Jede Eingabe wird zuerst klassifiziert (Intent), danach an den passenden Retrieval-Pfad gerouted.
- Antworten basieren ausschliesslich auf der RAG-Collection des jeweiligen Assistenten.
- Jede Antwort enthaelt Quellenverweise (Buch / Vortrag / Chunk-ID), analog zu den
  vorhandenen `context_refs` im `event_content`-Log.
- Conversation-Kontext (letzter N Turns) wird fuer korrekte Pronomen-Aufloesung mitgefuehrt.
- Halluzinationsschutz durch Confidence-Scoring und defensivere Formulierung bei
  schlechter Evidenzlage.

---

## 2. Ist-Zustand des ragrun-Repos (relevante Bausteine)

> **Stand nach Refactoring (Maerz 2026):** Obsolete Schichten wurden entfernt.
> Siehe Abschnitt 2a fuer Details.

Die folgenden Infrastrukturbausteine koennen direkt wiederverwendet werden:

| Baustein | Datei | Relevanz fuer Chat |
|---|---|---|
| `dense_retrieve` / `sparse_retrieve` / `hybrid_retrieve` | `app/retrieval/utils/retrievers.py` | Kernretrieval, unveraendert nutzbar |
| `rerank_by_embedding` | `app/retrieval/utils/retrievers.py` | Embedding-Reranking nach RRF-Fusion |
| `build_context` | `app/retrieval/utils/retrievers.py` | Kontextstring + `chunk_id`-Referenzen |
| `payload_filter` | `app/retrieval/utils/retrievers.py` | Weltanschauungs- und chunk_type-Filter |
| `_retrieve_with_widen` | `app/retrieval/graphs/concept_explain_worldviews.py` | Widening-Logik bei duennen Ergebnissen |
| `_chat_with_retry` | `app/retrieval/graphs/concept_explain_worldviews.py` | Retry + Sentence-Completion-Logik |
| `EventRecorder` / `GraphEventRecorder` | `app/retrieval/services/event_recorder.py` | Dual-Write-Logging (permanent + 7d-Rotation) |
| `event_metadata` / `event_content` | `app/db/tables.py` | Log-Tabellen – koennen direkt erweitert werden |
| `load_system_prompt()` | `app/retrieval/prompts/philo_von_freisinn.py` | Laedt Philo-Persona aus `ragkeep/assistants/philo-von-freisinn/prompts/instruction.prompt` |
| `DeepSeekClient` | `app/infra/deepseek_client.py` | LLM-Client |
| `EmbeddingClient` | `app/infra/embedding_client.py` | Embedding-Service |
| `QdrantClient` | `app/infra/qdrant_client.py` | Vektordatenbank |
| `CHUNK_TYPE_ENUM` | `app/shared/models.py` | Alle gueltigen chunk_types |

**Relevante chunk_types** (aus `app/shared/models.py`):
`book`, `secondary_book`, `chapter_summary`, `begriff_list`, `talk`, `talk_summary`,
`essay`, `essay_summary`, `quote`, `explanation`, `explanation_summary`, `typology`

### 2a. Refactoring-Stand (Maerz 2026)

Vor der Chat-Implementierung wurden obsolete und gebrochene Teile entfernt:

| Geloeschte Datei | Grund |
|---|---|
| `app/api/concept_explain.py` | Kaputte Imports; nie in `main.py` eingebunden |
| `app/retrieval/graphs/philo_von_freisinn.py` | Thin-Wrapper um alten, simpleren Chain; ersetzt durch `concept_explain_worldviews` |
| `app/retrieval/chains/concept_explain.py` | Alter Chain ohne Retry, Hybrid, Worldview-Filter |
| `app/retrieval/services/concept_explain_service.py` | Service fuer alten Chain; nicht eingebunden |
| `app/services/concept_explain_service.py` | Re-Export-Shim des obigen |
| `app/retrieval/api/essay_finetune.py` | Disabled, inkompatibel mit aktueller Implementierung |
| `app/retrieval/chains/essay_finetune.py` | Dto. |
| `app/retrieval/services/essay_finetune_service.py` | Dto. |
| `app/retrieval/services/retrieval_logging.py` | Ersetzt durch `EventRecorder` / `GraphEventRecorder` |

`GraphEventRecorder` wurde dabei verschlankt: delegiert jetzt sauber an `EventRecorder`
ohne eigene Logging-Logik.

---

## 3. Architekturvarianten

### Variante A: Einfacher LCEL-Flow + RouterChain

```
UserMessage
    → IntentChain (LLM-Klassifikation)
    → RouterChain (waehlt Retriever nach Intent)
    → RetrievalChain (dense/hybrid)
    → AnswerChain (DeepSeek)
    → CitationAttacher
    → Response
```

| | Wert |
|---|---|
| **Aufwand** | gering (~1-2 Tage) |
| **Vorteile** | schnell, linearer Datenfluss, kein Framework-Overhead |
| **Nachteile** | Multi-Intent, Retry-Logik und Verifikations-Loops werden unuebersichtlich |
| **Einsatz** | MVP-Iteration 1, wenn Time-to-Value dominiert |

### Variante B: Voller LangGraph (offizielle `langgraph`-Library)

State-Machine mit expliziten Nodes, Edges und konditionalen Branches. Jeder Node ist
eine Python-Funktion, die den `GraphState` veraendert.

| | Wert |
|---|---|
| **Aufwand** | mittel (~3-5 Tage Setup + Lernkurve) |
| **Vorteile** | Branching, Retry-Loops, Multi-Tool, Streaming-Support, gute Langfuse-Integration |
| **Nachteile** | Neue Abhaengigkeit (`langgraph`), initiales Designaufwand |
| **Einsatz** | Ab Iteration 2, sobald Verifikations-Node und Multi-Intent benoetigt werden |

### Variante C: Eigener Async-Graph (analog bisherigem Stil) + LCEL intern ✅ Empfohlen

Das Repo hat bereits eine eigenstaendige "Graph"-Konvention in
`app/retrieval/graphs/concept_explain_worldviews.py`: async-Funktionen mit `asyncio.Semaphore`,
explizitem State-Dataclass, `GraphEventRecorder` und `_chat_with_retry`.

Vorschlag: denselben Stil fuer den Chat-Graph weiterfuehren und **intern** LCEL-Chains fuer
einzelne Nodes verwenden. Das vermeidet eine weitere Framework-Abhaengigkeit und bleibt
konsistent mit dem bestehenden Code.

| | Wert |
|---|---|
| **Aufwand** | gering-mittel (~2-3 Tage) |
| **Vorteile** | konsistenter Stil, alle Hilfsfunktionen wiederverwendbar, kein neues Framework |
| **Nachteile** | Kein offiziales LangGraph-Streaming / Persistence out-of-the-box |
| **Einsatz** | Gesamte Roadmap (MVP → Produktionsreife) |

**Fazit: Variante C fuer Iteration 1+2, optionaler Umstieg auf echtes LangGraph (Variante B)
ab Iteration 3 falls Streaming/Persistence benoetigt werden.**

---

## 4. Intent-Klassifikation (vor Retrieval)

### Intent-Schema (multi-label)

| Intent-Label | Beschreibung | Beispiele |
|---|---|---|
| `begriff_definieren` | Definition/Erklaerung eines Konzepts | "Was ist Pneumatismus?" |
| `werk_lokalisieren` | Buch- / Vortragslookup | "In welchem Vortrag spricht Steiner ueber das Aetherleib?" |
| `zitat_suchen` | Wörtliche oder sinngemaeße Zitate | "Hast du ein Zitat zum Tema Karma?" |
| `vergleich` | Zwei Konzepte / Weltanschauungen vergleichen | "Unterschied Materialismus vs. Spiritualismus?" |
| `erklaerung_vertiefung` | Ausfuehrliche Herleitung | "Erklaere die 12 Weltanschauungen im Detail" |
| `zusammenfassung` | Kurzfassung eines Werks / Themas | "Fasse den 3. Vortrag kurz zusammen" |
| `beleg_pruefung` | Stimmt Aussage X? Mit Quellen belegen | "Hat Steiner wirklich gesagt, dass..." |
| `meta_assistent` | Selbstauskunft des Assistenten | "Was kannst du? Welche Buecher kennst du?" |
| `konversationell` | Smalltalk / allgemeine Hoeflichkeit | "Danke" / "Hallo" |
| `out_of_scope` | Ausserhalb der Collection / Domain | Fragen ueber Aktienkurse etc. |
| `follow_up` | Anschluss an vorigen Turn | "Kannst du das naeher erklaeren?" |
| `hypothetisch` | Spekulative / vergleichende Fragen | "Was wuerde ein Materialist dazu sagen?" |

**Confidence-Logik:**
- Gibt es 2 Labels mit `confidence_delta < 0.15` → Multi-Tool-Pfad aktivieren.
- `out_of_scope` mit `confidence > 0.7` → direkte Ablehnungsantwort ohne Retrieval.
- `konversationell` → keine Retrieval, direkte LLM-Antwort mit Persona.

### Klassifikationsstrategie (Iteration 1 → 3)

| Phase | Methode | Aufwand | Qualitaet |
|---|---|---|---|
| Iteration 1 | Zero-shot LLM-Intent via `deepseek-chat` + strukturierter Prompt | gering | gut |
| Iteration 2 | Embedding + kNN (Cosine auf Few-shot-Beispielen) als schnelle Vorstufe | mittel | sehr gut |
| Iteration 3 | Hybrid: Embedding-Klassifikator + LLM-Fallback bei Unsicherheit | hoch | optimal |

Fuer Iteration 1 genuegt ein strukturierter Prompt der Form:

```python
# app/retrieval/chains/intent_classification_chain.py
INTENT_SYSTEM_PROMPT = """Du klassifizierst Nutzerfragen fuer den Assistenten "Philo von Freisinn".
Antworte NUR mit einem JSON-Objekt:
{"labels": ["<label1>", ...], "confidence": {"<label1>": 0.0-1.0, ...}, "reasoning": "<kurz>"}
Moegliche Labels: {INTENT_LABELS}"""
```

---

## 5. Routing nach chunk_types

### Routing-Matrix

| Intent | Primaer chunk_types | Sekundaer chunk_types | Hinweise |
|---|---|---|---|
| `begriff_definieren` | `begriff_list`, `chapter_summary` | `book`, `explanation` | Erst schnelle Definition, dann Originalbeleg |
| `werk_lokalisieren` | `chapter_summary`, `talk_summary` | `book`, `talk` | Sparse-BM25 gut fuer Titelmatch |
| `zitat_suchen` | `quote` | `book`, `talk` | `quote`-Chunks direkt; `book` fuer Kontext |
| `vergleich` | `chapter_summary`, `essay` | `book`, `secondary_book` | Breiter Recall, dann Reranking |
| `erklaerung_vertiefung` | `book`, `talk` | `essay`, `secondary_book` | Dense/Hybrid, max_chars erhoehen |
| `zusammenfassung` | `chapter_summary`, `talk_summary` | `essay_summary` | Nur Summary-Typen; kein `book` noetig |
| `beleg_pruefung` | `book`, `quote` | `talk` | Strikt Primaerquellen; `essay` nachrangig |
| `meta_assistent` | – | – | Kein Retrieval; Antwort aus Assistenten-Profil |
| `konversationell` | – | – | Kein Retrieval |
| `follow_up` | (aus Vorturn erben) | ggf. `book` fuer mehr Kontext | Vorturn-Chunks recyceln |
| `hypothetisch` | `book`, `secondary_book` | `essay` | Breite Suche, defensive Formulierung |

### Praktische Filterregel (passt auf `payload_filter` in `retrievers.py`)

```python
INTENT_CHUNK_TYPE_MAP: dict[str, list[str]] = {
    "begriff_definieren":   ["begriff_list", "chapter_summary", "book", "explanation"],
    "werk_lokalisieren":    ["chapter_summary", "talk_summary", "book", "talk"],
    "zitat_suchen":         ["quote", "book", "talk"],
    "vergleich":            ["chapter_summary", "essay", "book", "secondary_book"],
    "erklaerung_vertiefung":["book", "talk", "essay", "secondary_book"],
    "zusammenfassung":      ["chapter_summary", "talk_summary", "essay_summary"],
    "beleg_pruefung":       ["book", "quote", "talk"],
    "follow_up":            [],  # dynamisch aus Vorturn
    "hypothetisch":         ["book", "secondary_book", "essay"],
}
```

Die bestehende `payload_filter`-Funktion in `retrievers.py` unterstuetzt bereits den Filter
`{"key": "chunk_type", "match": {"any": [...]}}` – direkt verwendbar.

---

## 6. LangGraph-State / Nodes / Edges (Entwurf)

### State-Dataclass

```python
# app/retrieval/graphs/assistant_chat_graph.py

@dataclass(slots=True)
class ChatState:
    # Input
    assistant_id: str
    thread_id: str
    user_message: str
    conversation_history: list[dict]       # letzte N Turns [{"role": ..., "content": ...}]

    # Intent
    intent_labels: list[str] = field(default_factory=list)
    intent_confidence: dict[str, float] = field(default_factory=dict)
    multi_intent: bool = False

    # Retrieval
    retrieval_plan: list[str] = field(default_factory=list)  # chunk_types
    retrieved_chunks: list[RetrievedSnippet] = field(default_factory=list)
    context_text: str = ""
    context_refs: list[str] = field(default_factory=list)
    retrieval_mode: str = "dense"
    sufficiency: str = "unknown"           # "high" / "medium" / "low" / "insufficient"

    # Antwort
    draft_answer: str = ""
    verified_answer: str = ""
    confidence_score: float = 0.0
    needs_retry: bool = False
    retry_count: int = 0

    # Output
    citations: list[dict] = field(default_factory=list)   # [{chunk_id, source_title, page}]
    final_response: str = ""
    errors: list[str] = field(default_factory=list)
```

### Nodes

```
Node 1: classify_intent
    Input:  user_message + conversation_history (letzte 4 Turns als Kontext)
    Output: intent_labels, intent_confidence, multi_intent
    Impl:   DeepSeek-Chat + strukturierter JSON-Prompt
    Datei:  app/retrieval/chains/intent_classification_chain.py

Node 2: route_retrieval_plan
    Input:  intent_labels, multi_intent
    Output: retrieval_plan (geordnete Liste von chunk_types)
    Impl:   Lookup in INTENT_CHUNK_TYPE_MAP, Multi-Intent → Union mit Priorisierung
    Datei:  app/retrieval/chains/retrieval_plan_chain.py

Node 3: retrieve_chunks
    Input:  user_message + conversation_history, retrieval_plan, assistant_id → collection
    Output: retrieved_chunks, context_text, context_refs, retrieval_mode
    Impl:   _retrieve_with_widen (existiert bereits), chunk_type-Filter via payload_filter
    Datei:  wiederverwendet app/retrieval/utils/retrievers.py

Node 4: assess_sufficiency
    Input:  retrieved_chunks, context_text
    Output: sufficiency, needs_retry
    Impl:   _assess_sufficiency (existiert bereits in concept_explain_worldviews.py)
    Datei:  extrahieren nach app/retrieval/utils/sufficiency.py

Node 5: compose_answer
    Input:  user_message, conversation_history, context_text, intent_labels
    Output: draft_answer
    Impl:   _chat_with_retry (existiert bereits), angepasster Chat-Prompt
    Datei:  app/retrieval/chains/chat_answer_chain.py

Node 6: verify_grounding
    Input:  draft_answer, context_text, context_refs
    Output: confidence_score, needs_retry
    Impl:   zweiter LLM-Call: "Widerspricht die Antwort dem Kontext?" + Claim-Coverage-Score
    Datei:  app/retrieval/chains/grounding_verification_chain.py

Node 7: attach_citations
    Input:  verified_answer, context_refs (chunk_ids)
    Output: citations [{chunk_id, source_title, lecture_date, page_ref}]
    Impl:   Lookup in rag_chunks-Tabelle (metadata-Spalte enthaelt Quelleninfos)
    Datei:  app/retrieval/services/citation_service.py

Node 8: finalize_response
    Input:  verified_answer, citations, confidence_score, sufficiency
    Output: final_response (Markdown mit eingebetteten Quellenlinks)
    Impl:   Template-Rendering + defensiver Modus bei niedrigem confidence_score
```

### Edges und Bedingungen

```
classify_intent
    → (intent == "out_of_scope" oder "konversationell") → finalize_response  [Shortcut]
    → sonst → route_retrieval_plan

route_retrieval_plan → retrieve_chunks

retrieve_chunks → assess_sufficiency

assess_sufficiency
    → (sufficiency == "insufficient" und retry_count < 2) → retrieve_chunks  [Widen-Retry]
    → sonst → compose_answer

compose_answer → verify_grounding

verify_grounding
    → (needs_retry und retry_count < 2) → retrieve_chunks  [Evidence-Retry]
    → sonst → attach_citations

attach_citations → finalize_response
```

---

## 7. LCEL-Ketten-Vorschlaege (fuer einzelne Nodes)

### Intent-Klassifikation (Node 1)

```python
# app/retrieval/chains/intent_classification_chain.py
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

intent_chain = (
    ChatPromptTemplate.from_messages([
        ("system", INTENT_SYSTEM_PROMPT),
        ("human", "{user_message}"),
    ])
    | deepseek_llm
    | JsonOutputParser()
    | RunnableLambda(validate_intent_output)
)
```

> Hinweis: `deepseek_llm` kann per `langchain_openai.ChatOpenAI` mit `base_url` auf DeepSeek
> gesetzt werden – kompatibel mit OpenAI-API-Format.

### Grounding-Verifikation (Node 6)

```python
# app/retrieval/chains/grounding_verification_chain.py
GROUNDING_PROMPT = """Du pruefst, ob eine Antwort dem folgenden Kontext widerspricht.
Kontext:
{context}

Antwort:
{draft_answer}

Antworte mit JSON: {{"contradicts": bool, "coverage": 0.0-1.0, "issues": [...]}}"""

grounding_chain = (
    ChatPromptTemplate.from_template(GROUNDING_PROMPT)
    | deepseek_llm
    | JsonOutputParser()
)
```

### Chat-Antwort (Node 5)

Der Prompt baut auf der Persona von Philo von Freisinn auf und kombiniert:
1. System-Prompt mit Assistenten-Charakter (aus `ragkeep/assistants/philo-von-freisinn/`)
2. Letzten N Turns als `conversation_history`
3. RAG-Kontext als separater System-Block

```python
# app/retrieval/chains/chat_answer_chain.py
answer_chain = (
    ChatPromptTemplate.from_messages([
        ("system", PHILO_PERSONA_PROMPT),
        ("system", "Kontext aus den Quellen:\n{context}"),
        *[(m["role"], m["content"]) for m in conversation_history[-6:]],
        ("human", "{user_message}"),
    ])
    | deepseek_llm
    | StrOutputParser()
)
```

---

## 8. Conversation Memory

### Kurzfristiges Memory (Iteration 1)

- Die letzten 6-12 Turns (User + Assistent) werden als `conversation_history` im `ChatState`
  mitgefuehrt.
- Werden direkt in den Prompt eingebettet (keine externe Datenbank).
- **Thread-ID** identifiziert den Chat-Session-Kontext.

### Persistenz (Iteration 2)

Neue Tabelle `chat_threads` (Alembic-Migration):

```sql
-- Neue Tabelle fuer Chat-Persistenz
CREATE TABLE chat_threads (
    id          BIGSERIAL PRIMARY KEY,
    thread_id   UUID NOT NULL,
    assistant_id VARCHAR(128) NOT NULL,
    turn_index  INTEGER NOT NULL,
    role        VARCHAR(16) NOT NULL,   -- 'user' | 'assistant'
    content     TEXT NOT NULL,
    intent_labels TEXT[],
    chunk_ids   JSONB,
    confidence  FLOAT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON chat_threads (thread_id, turn_index);
```

Datei: `app/db/migrations/versions/0008_add_chat_threads.py`

### Mittelfristiges Memory (Iteration 3)

- Nach jeweils N Turns wird ein LLM-Call gemacht, der die wesentlichen "Conversation Facts"
  des Threads verdichtet (Topics, offene Fragen, erwaehnte Konzepte).
- Diese Facts werden als `thread_summary` neben dem Rolling-Window gehalten.
- Datei: `app/retrieval/services/memory_service.py`

---

## 9. Re-Ranking

Das bestehende `rerank_by_embedding` (Dot-Product) in `retrievers.py` ist fuer Iteration 1
ausreichend. Ab Iteration 2:

| Option | Qualitaet | Laufzeit | Aufwand |
|---|---|---|---|
| Embedding Dot-Product (aktuell) | gut | ~50ms | 0 (existiert) |
| Cross-Encoder (z.B. `ms-marco-MiniLM-L-6-v2`) | sehr gut | ~200ms | mittel |
| LLM-Reranker (DeepSeek als Judge) | exzellent | ~500ms | gering (nur Prompt) |

**Empfehlung:** Ab Iteration 2 einen LLM-Reranker als separaten Node einsetzen, da DeepSeek
bereits vorhanden ist und keine neue Modellabhaengigkeit entsteht.

---

## 10. Halluzinationsschutz und Confidence-Score

### Confidence-Berechnung

```
confidence_score = (
    0.4 * retrieval_score      # Durchschnittlicher Rerank-Score der Top-Chunks
    + 0.3 * coverage_score     # Grounding-Check: Anteil der Claims mit Beleg
    + 0.2 * intent_confidence  # Intent-Klassifikations-Konfidenz
    + 0.1 * sufficiency_score  # "high"=1.0, "medium"=0.7, "low"=0.3, "insufficient"=0.0
)
```

### Antwortmodi je Confidence

| confidence_score | Modus | Formulierung |
|---|---|---|
| ≥ 0.75 | Normal | Direkte Antwort mit Quellenverweisen |
| 0.50–0.74 | Vorsichtig | "Nach dem vorliegenden Material am ehesten..." + Quellen |
| 0.30–0.49 | Defensiv | "Im verfuegbaren Material ist das nur bedingt belegt..." |
| < 0.30 | Ablehnend | "Dazu finde ich in den Quellen keine hinreichende Grundlage." |

### RAG-Verifikationsstrategie

1. **Answer-vs-Evidence Node** (Node 6): Zweiter LLM-Call prueft Widerspruch.
2. **Source-Inline-Citations**: Antwort-Template erzwingt `[Quelle: Titel, Jahr]`-Marker.
3. **Retry bei schlechter Evidenz**: `assess_sufficiency` triggert `retrieve_chunks` erneut
   mit erweitertem `widen_to`-Parameter (Mechanismus existiert bereits in `_retrieve_with_widen`).

---

## 11. Lokaler Classifier (MacBook Pro M1)

### Machbarkeit

Ja, grundsaetzlich moeglich. Empfehlung nach Iterations-Reife:

| Phase | Methode | Modell | Laufzeit M1 | Trainingsaufwand |
|---|---|---|---|---|
| Iter. 1 | Zero-shot DeepSeek | `deepseek-chat` | ~300ms API | keine |
| Iter. 2 | Embedding + LogReg | `multilingual-e5-small` (lokal) | ~20ms | ~50 Beispiele |
| Iter. 3 | Fine-tuned MiniLM | z.B. `paraphrase-multilingual-MiniLM-L12` | ~30ms MPS | ~200 Beispiele |
| Iter. 4 | Destilliertes Modell aus Chat-Logs | custom | ~15ms | viel |

**Pragmatischste Option:** Iteration 2 – `multilingual-e5-small` laeuft problemlos auf M1
mit PyTorch MPS, braucht ~50 annotierte Beispiele aus echten Chat-Logs und ersetzt den
teuren API-Call fuer die Intent-Klassifikation.

---

## 12. API-Endpunkt und Dateistruktur

### Neuer FastAPI-Router

```
app/api/chat.py
```

```
POST /api/v1/agent/{assistant_slug}/chat
Body: {
    "thread_id": "uuid",        # optional; neu generiert wenn nicht angegeben
    "user_message": "...",
    "conversation_history": [...],  # optional; vom Client gehalten (Iter. 1)
    "stream": false             # Streaming-Support ab Iter. 3
}
Response: {
    "thread_id": "uuid",
    "response": "...",
    "citations": [...],
    "confidence_score": 0.0-1.0,
    "intent_labels": [...],
    "sufficiency": "high|medium|low|insufficient"
}
```

### Neue Dateien (vollstaendige Liste)

| Datei | Inhalt |
|---|---|
| `app/api/chat.py` | FastAPI-Router fuer Chat-Endpunkt |
| `app/retrieval/graphs/assistant_chat_graph.py` | Haupt-Graph mit ChatState + Node-Orchestrierung |
| `app/retrieval/chains/intent_classification_chain.py` | Intent-Klassifikation via LLM |
| `app/retrieval/chains/retrieval_plan_chain.py` | Intent → chunk_types Mapping |
| `app/retrieval/chains/chat_answer_chain.py` | RAG-gestuetzte Antwortgenerierung |
| `app/retrieval/chains/grounding_verification_chain.py` | Answer-vs-Evidence Pruefung |
| `app/retrieval/services/chat_orchestrator_service.py` | Service-Layer ueber Graph |
| `app/retrieval/services/citation_service.py` | Chunk-ID → Quellen-Metadaten Lookup |
| `app/retrieval/services/memory_service.py` | Thread-Summary-Verdichtung (Iter. 3) |
| `app/retrieval/utils/sufficiency.py` | _assess_sufficiency (aus graph extrahiert) |
| `app/retrieval/prompts/chat_answer.prompt` | Persona-Prompt fuer Philo von Freisinn |
| `app/retrieval/prompts/intent_classify.prompt` | Intent-Klassifikations-Prompt |
| `app/retrieval/prompts/grounding_verify.prompt` | Grounding-Check-Prompt |
| `app/db/migrations/versions/0008_add_chat_threads.py` | Alembic-Migration fuer chat_threads |

### Wiederverwendete Dateien (unveraendert oder minimal angepasst)

- `app/retrieval/utils/retrievers.py` – alle Retrieve-Funktionen
- `app/retrieval/services/event_recorder.py` – Logging (Chat-Events loggen)
- `app/retrieval/services/graph_event_recorder.py` – Thin-Wrapper um EventRecorder
- `app/retrieval/prompts/philo_von_freisinn.py` – `load_system_prompt()` laedt Philo-Persona aus ragkeep
- `app/infra/deepseek_client.py`, `embedding_client.py`, `qdrant_client.py`
- `app/core/providers.py` – Lazy Singletons

---

## 13. Risikoliste und Gegenmassnahmen

| Risiko | Wahrscheinlichkeit | Impact | Gegenmassnahme |
|---|---|---|---|
| Hohe Latenz (Intent + Retrieval + LLM = 1-2s+) | hoch | mittel | Streaming (Iter. 3); Intent-Classifier lokal (Iter. 2) |
| Halluzination bei duennem Kontext | mittel | hoch | Sufficiency-Check + defensiver Modus + Confidence-Score |
| Intent-Klassifikation falsch | mittel | mittel | Multi-Intent-Pfad; Fallback auf breites Retrieval |
| chunk_type-Filter filtert zu stark | mittel | mittel | Fallback ohne chunk_type-Filter wenn < 2 Treffer |
| Thread-Kontext wird zu lang | niedrig | niedrig | Sliding Window (max 12 Turns) + Summary-Verdichtung |
| DeepSeek-API-Ausfall | niedrig | hoch | `_chat_with_retry` (existiert); kein Fallback (Designprinzip) |
| `quote`-Chunks nicht immer vorhanden | mittel | niedrig | `quote` ist Ergaenzung, nie alleinige Quelle |
| Persona-Drift ueber langen Thread | niedrig | mittel | System-Prompt wird bei jedem Turn erneut eingesetzt |

---

## 14. Umsetzungs-Roadmap

### Iteration 1 – Funktionaler Chat (ca. 3-5 Tage)

**Ziel:** Ende-zu-Ende-Chat fuer "Philo von Freisinn" ohne UI.

- [ ] `app/api/chat.py`: POST-Endpunkt mit `thread_id`, `user_message`, `conversation_history`
- [ ] `app/retrieval/graphs/assistant_chat_graph.py`: ChatState + Nodes 1-5 + 8 (ohne Verifikation)
- [ ] `app/retrieval/chains/intent_classification_chain.py`: Zero-shot LLM-Klassifikation
- [ ] `app/retrieval/chains/retrieval_plan_chain.py`: INTENT_CHUNK_TYPE_MAP
- [ ] `app/retrieval/chains/chat_answer_chain.py`: RAG-Kontext + Chat-History; Persona via `load_system_prompt()` (existiert bereits)
- [ ] `app/retrieval/services/chat_orchestrator_service.py`: duenner Service-Layer
- [ ] `app/retrieval/prompts/chat_answer.prompt`: Chat-spezifische User-Prompt-Vorlage (Persona-System-Prompt liegt bereits in ragkeep)
- [ ] Event-Logging: Chat-Events analog `GraphEventRecorder` loggen
- [ ] Manueller E2E-Test ueber curl / httpie

### Iteration 2 – Qualitaet (ca. 3-4 Tage)

**Ziel:** Grounding-Check, Confidence-Score, defensiver Antwortmodus.

- [ ] `app/retrieval/chains/grounding_verification_chain.py`: Node 6
- [ ] `app/retrieval/utils/sufficiency.py`: extrahierte Sufficiency-Logik
- [ ] `app/retrieval/services/citation_service.py`: Chunk-ID → Quellen-Metadaten
- [ ] Confidence-Score-Berechnung im `finalize_response`-Node
- [ ] `app/db/migrations/versions/0008_add_chat_threads.py` + Thread-Persistenz
- [ ] LLM-Reranker als optionaler Node 4b

### Iteration 3 – Robustheit (ca. 4-5 Tage)

**Ziel:** Multi-Intent, Conversation Memory, Quote-Einbindung, Streaming.

- [ ] Multi-Intent-Routing: INTENT_CHUNK_TYPE_MAP per Union-Merge
- [ ] `app/retrieval/services/memory_service.py`: Thread-Summary nach N Turns
- [ ] Quote-Chunks als Ergaenzungs-Retrieval nach Hauptantwort
- [ ] Streaming-Support (SSE) am Chat-Endpunkt
- [ ] Lokaler Embedding-Classifier fuer Intent (multilingual-e5-small)

### Iteration 4 – Optimierung (ca. 3-5 Tage)

**Ziel:** Metriken, Kostenoptimierung, Classifier-Finetuning.

- [ ] Metriken: hallucination_rate, citation_coverage, intent_accuracy aus Chat-Logs
- [ ] Classifier-Finetuning mit annotierten Chat-Logs
- [ ] Latenz-Profiling; ggf. Parallelisierung von Retrieval und Intent-Klassifikation
- [ ] Kosten-Dashboard (DeepSeek-API-Calls pro Intent-Typ)

---

## 15. Offene Entscheidungen

1. **Intent-Klassifikation strikt lokal vs. Hybrid mit LLM-Fallback?**
   Empfehlung: Start mit LLM (Iter. 1), Umstieg auf lokalen Classifier sobald ~50 annotierte
   Beispiele aus echten Logs vorliegen (Iter. 2).

2. **Essay-Material bei belegpflichtigen Antworten?**
   `essay`-Chunks sind Interpretationstext, kein Primaerbeleg. Empfehlung: nur bei
   `erklaerung_vertiefung` und `vergleich` als Sekundaerquelle; bei `beleg_pruefung` explizit
   ausschliessen.

3. **Citation-Tiefe: nur Werk/Vortrag oder bis Chunk-Ebene?**
   Die `metadata`-Spalte in `rag_chunks` enthaelt bereits Quelleninfos. Empfehlung: Mindest-
   anforderung = Werk + Datum; optional Kapitel/Seite wenn in Metadaten vorhanden.

4. **Maximal-Latenz?**
   Intent + Retrieval + LLM = realistisch 1.0-2.5s ohne Streaming. Mit Streaming: erste
   Token nach ~0.5s. Empfehlung: Streaming ab Iteration 3 als Pflicht, Ziel P95 < 3s.

5. **Wo wird `conversation_history` gehalten – Client oder Server?**
   Iteration 1: Client haelt History und sendet sie mit. Iteration 2+: Server haelt History
   in `chat_threads`-Tabelle, Client sendet nur `thread_id`.

6. **Assistenten-uebergreifend oder Philo-spezifisch?**
   Der `assistant_slug` im Endpunkt-Pfad (`/agent/{assistant_slug}/chat`) erlaubt von Anfang
   an Multi-Assistenten-Support. Nur der Persona-Prompt und die Collection-ID aendern sich.
