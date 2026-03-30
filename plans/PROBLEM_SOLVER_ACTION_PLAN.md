# Plan: Aktion „problem-solver" – Automatisierter Sokratischer Lösungsdialog

## Ziel

Der `problem-solver` nimmt **TEXT 2** aus einer vorangegangenen `summarize`-Aktion
(offene Probleme, ungelöste Spannungen) und versucht, das darin formulierte Problem
in einem automatisierten, sokratischen Mehrschrittverfahren zu lösen –
**ohne den User zu fragen**. Nach jedem LLM-Aufruf meldet er einen einzeiligen Digest
per SSE an den Client.

---

## Ablauf (Flowchart)

```
Client sendet TEXT 2 + assistant_slug
           │
           ▼
  ┌─────────────────────┐
  │  Schritt 1: Solver  │  ← problem-solver.prompt + RAG-Retrieval
  │  „Wie könnte man    │
  │   das lösen?"       │
  └────────┬────────────┘
           │  digest → Client  (SSE: "step": 1, "role": "solver", "digest": "...")
           ▼
  ┌─────────────────────┐
  │  Schritt 2: Sokrates│  ← socratic-dialog.prompt + RAG-Retrieval
  │  Prüft, hinterfragt,│
  │  stimmt zu / nicht  │
  └────────┬────────────┘
           │  digest → Client  (SSE: "step": 2, "role": "socrates", "digest": "...")
           │
           ├── Zustimmung erkannt? → DONE (Abschluss-Event)
           │
           ▼
  [Schritt 3–7: Wechsel Solver / Sokrates]
           │
           └── Nach Schritt 7 oder Zustimmung → DONE
```

**Max. Iterationen:** 7 (= max. 4 Solver-Runden + 3–4 Sokrates-Runden im Wechsel)
**Frühzeitiger Abschluss:** Sobald Sokrates-Antwort als Zustimmung klassifiziert wird.

---

## 1. Neue Dateien

### 1.1 `app/retrieval/actions/problem-solver/action-manifest.yaml`

```yaml
id: problem-solver
label: Problem lösen
description: Löst offene Probleme in einem automatisierten sokratischen Dialog (max. 7 Schritte). Ohne Prompt wird TEXT 2 der letzten summarize-Aktion verwendet.

requires_retrieval: true    # Solver und Sokrates brauchen beide RAG
requires_prompt: false      # Kein Prompt erforderlich – TEXT 2 wird als Fallback genutzt
allows-empty-prompt: true   # Leerer Prompt → TEXT 2 aus summarize-Kontext

position-in-chat: [continue, end]

queries:
  - name: primary
    chunk_types: [book]
    k: 6
    method: hybrid

  - name: secondary
    chunk_types: [secondary_book, essay, talk, explanation]
    k: 4
    method: dense

  - name: quotes
    chunk_types: [quote]
    k: 2
    method: dense

  - name: concepts
    chunk_types: [begriff_list]
    method: lemma-lookup

follow_ups:
  - action_id: save-dialog
  - action_id: give-feedback
```

### 1.2 `app/retrieval/actions/problem-solver/solver.prompt`

Prompt für die **Solver-Runden** (Schritt 1, 3, 5, 7).

```
Du analysierst ein ungelöstes philosophisches oder gesellschaftliches Problem und
schlägst einen konkreten Lösungsweg vor.

Problem:
{problem_text}

Bisheriger Dialogverlauf:
{dialog_history}

---

Primäre Quellen:
{primary}

Sekundäre Quellen:
{secondary}

Begriffe:
{concepts}

Zitate:
{quotes}

---

Formuliere einen klaren, argumentativ begründeten Lösungsvorschlag (3–5 Sätze).
Beziehe dich auf die Quellen. Kein Metakommentar, keine Einleitung – direkt zur Sache.
```

### 1.3 `app/retrieval/actions/problem-solver/socrates.prompt`

Prompt für die **Sokrates-Runden** (Schritt 2, 4, 6).

```
Du bist Sokrates. Du prüfst den vorliegenden Lösungsvorschlag kritisch.

Problem:
{problem_text}

Lösungsvorschlag:
{solver_response}

Bisheriger Dialogverlauf:
{dialog_history}

---

Primäre Quellen (Maßstab):
{primary}

Sekundäre Quellen:
{secondary}

Begriffe:
{concepts}

---

Wenn der Vorschlag überzeugend und widerspruchsfrei ist: Stimme explizit zu.
Beginne deine Antwort dann mit: „Ich stimme zu."

Andernfalls: Benenne den entscheidenden Einwand in einem Satz. Stelle eine einzige
klärende Gegenfrage. Kein Mehrfachfragen. Nicht zu lang.
```

---

## 2. Neues Graph-Modul

### `app/retrieval/graphs/problem_solver_graph.py`

**State-Felder:**

| Feld | Typ | Bedeutung |
|------|-----|-----------|
| `problem_text` | `str` | `user_prompt` wenn gefüllt, sonst TEXT 2 aus summarize |
| `broad_context` | `str` | RAG-Ergebnis Schritt 1: primary + secondary + quotes + concepts |
| `primary_context` | `str` | RAG-Ergebnis Sokrates-Runden: nur primary (k=6) |
| `dialog_history` | `list[dict]` | Wechsel: `{role: solver/socrates, text: ...}` |
| `current_step` | `int` | 1–7 |
| `agreed` | `bool` | True wenn Sokrates zugestimmt hat |
| `collection` | `str` | RAG-Collection |
| `assistant_slug` | `str` | Für Instruction-Prompt |

**Graph-Knoten:**

```
__start__
    │
    ▼
retrieve_broad            ← Einmalig: primary + secondary + quotes + concepts
    │                       Ergebnis wird in State gespeichert (broad_context)
    ▼
retrieve_primary          ← Einmalig: nur primary (k=6, hybrid)
    │                       Ergebnis in State (primary_context)
    ▼
solver_step               ← Nutzt broad_context; füllt solver.prompt; max_tokens=300
    │  SSE-Digest (erster Satz)
    ▼
check_max_steps           ← current_step >= 7 → __end__
    │
    ▼
socrates_step             ← Nutzt primary_context; füllt socrates.prompt; max_tokens=300
    │  SSE-Digest (erster Satz)
    ▼
check_agreement           ← Antwort beginnt mit „Ich stimme zu."?
    ├── ja → __end__ (agreed=True)
    └── nein → solver_step (nächste Iteration)
```

**Zustimmungs-Erkennung:**

Einfaches String-Matching: Sokrates-Antwort beginnt mit `„Ich stimme zu"` (case-insensitive,
mit/ohne Anführungszeichen). Kein extra LLM-Call nötig.

---

## 3. Neuer API-Endpoint

### `POST /agent/{assistant_slug}/problem-solver`

**Request:**
```json
{
  "assistant_slug": "philo-von-freisinn",
  "user_prompt": "",           // leer → TEXT 2 aus letzter summarize-Aktion
  "problem_text": "...",       // vom Client vorausgefüllt wenn user_prompt leer
  "language": "de-DE"
}
```

**Eingabe-Logik (server-seitig):**

```python
if user_prompt.strip():
    problem_text = user_prompt          # User hat ein eigenes Thema eingegeben
else:
    problem_text = body.problem_text    # TEXT 2 aus summarize (vom Client übergeben)
```

Der Client befüllt `problem_text` immer aus dem zuletzt gespeicherten TEXT 2 der
`summarize`-Aktion. Gibt der User etwas ein, überschreibt `user_prompt` diesen Wert.

**Response:** SSE-Stream

```
data: {"type": "start"}

data: {"type": "digest", "step": 1, "role": "solver",
       "digest": "Vorschlag: Dreigliederung als ethische Vorentscheidung, nicht als äußere Institution."}

data: {"type": "digest", "step": 2, "role": "socrates",
       "digest": "Einwand: Kann die innere Haltung ohne äußere Entsprechung dauerhaft tragen?"}

data: {"type": "digest", "step": 3, "role": "solver",
       "digest": "Präzisierung: Die Dreigliederung als innere Orientierung setzt keine äußere Struktur voraus."}

data: {"type": "digest", "step": 4, "role": "socrates",
       "digest": "Ich stimme zu. Die innere Dreigliederung als Vorbedingung der äußeren ist überzeugend."}

data: {"type": "done", "agreed": true, "steps": 4,
       "full_dialog": [...],
       "final_answer": "Die innere Dreigliederung..."}
```

**Router-Datei:** `app/api/problem_solver.py`
**Einbinden:** in `app/api/__init__.py` oder `main.py` wie die anderen Router.

---

## 4. SSE-Digest-Format

Jeder `digest`-Event hat:

| Feld | Typ | Beschreibung |
|------|-----|-------------|
| `type` | `"digest"` | Event-Typ |
| `step` | `int` | 1–7 |
| `role` | `"solver"` \| `"socrates"` | Wer hat gesprochen |
| `digest` | `str` | Einzeiliger Satz, max. 120 Zeichen |

Der `digest` wird **nicht** extra generiert, sondern ist der **erste Satz** der LLM-Antwort
(bis zum ersten `.`, `!`, oder `?`), gekürzt auf 120 Zeichen.

---

## 5. Integration in `summarize` follow_ups

`summarize/action-manifest.yaml` bekommt `problem-solver` als Follow-up,
nur wenn TEXT 2 vorhanden ist (Bedingung im Client prüfbar):

```yaml
follow_ups:
  - action_id: problem-solver
    condition: "has_problem_text"   # Client-seitige Bedingung
  - action_id: save-dialog
  - action_id: give-feedback
```

---

## 6. Implementierungsreihenfolge

| Schritt | Was | Datei(en) |
|---------|-----|-----------|
| 1 | Action-Manifest + Prompt-Dateien anlegen | `actions/problem-solver/*` |
| 2 | State-Modell definieren | `retrieval/models.py` |
| 3 | Graph implementieren | `retrieval/graphs/problem_solver_graph.py` |
| 4 | API-Endpoint + SSE-Streaming | `api/problem_solver.py` |
| 5 | Router registrieren | `main.py` / `api/__init__.py` |
| 6 | `summarize` follow_ups erweitern | `actions/summarize/action-manifest.yaml` |
| 7 | Manueller Test mit einem echten TEXT 2 | — |

---

## 7. Designentscheide (festgelegt)

| Frage | Entscheid |
|-------|-----------|
| **RAG-Strategie** | Schritt 1 (Solver) nutzt breites Retrieval (primary + secondary + quotes + concepts). Sokrates-Runden (Schritte 2, 4, 6) nutzen nur `primary` (k=6, hybrid). Spart Tokens und hält Sokrates am Maßstab der Primärquellen. |
| **Max. Tokens pro Schritt** | 300 Tokens — via `max_tokens` am LLM-Client für jeden Schritt. |
| **Übergabe von problem_text** | Direkt im Request-Body des `/problem-solver`-Endpoints (kein Prompt-State-Cache). |
| **Digest** | Erster Satz der LLM-Antwort (bis zum ersten `.`, `!`, `?`), gekürzt auf 120 Zeichen. Kein separater LLM-Call. |
