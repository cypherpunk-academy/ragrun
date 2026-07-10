# ragrun REST API — Übersicht & Sicherheitsreview

Erstellt: 2026-06-21
Server: FastAPI (Python async) · Port 8000 · Prefix-Strategie: `/app/*` für App, `/api/v1/*` für CLI/Admin

---

## 1. Architektur

ragrun hat zwei klar getrennte API-Bereiche:

| Bereich | Prefix | Auth | Genutzt von |
|---------|--------|------|-------------|
| App-API | `/app/*` | Supabase JWT | ragapp (iOS/Android) |
| RAG-API | `/api/v1/rag/*` | — | ragprep (CLI) |
| Admin-API | `/api/v1/admin/*` | — | Admin-Tools, CLI |
| Agent-API | `/api/v1/agent/*` | — | Web-Frontend, CLI |

---

## 2. App-API — Alle Endpunkte (`/app/*`)

Diese Gruppe ist die einzige, die ragapp nutzt. **Alle Endpunkte außer Health und Personalities erfordern ein gültiges Supabase-JWT.**

| Method | Pfad | Auth | Funktion |
|--------|------|------|----------|
| GET | `/app/health` | Nein | Heartbeat (online/offline-Check in App) |
| GET | `/app/personalities` | Nein | Verfügbare KI-Persönlichkeiten |
| POST | `/app/search` | JWT | Semantische/Volltextsuche |
| GET | `/app/sources` | JWT | Alle verfügbaren Bücher/Quellen |
| GET | `/app/sources/{id}/segments` | JWT | Kapitelstruktur einer Quelle |
| GET | `/app/chunks/{id}` | JWT | Volltext eines einzelnen Chunks |
| POST | `/app/chat` | JWT | KI-Gespräch senden |
| POST | `/app/chat/{talk_id}/summarize` | JWT | Gespräch zusammenfassen |
| POST | `/app/sync/pull` | JWT | WatermelonDB-Sync: Änderungen holen |
| POST | `/app/sync/push` | JWT | WatermelonDB-Sync: Lokale Änderungen senden |

### Request/Response-Schema (wichtigste Endpunkte)

**POST /app/search**
```json
// Request
{ "query": "Freiheit des Denkens", "types": ["talk"], "limit": 20, "collection": "philo" }

// Response
{ "results": [{ "chunk_id": "...", "text": "...", "score": 0.87, "source_title": "...", "segment_title": "..." }] }
```

**POST /app/chat**
```json
// Request
{ "message": "Was meint Steiner mit reinem Denken?", "personality": "philo-von-freisinn",
  "talk_id": "uuid-oder-null", "context_mode": "auto", "context_ids": [] }

// Response
{ "talk_id": "uuid", "turn_id": "uuid", "reply": "..." }
```

**POST /app/sync/pull**
```json
// Request
{ "last_pulled_at": 1718000000, "schema_version": 16 }

// Response (WatermelonDB-Format)
{ "changes": { "paragraphs": { "created": [...], "updated": [...], "deleted": [...] } }, "timestamp": 1718001000 }
```

---

## 3. RAG-API (`/api/v1/rag/*`) — CLI/Ingestion

Genutzt von ragprep für Datenaufbereitung und Embedding. Kein Auth.

| Method | Pfad | Funktion |
|--------|------|----------|
| POST | `/api/v1/rag/store-chunks` | Chunks in DB schreiben (JSONL) |
| POST | `/api/v1/rag/embed-chunks` | Chunks in Qdrant einbetten |
| POST | `/api/v1/rag/deprecate-chunk-ids` | Chunks als veraltet markieren |
| POST | `/api/v1/rag/delete-chunk-ids` | Chunks gezielt löschen |
| POST | `/api/v1/rag/delete-chunks` | Chunks per Filter löschen |
| POST | `/api/v1/rag/list-chunks` | Chunk-Inventar auflisten |
| POST | `/api/v1/rag/quote-explain` | Zitat erklären |
| GET | `/api/v1/rag/books/titles` | Buchtitel in Collection |
| GET | `/api/v1/rag/books/chapters` | Kapitel einer Quelle |
| GET | `/api/v1/rag/books/context-chunks` | Kontext-Chunks eines Segments |
| GET | `/api/v1/rag/collections` | Alle Qdrant-Collections |
| GET | `/api/v1/rag/collections/{name}/verify-sparse` | Sparse-Index prüfen |
| GET | `/api/v1/rag/monitoring/chunks` | Chunk-Statistiken |
| GET | `/api/v1/rag/talks/published` | Publizierte Gespräche |

---

## 4. Admin-API (`/api/v1/admin/*`)

Verwaltung von Inhalten und Nutzern. Kein Auth.

| Method | Pfad | Funktion |
|--------|------|----------|
| GET | `/api/v1/admin/collections` | Collection-Statistiken |
| GET | `/api/v1/admin/stats` | Systemstatistiken (Chunks, Talks, Usage) |
| GET | `/api/v1/admin/talks` | Alle Gespräche (filterbar) |
| GET | `/api/v1/admin/talks/{id}` | Gesprächsdetail |
| PATCH | `/api/v1/admin/turns/{id}` | Gesprächsrunde bearbeiten |
| PATCH | `/api/v1/admin/talks/{id}` | Gesprächsstatus ändern (publishing_status) |
| POST | `/api/v1/admin/users/upsert` | Nutzer anlegen/aktualisieren |

---

## 5. Agent-API (`/api/v1/agent/*`)

Web-Frontend und erweiterte Workflows. Kein Auth.

| Method | Pfad | Funktion |
|--------|------|----------|
| POST | `/api/v1/agent/{slug}/chat/stream` | SSE-Stream: Chat |
| GET | `/api/v1/agent/{slug}/chat/thread/{id}` | Gesprächsverlauf |
| GET | `/api/v1/agent/{slug}/actions` | Verfügbare Aktionen |
| POST | `/api/v1/agent/{slug}/generate-prompt` | Prompt generieren (mit Retrieval) |
| POST | `/api/v1/agent/{slug}/execute-prompt` | Gecachten Prompt ausführen (SSE) |
| POST | `/api/v1/agent/{slug}/problem-solver` | Sokratischer Dialog (SSE) |
| POST | `/api/v1/agent/philo-von-freisinn/graphs/concept-explain-worldviews` | Begriffserklärung über Weltanschauungen |

---

## 6. Auth-Mechanismus (Supabase JWT)

**Implementierung:** `app/api/auth.py`

### Ablauf
```
ragapp               ragrun               Supabase
  |--- POST /app/chat --|                    |
  |    Bearer: <jwt>    |                    |
  |                     |-- JWKS fetch ----->|
  |                     |<-- public keys ----|
  |                     | verify(jwt, keys)  |
  |                     | check exp + aud    |
  |                     | extract sub → user_id
  |<--- 200 reply ------|
```

### JWT-Verifikation
- Algorithmen: RS256, ES256, EdDSA (via JWKS), HS256 (Fallback mit `SUPABASE_JWT_SECRET`)
- JWKS-Endpunkt: `{SUPABASE_URL}/auth/v1/.well-known/jwks.json`
- JWKS-Cache: 300 Sekunden
- Required Claims: `sub` (user_id), `exp` (Ablaufzeit), `aud: "authenticated"`

### ragapp-seitig
- Token-Quelle: `supabase.auth.getSession()` → bei Fehler: `refreshSession()`
- Header: `Authorization: Bearer {token}`
- Public Endpoints (kein Token): `/app/health`, `/app/personalities`

---

## 7. Middleware & Cross-Cutting

| Aspekt | Status | Detail |
|--------|--------|--------|
| CORS | Konfigurierbar | `RAGRUN_CORS_ORIGINS` (kommasepariert); erlaubte Methods: GET, POST, PATCH |
| Rate Limiting | Nicht implementiert | Kein Throttling auf FastAPI-Ebene |
| Request-Logging | Minimalistisch | Standard-Logging; kein strukturiertes Access-Log |
| Input-Validierung | Pydantic | Alle Request-Bodies via Pydantic-Modelle |
| Error-Format | FastAPI-Standard | `{"detail": "Fehlermeldung"}` bei HTTP-Fehlern |

---

## 8. Konfiguration & Secrets

Alle Einstellungen über Env-Variablen mit Prefix `RAGRUN_` (`.env`-Datei wird gelesen):

| Variable | Sensitivität | Zweck |
|----------|-------------|-------|
| `RAGRUN_SUPABASE_URL` | Niedrig | Supabase-Projekt-URL |
| `RAGRUN_SUPABASE_ANON_KEY` | Niedrig | Öffentlicher Anon-Key |
| `RAGRUN_SUPABASE_JWT_SECRET` | **Hoch** | Symmetrischer JWT-Verifikationsschlüssel |
| `RAGRUN_POSTGRES_DSN` | **Hoch** | DB-Verbindung incl. Passwort |
| `RAGRUN_QDRANT_API_KEY` | **Hoch** | Qdrant-Zugang |
| `RAGRUN_DEEPSEEK_API_KEY` | **Hoch** | LLM-API-Key (Kosten!) |
| `RAGRUN_LANGFUSE_SECRET_KEY` | Mittel | Telemetrie |
| `RAGRUN_CORS_ORIGINS` | Niedrig | Erlaubte Browser-Origins |

---

## 9. Sicherheitsreview

### 9.1 Stärken

**App-API gut abgesichert**
Alle datensensitiven `/app/*`-Endpunkte (Sync, Chat, Suche) erfordern ein gültiges Supabase-JWT. Die Verifikation ist korrekt implementiert mit JWKS-Rotation und Algorithm-Fallback. ragapp sendet Tokens korrekt und refresht sie automatisch.

**Secrets nicht im App-Bundle**
ragapp enthält nur den Supabase Anon-Key und die ragrun-URL (beide unkritisch). Der Service-Role-Key ist nicht gebündelt.

**JWT-Implementierung solide**
JWKS-basierte Verifikation, Expiry-Check, Audience-Validierung (`aud: "authenticated"`), 5-Minuten-Cache für JWKS.

---

### 9.2 Risiken

#### KRITISCH — Unauthentifizierte Admin- und RAG-Endpunkte

Die `/api/v1/rag/*`- und `/api/v1/admin/*`-Endpunkte haben **keine Authentifizierung**. Wer den Server erreichen kann, kann:

- Alle Chunks löschen oder überschreiben (`DELETE /api/v1/rag/delete-chunks` mit `all=true`)
- Alle Nutzerdaten und Gespräche lesen (`GET /api/v1/admin/talks`)
- Nutzerdaten manipulieren (`PATCH /api/v1/admin/turns/{id}`)
- Beliebige Nutzer anlegen (`POST /api/v1/admin/users/upsert`)

**Einschätzung:** Das ist kein Bug, sondern Design — ragrun ist als persönlicher Selbst-Host-Dienst konzipiert, nicht als öffentlich exponierter Server. Die RAG/Admin-Routes sind interne CLI-Interfaces. **Solange ragrun nur im Heimnetz oder via VPN/Tunnel erreichbar ist, ist das akzeptabel.**

**Risiko wird kritisch wenn:** ragrun über eine öffentliche IP oder einen öffentlichen Reverse-Proxy (ohne weitere Absicherung) erreichbar ist.

#### MITTEL — Kein Rate Limiting

Die `/app/chat`- und `/app/search`-Endpunkte sind JWT-geschützt, aber ohne Throttling. Ein gültiger Token kann beliebig viele LLM-Calls triggern. Kosten-DoS durch einen kompromittierten Account ist möglich.

**Empfehlung:** Nginx- oder Traefik-Rate-Limiting vor FastAPI (z. B. 30 req/min per `sub`-Claim).

#### NIEDRIG — CORS zu weit offen

`allowed_headers: "*"` erlaubt alle Custom-Header. Für eine mobile App irrelevant (CORS gilt nur im Browser), aber für Web-Frontend-Nutzung beachtenswert.

#### NIEDRIG — X-Account-Id nicht verifiziert

Die Agent-Endpunkte akzeptieren `X-Account-Id` als optionalen Header für Logging. Dieser wird nicht verifiziert — jeder kann eine beliebige Account-ID senden. Nur relevant wenn Usage-Tracking nach Account-ID ausgewertet wird.

---

### 9.3 Empfehlungen

| Priorität | Maßnahme |
|-----------|----------|
| P1 (wenn öffentlich) | Nginx-Basis-Auth oder IP-Whitelist für `/api/v1/*` |
| P1 (wenn öffentlich) | Firewall: Port 8000 nur intern erreichbar, öffentlich nur `/app/*` via Reverse-Proxy |
| P2 | Rate Limiting für `/app/chat` und `/app/search` per JWT-`sub` |
| P3 | Strukturiertes Access-Log (wer ruft welchen Endpunkt wie oft auf) |
| P3 | `allowed_headers` in CORS auf konkrete Liste einschränken |

---

### 9.4 Deployment-Modell (aktuell angenommen)

```
Internet
   |
   +-- [Öffentlich] --> ragrun /app/* (JWT-geschützt) ✓
   |
Heimnetz / VPN
   |
   +-- [Privat] --> ragrun /api/v1/* (kein Auth) -- CLI/Admin-Zugang
```

Dieses Modell ist sicher, solange der Netzwerk-Perimeter hält. Für Produktivbetrieb mit mehreren Nutzern oder öffentlichem Zugang müssten `/api/v1/*` explizit abgesichert werden.
