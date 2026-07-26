# Drei-Stufen-Deployment-Architektur

Stand: 2026-07-09

## Ziel

| Stufe | Infrastruktur | Supabase | Zweck |
|-------|--------------|----------|-------|
| **development** | lokal (Docker Compose) | free-account „dev" | lokales Entwickeln, Hot-Reload |
| **staging** | Railway (eigene Environment) | free-account „staging" | Integrationstests, QA |
| **production** | Railway (eigene Environment) | Pro-Account | Endnutzer |

---

## 1. Supabase – Eignung und Einschränkungen

### 1.1 Free-Tier-Constraints (Stand mid-2026)

| Merkmal | Free | Konsequenz |
|---------|------|-----------|
| DB-Größe | 500 MB | ausreichend für dev/staging |
| Projekte pro Org | 2 | braucht 2 Orgs oder 2 Accounts → s.u. |
| **Inaktivitäts-Pause** | **nach 7 Tagen ohne API-Zugriff** | **kritisch für staging** |
| Simultane Verbindungen (Pooler) | ~50 | für staging unkritisch |
| PITR / Branching | nein | kein automatisches Rollback |
| Custom Domains | nein | kein Problem für APIs |
| Erreichbarkeit nach Pause | **manuelles Restore im Dashboard nötig** (~1–2 min) | nicht selbstheilend auf Free |

### 1.2 Inaktivitäts-Pause: Staging-Risiko und Lösung

Das größte Problem für Staging ist die 7-Tage-Pause. Staging-Umgebungen liegen typischerweise tagelang brach.

**Empfehlung: Keepalive-Cronjob**

Ein Railway-Cron-Service (oder externer Cron) ruft täglich einmal den Supabase-Health-Endpoint des Staging-Projekts auf. Damit bleibt das Projekt aktiv.

```
# Beispiel: Railway Cron Service (täglich 06:00 UTC)
curl -s https://<staging-project-ref>.supabase.co/rest/v1/ \
  -H "apikey: <staging-anon-key>" > /dev/null
```

Alternativ: Railway hat einen eingebauten `Cron`-Service-Typ. Darin reicht ein minimales `curl`-Image.

**Staging-QA-Urteil: vernünftig möglich**, solange:
- Der Keepalive läuft
- Die Tester wissen, dass ein versehentlich pausiertes Projekt erst im Dashboard manuell wiederhergestellt werden muss
- Keine Produktionsdaten in Staging gespiegelt werden (eigenes Schema/Daten)

### 1.3 Account-Struktur (2 free + 1 pro)

Supabase erlaubt 2 free Projekte pro **Organisation**. Lösung:

```
Supabase-Account „reniets-dev"   (eigene free Org)
  └── Projekt: ragrun-dev

Supabase-Account „reniets-staging" (eigene free Org, oder 2. Projekt in derselben Org)
  └── Projekt: ragrun-staging

Supabase-Account „reniets" (bestehend, Pro)
  └── Projekt: ragrun-production  ← aktuell genutzt
```

Einfachste Option: 1 Account mit 2 Orgs, oder 2 separate Accounts (beide kostenlos).

### 1.4 Schema-Synchronisation

Der Supabase-Migrations-Workflow (supabase CLI / `supabase/migrations/`) muss auf alle drei Projekte angewendet werden:

```bash
supabase db push --project-ref <dev-ref>
supabase db push --project-ref <staging-ref>
supabase db push --project-ref <prod-ref>
```

ragprep-Skripte, die Supabase-Tabellen schreiben, lesen ihre Verbindungsdaten bereits aus Env-Variablen – d.h. kein Code muss geändert werden, nur `.env.development`, `.env.staging`, `.env.production`.

---

## 2. Railway – Drei-Stufen-Infrastruktur

### 2.1 Railway Environments (empfohlener Ansatz)

Railway unterstützt **mehrere Environments pro Projekt** nativ. Ein Environment ist eine vollständige, isolierte Kopie aller Services mit eigenem Variablen-Set, eigenen Volumes und eigenem Deploy-State.

```
Railway-Projekt: ragrun
  ├── Environment: production   ← aktuell (api + ui)
  └── Environment: staging      ← NEU
```

Vorteil gegenüber separaten Projekten:
- Einheitliche Service-Definitionen
- Variablen per Environment überridebar
- Deployments unabhängig pro Branch/Environment

### 2.2 Services pro Environment

Aktuell hat Railway nur `api` und `ui`. Qdrant und Embeddings laufen lokal.

Zielzustand:

```
Railway Environment: production
  ├── api          (Dockerfile aus Repo-Root, 24/7)
  └── ui           (Next.js, NIXPACKS, ./ui, 24/7)

Railway Environment: staging  ← Smoke-Tests, Sleep-Modus
  ├── api-staging      (schläft wenn nicht genutzt)
  └── qdrant-staging   (Image: qdrant/qdrant:v1.11.0, Volume, schläft wenn nicht genutzt)

Extern (beide Environments):
  ├── Qdrant Cloud     (Production-Vektordaten, Free Tier, managed)
  └── Modal            (Embeddings, serverless GPU, pay-per-call, staging + production)
```

Staging hat **keine** ui und kein eigenes Modal-Deployment – es nutzt denselben Modal-Endpoint wie Production. Staging testet ausschließlich: deployt der Code, starten die Services, antworten die Endpoints korrekt.

Sleep-Modus bedeutet: Railway fährt den Service beim ersten Aufruf hoch (~30–60 s Kaltstart), danach läuft er bis zur nächsten Inaktivitätsphase. Kosten entstehen nur während aktiver Nutzung.

### 2.3 Railway-spezifische Überlegungen

**qdrant-staging auf Railway (nur Staging):**
- Image-Service: `qdrant/qdrant:v1.11.0`, persistentes Volume für `/qdrant/storage`
- Kein Public Port nötig – api-staging erreicht qdrant-staging via `*.railway.internal`
- Sleep-Modus: schläft bei Inaktivität, Volume bleibt erhalten
- Initial mit Snapshot aus Production-Qdrant (Qdrant Cloud) befüllt (Entscheidung 1b)
- Production nutzt stattdessen Qdrant Cloud (kein Railway-Qdrant in Production)

**Embeddings (staging + production): Modal, kein Railway-Service**
- Der `personal-embeddings-service` wird als Modal-App deployt, nicht auf Railway
- Staging und Production teilen denselben Modal-Endpoint (serverless, per-call)
- Kein Railway-Volume, kein Kaltstart-Problem für das Modell

**Kein Railway-Template für docker-compose.yml:** Railway liest keine `docker-compose.yml`. Jeder Service wird separat in Railway konfiguriert. Die bestehende `docker-compose.yml` dient als lokale Referenz, nicht als Deployment-Artefakt.

### 2.4 Environment-Variablen pro Service

```
# api – staging (Railway Environment: staging)
RAGRUN_APP_ENV=staging
RAGRUN_QDRANT_URL=http://qdrant-staging.railway.internal:6333
RAGRUN_EMBEDDINGS_BASE_URL=https://<modal-endpoint>.modal.run
RAGRUN_POSTGRES_DSN=postgresql+psycopg://...@<staging-supabase>.pooler.supabase.com:5432/postgres
RAGRUN_SUPABASE_URL=https://<staging-ref>.supabase.co
RAGRUN_SUPABASE_ANON_KEY=<staging-anon-key>
RAGRUN_SUPABASE_JWT_SECRET=<staging-jwt-secret>

# api – production (Railway Environment: production)
RAGRUN_APP_ENV=production
RAGRUN_QDRANT_URL=https://<cluster-id>.us-east4-0.gcp.cloud.qdrant.io:6333
RAGRUN_QDRANT_API_KEY=<qdrant-cloud-api-key>
RAGRUN_EMBEDDINGS_BASE_URL=https://<modal-endpoint>.modal.run
RAGRUN_POSTGRES_DSN=postgresql+psycopg://...@<prod-supabase>.pooler.supabase.com:5432/postgres
RAGRUN_SUPABASE_URL=https://<prod-ref>.supabase.co
RAGRUN_SUPABASE_ANON_KEY=<prod-anon-key>
RAGRUN_SUPABASE_JWT_SECRET=<prod-jwt-secret>
```

Staging nutzt `railway.internal` für qdrant (Service im selben Railway-Projekt).
Production nutzt den öffentlichen Qdrant-Cloud-Endpoint (HTTPS + API-Key).

### 2.5 railway.toml – Anpassungsbedarf

Die aktuelle `railway.toml` definiert `api` und `ui`. Für Staging kommt nur `qdrant-staging` als weiterer Railway-Service hinzu (Image-Service, direkt im Dashboard angelegt). Embeddings entfällt als Railway-Service vollständig.

---

## 3. ragapp – Eine App pro Stufe

### 3.1 Aktuelle Architektur

ragapp ist eine Expo/React Native App. Sie verbindet sich mit:
- Supabase (Auth, Daten via EXPO_PUBLIC_SUPABASE_URL)
- ragrun-API (via konfigurierbare Backend-URL)

### 3.2 EAS Build Profile (empfohlener Ansatz)

Expo Application Services (EAS) erlaubt Build-Profile in `eas.json`:

```json
{
  "build": {
    "development": {
      "developmentClient": true,
      "distribution": "internal",
      "env": {
        "EXPO_PUBLIC_SUPABASE_URL": "https://<dev-ref>.supabase.co",
        "EXPO_PUBLIC_SUPABASE_ANON_KEY": "<dev-anon-key>",
        "EXPO_PUBLIC_RAGRUN_API_URL": "http://localhost:8000/api/v1"
      }
    },
    "staging": {
      "distribution": "internal",
      "env": {
        "EXPO_PUBLIC_SUPABASE_URL": "https://<staging-ref>.supabase.co",
        "EXPO_PUBLIC_SUPABASE_ANON_KEY": "<staging-anon-key>",
        "EXPO_PUBLIC_RAGRUN_API_URL": "https://api-staging.up.railway.app/api/v1"
      }
    },
    "production": {
      "distribution": "store",
      "env": {
        "EXPO_PUBLIC_SUPABASE_URL": "https://<prod-ref>.supabase.co",
        "EXPO_PUBLIC_SUPABASE_ANON_KEY": "<prod-anon-key>",
        "EXPO_PUBLIC_RAGRUN_API_URL": "https://api.up.railway.app/api/v1"
      }
    }
  }
}
```

Das ergibt drei unterschiedliche **App-Builds** (je eine `.ipa`/`.apk` pro Stufe), die intern verteilt werden können (via Expo Go / TestFlight für staging, App Store für production).

### 3.3 Offline-Entwicklung (development)

Im `development`-Profil zeigt die App auf `localhost:8000`. Das funktioniert für Simulator/Emulator direkt. Für physische Geräte braucht man die lokale IP oder einen Tunnel (ngrok/cloudflared).

### 3.4 Offene Frage: ragrun-UI (Next.js) vs. ragapp

Die ragrun UI (`/ui`, Next.js) und ragapp (Expo) sind zwei separate Frontends. Beide müssen pro Tier konfiguriert werden:

- **ragrun/ui**: Next.js-Env-Variablen via Railway Environment Variables (automatisch getrennt)
- **ragapp**: EAS Build-Profile wie oben beschrieben

Für den Staging-Test der UI reicht es, den Railway-Staging-Environment-URL zu nutzen.

---

## 4. Development (lokal) – bestehender Workflow

Development bleibt wie bisher lokal:

```
docker compose up        # startet qdrant + embeddings + api
# oder: start-dev.sh / build-dev.sh
```

Die lokale `.env` zeigt auf:
- Supabase-dev-Projekt (NEU – bisher wahrscheinlich direkt Prod)
- lokalen Qdrant (localhost:6333)
- lokalen Embeddings-Service (localhost:8001)

Änderung gegenüber heute: `.env` wird auf das neue `ragrun-dev`-Supabase-Projekt umgestellt.

---

## 5. Umsetzungsreihenfolge

```
Phase 1 – Supabase trennen
  [ ] Neues Supabase-Projekt ragrun-dev anlegen
  [ ] Migrations auf dev ausführen
  [ ] Lokale .env auf dev-Projekt umstellen
  [ ] Neues Supabase-Projekt ragrun-staging anlegen
  [ ] Migrations auf staging ausführen
  [ ] Keepalive-Cron für staging konfigurieren (Railway Cron-Service)

Phase 2 – Externe Production-Services aufsetzen (Modal + Qdrant Cloud)
  [ ] Qdrant Cloud Free-Tier-Cluster anlegen
  [ ] Qdrant-Snapshot aus lokalem Docker in Qdrant Cloud einspielen
  [ ] personal-embeddings-service als Modal-App deployen (modal deploy)
  [ ] Modal-Endpoint-URL notieren → wird in Phase 3 + 4 gesetzt

Phase 3 – Railway Production auf externe Services umstellen
  [ ] RAGRUN_QDRANT_URL → Qdrant Cloud Endpoint + API-Key setzen
  [ ] RAGRUN_EMBEDDINGS_BASE_URL → Modal-Endpoint-URL setzen
  [ ] Alle Production-Variablen von docker-compose.yml zu Railway Variables migrieren
  [ ] Hartcodierte Postgres-DSN in docker-compose.yml durch Variable ersetzen
  [ ] Smoke-Test Production: api health, Embedding-Call, Qdrant-Query

Phase 4 – Railway Staging-Environment
  [ ] Railway-Environment „staging" anlegen
  [ ] api-staging deployen (develop-Branch, Sleep-Modus aktivieren)
  [ ] qdrant-staging als Railway Image-Service hinzufügen (qdrant/qdrant:v1.11.0, Volume, Sleep-Modus)
  [ ] Qdrant-Snapshot aus Qdrant Cloud in qdrant-staging einspielen
  [ ] Staging-Umgebungsvariablen setzen:
      RAGRUN_QDRANT_URL        → qdrant-staging.railway.internal:6333
      RAGRUN_EMBEDDINGS_BASE_URL → Modal-Endpoint-URL (gleicher wie Production)
      RAGRUN_POSTGRES_DSN / SUPABASE_* → staging-Supabase-Projekt
  [ ] Sleep-Verhalten testen: Service schläft, wacht bei erstem Request auf (~30–60 s)
  [ ] Smoke-Test Staging: api health, /api/v1/rag/books/titles, Embedding-Call

Phase 5 – ragapp EAS
  [ ] eas.json mit 3 Build-Profilen anlegen (development / staging / production)
  [ ] app.config.ts auf env-basierte Konfiguration umstellen (falls noch nicht)
  [ ] Staging-Build intern testen (Expo Go)
  [ ] Production-Build konfigurieren

Phase 6 – ragprep CLI
  [ ] .env.development / .env.staging / .env.production anlegen
  [ ] Sicherheitsnetz: production-env nur explizit per source laden
```

---

## 6. Entscheidungen

| # | Frage | Entscheidung |
|---|-------|-------------|
| 1 | Wie werden Staging-Qdrant-Daten initial befüllt? | **Snapshot aus Production kopieren** |
| 2 | Welcher Git-Branch triggert Staging-Deploy? | **`develop`-Branch** |
| 3 | Wie wird ragapp Staging verteilt? | **Expo Go + staging-Build** |
| 4 | Braucht ragrun-UI eine eigene Staging-Domain? | **`.railway.app`-Subdomain genügt** |
| 5 | ragprep Staging/Production-Targeting | **`.env.staging` / `.env.production` in ragprep** – explizit laden, nie automatisch; verhindert versehentliche Production-Writes |

### Hinweis zu ragprep

ragprep wird nicht deployed. Es läuft lokal und richtet sich via Env-Variablen, welches Environment es befüllt:

- schreibt Metadaten direkt in Supabase (via Postgres-DSN / Supabase API)
- schickt Chunks an ragrun-API (via `RAGRUN_BASE_URL`) → ragrun embedded + Qdrant-Upsert

```bash
# Staging befüllen (explizit)
source .env.staging
rp embed --book ...

# Production befüllen (explizit)
source .env.production
rp embed --book ...
```

---

## 7. Kostenschätzung (empfohlene Architektur)

### Production

| Service | Anbieter | Kosten/Monat |
|---------|----------|-------------|
| api | Railway (bestehend) | ~$4 |
| ui | Railway (bestehend) | ~$2 |
| qdrant | **Qdrant Cloud Free Tier** | $0 |
| embeddings | **Modal (serverless GPU, T4)** | ~$0–2 (pay-per-call) |
| Supabase | Pro-Account | bestehend |
| **Production gesamt** | | **~$6–8/Monat** |

Railway Hobby ($5/Monat inkl. Credit) reicht damit weiterhin. Modal-Kosten bei niedrigem Traffic praktisch $0.

### Staging (Sleep-Modus)

| Service | Anbieter | Kosten/Monat |
|---------|----------|-------------|
| api-staging | Railway (schläft) | ~$0.20 bei ~10 h Nutzung |
| qdrant-staging | Railway (schläft) | ~$0.20 |
| embeddings-staging | Modal (serverless) | ~$0 |
| Supabase staging | Free-Account + Keepalive-Cron | $0 |
| **Staging gesamt** | | **<$1/Monat** |

### Gesamtbild

| Stufe | Kosten/Monat |
|-------|-------------|
| Development | $0 (lokal) |
| Staging | <$1 |
| Production | ~$6–8 |
| Supabase Pro (Production) | bestehend |
| **Total neu** | **~$7–9/Monat** |

### Hinweise

- **Modal**: Der `personal-embeddings-service` wird als Modal-Funktion deployt. Abrechnung pro Sekunde GPU-Nutzung. Bei wenigen Nutzern bleibt das im einstelligen Dollar-Bereich pro Monat.
- **Qdrant Cloud Free Tier**: 1 GB RAM-Cluster, managed, kein eigener Service auf Railway nötig. Für Production-Datenvolumen prüfen ob 1 GB reicht; nächste Stufe ist ~$25/Monat.
- **Staging-Qdrant auf Railway** (statt Qdrant Cloud) wegen isolierter Staging-Daten (Snapshot aus Production, Entscheidung 1b).
