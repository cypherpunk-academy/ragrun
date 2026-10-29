# Embeddings-Hosting — Strategie

Stand: 2026-07-27

Modal `personal-embeddings-service` ist **gestoppt** (Ursache: `min_containers=1` → dauerhafte T4-Kosten ~$14/Tag). Scale-to-zero mit Cold Start ist für die UX nicht akzeptabel; Railway always-on CPU (~$60–120/Monat) ist teurer als die bisherigen Planungen.

## Aktuell (jetzt)

**Hugging Face Serverless** für Query-Embeddings; **Ingest lokal** (Laptop / Docker `personal-embeddings-service`).

| | |
|---|---|
| Modell | unverändert `intfloat/multilingual-e5-large` (1024d) — kein Re-Embed |
| Queries | HF Inference Providers / `hf-inference` (feature-extraction) |
| Ingest (`rag:embed`, 50k Chunks) | lokal — nicht über HF-Free-Credits |
| Last-Annahme | selten ~5 User, ~alle 30 s → ~10 Anfragen/min Peak, machbar |
| Kosten | Free $0.10 / PRO $2 Credits/Monat, danach Pay-as-you-go (klein bei Low Traffic) |
| Risiken | Cold Start, 429/503, kein SLA; Vektor-Kompatibilität einmal gegen lokalen Embedder prüfen |
| Modal | bleibt `stopped` |

Siehe Umsetzungsskizze: [HF_SERVERLESS_EMBEDDINGS.md](HF_SERVERLESS_EMBEDDINGS.md)

## Hinterhand (future)

**GPU-Mini-PC / SFF** always-on mit Docker `personal-embeddings-service`, erreichbar per Tailscale oder Cloudflare Tunnel. Queries + Ingest auf demselben Host.

Siehe: [HOME_GPU_EMBEDDER.md](HOME_GPU_EMBEDDER.md)

## Verworfene / nicht priorisierte Optionen

- Modal T4 warm (`min_containers=1`) — zu teuer
- Modal scale-to-zero — Cold Start inakzeptabel
- Railway CPU always-on — ~$60–120/Monat
- Raspberry Pi — nur Query möglich (16 GB), Ingest unangenehm; kein fertiges Repo-Setup
- DeepSeek v4 lokal neben Embedder — v4-Flash braucht ~90–180 GB VRAM; weiter API nutzen

## DeepSeek

Chat/Reasoning bleibt bei **DeepSeek API** (`deepseek-v4-flash`). Lokales DeepSeek v4 ist kein Mini-PC-Ziel.
