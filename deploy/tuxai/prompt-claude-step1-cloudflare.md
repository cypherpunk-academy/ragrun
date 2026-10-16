# Claude-Prompt: Schritt 1 (Cloudflare + Tuxai)

Zwei Varianten — je nach Maschine **einen** Block kopieren und in Claude einfügen.

**Regeln für Claude:** Keine Secrets in Chat/Commits. Client ID/Secret nur in 1Password/`rotate.env` (600). Tunnel-Connector-Token nur auf Tuxai. Referenz: `deploy/tuxai/plan-tuxai-cloudflare-railway.md`, `deploy/tuxai/cloudflare-tunnel.md`.

**DNS (fix):** Zwei **CNAME** bei Chris — **keine** Cloudflare-Zone nötig.

| Hostname | → CNAME |
|----------|---------|
| `qdrant.ragxxx.com` | `<tunnel-uuid>.cfargotunnel.com` |
| `embed.ragxxx.com` | dasselbe |

---

## Variante A — Mac (Cloudflare + Abnahme)

```markdown
Du hilfst mir bei Schritt 1: Cloudflare Tunnel + Access für Tuxai (Qdrant + Embed).

## Hostnames (fix)
- `qdrant.ragxxx.com` → `http://127.0.0.1:6333`
- `embed.ragxxx.com` → `http://127.0.0.1:8001`
- Tunnel `ragrun-tuxai` ist healthy; Published routes noch anlegen falls „No routes“.

## DNS
Chris setzt zwei CNAME auf dasselbe `<uuid>.cfargotunnel.com`. staging/api/MX unverändert.

## Access
Zwei Apps (Self-hosted), Service Auth only, Tokens `staging-tuxai` und `prod-tuxai`.

## Test
Nach CNAME: `verify-tunnel.sh` im ragrun-Repo mit QDRANT_API_KEY + CF-Token.

Repo: /Users/michael/Reniets/Ai/ragrun
```

---

## Variante B — Tuxai (cloudflared)

```markdown
Schritt 1 auf Tuxai: cloudflared für Tunnel `ragrun-tuxai`, Origins 127.0.0.1:6333 und :8001.

Hostnames in CF: `qdrant.ragxxx.com`, `embed.ragxxx.com` (Published application).

Prüfe Docker/Qdrant/Embed lokal, systemd cloudflared, kein doppelter Connector.

Repo-Beispiel: deploy/tuxai/cloudflared/config.yml.example
```

---

## Nach Schritt 1

Access-Tokens → `rotate.env` → `apply.sh` (Plan Schritt 3).
