# Philo-MCP-Tests (Ebene A)

Live-Tests gegen den Dev-Server (ngrok) oder Staging. Tokens werden bei Bedarf
über ``scripts/mcp_tests/mint.py`` gemintet. Secrets stehen nur in der Umgebung
oder in gitignored Env-Dateien, nie im Repo.

Der Server spricht **Streamable HTTP** (`app/mcp_server/server.py`,
`streamable_http_app`). Die Client-URL ist `{MCP_BASE_URL}/mcp`.

## Variablen

| Variable | Bedeutung |
|---|---|
| `MCP_TOKEN_USER_A` / `MCP_TOKEN_USER_B` | optional; sonst Mint über Service-Role |
| `MCP_BASE_URL` | Dev/Staging-Basis-URL. Sonst `RAGRUN_MCP_BASE_URL` |
| `RAGRUN_POSTGRES_DSN` | für Seed, Handoffs, Cleanup (steht in `.env.dev`) |
| `RAGRUN_SUPABASE_URL` / `*_ANON_KEY` / `*_SERVICE_ROLE_KEY` | für Token-Mint |

`tests/.env.mcp`, `.env.dev` und `.env` werden gelesen; bereits gesetzte Variablen bleiben.

## Skripte (`scripts/mcp_tests/`)

```bash
python scripts/mcp_tests/seed_fixtures.py   # [FIXTURE]-Notizen anlegen
python scripts/mcp_tests/test_handoffs.py   # frische [TEST]-Handoffs
python scripts/mcp_tests/test_cleanup.py    # nur [TEST] löschen
python scripts/mcp_tests/mint.py --user a   # Token minten (gibt nur die E-Mail aus)
```

Cursor-Anschluss (stdio-Proxy, Token nur im Speicher):

```bash
.venv/bin/python scripts/mcp_tests/proxy.py --env dev --user a
.venv/bin/python scripts/mcp_tests/proxy.py --env staging --user antonia
```

Einträge in [`.cursor/mcp.json`](../../../.cursor/mcp.json): `philo-dev-anton`,
`philo-dev-antonia`, `philo-staging-anton`, `philo-staging-antonia`.

## Ebene A

```bash
pytest tests/mcp/level_a
pytest tests/mcp/level_a --mcp-id W-01
```

Ein gewöhnliches `pytest` im Repo lässt diese Tests aus. Vor dem Lauf: Cleanup,
Seed der `[FIXTURE]`-Notizen, frische Handoffs. Am Ende wieder Cleanup für
`[TEST]`. G-02 läuft vor den A-Tests.

Bericht: `tests/reports/ebene_a_<datum>.md`.

V-04 und C-07 sind mit `xfail(strict=True)` markiert (Befunde F-3, F-4).
S-06 (F-1) und G-03 (F-2) laufen ohne Markierung, wenn der Server das Soll erfüllt.

A-03, C-03 und U-04 sind im Testplan offen. Der Test besteht, wenn der Server
den Wert speichert oder strukturiert ablehnt, und schreibt die Beobachtung in die Bemerkung.
S-02 erwartet 0 Treffer (Stand 1.10). A-07 und C-06 prüfen das Soll „Fehlermeldung“.

Claude und andere Sprachmodelle sind nicht Teil dieser Suite.
