# App-Tools (Filo)

Tool-Registry für den App-Chat (`/app/chat/stream`). Siehe
`plans/ragrun-app-tools-architecture.md` (ragapp-Repo) für die volle
Architektur und `plans/filo-arbeitstext-contract.md` für den
ragapp↔ragrun-Vertrag.

App-Tools sind **nicht** dasselbe wie:
- **Personality-Actions** (`app/retrieval/actions/…` bzw. `ragrun-personalities/<id>/`) — für `ragprep rag:chat`.
- **Helper-Actions** (`app/retrieval/helper-actions/…`) — z. B. `summarize`.

## Konvention: ein Verzeichnis pro Tool

```
app/tools/
  types.py            # ToolContext, ToolResult, ToolManifest
  limits.py            # MAX_DOCUMENT_CHARS (Spiegel von documentLimits.ts)
  document_tree.py      # Read-only Parser (Spiegel von documentTree.ts) — nur für read_blocks
  registry.py           # AppToolRegistry: discover / list_tools / schemas_for_llm / invoke
  index.py              # register_all_tools() — aufgerufen aus app/main.py (lifespan)
  app/
    create_document/
      tool-manifest.yaml
      schema.json
      handler.py
      tests/
    read_blocks/
      ...
    update_document/
      ...
```

Jedes Tool-Verzeichnis unter `app/tools/app/<tool_id>/` braucht:
- `tool-manifest.yaml` — `id`, `label`, `description`, `category`, `execution`
  (`client`|`server`), `result_key`, optional `availability.requires_linked_document`
  (`true`/`false`/weggelassen) und `availability.modes`.
- `schema.json` — JSON-Schema der Tool-Argumente (an das Modell als
  Function-Calling-Parameter durchgereicht).
- `handler.py` — `async def run(ctx: ToolContext, args: dict) -> ToolResult`.
- `__init__.py` — macht das Verzeichnis zu einem importierbaren Package
  (Handler wird dynamisch via `importlib` geladen).

## Discovery

`AppToolRegistry.discover()` glob't `app/tools/app/*/tool-manifest.yaml`,
lädt `schema.json` daneben und importiert `handler.py` als
`app.tools.app.<tool_id>.handler`. Registrierung erfolgt in
`app/main.py` (`lifespan`) → `app.state.app_tool_registry`.

## Tests

Pro Tool unter `app/tools/app/<tool_id>/tests/`. `pytest.ini` hat
`testpaths = tests` (Top-Level) — Tool-Tests werden von einem nackten
`pytest`-Lauf **nicht** automatisch gefunden. CI/lokal explizit aufrufen:

```
pytest app/tools/app/*/tests/
```

## execution: client vs. server

- `client` — ragapp wendet das Ergebnis lokal an (WatermelonDB/Supabase),
  z. B. `create_document`, `update_document`. Der Handler in ragrun
  validiert nur und reicht das Ergebnis über `tool_results` im SSE-`done`-
  Event zurück.
- `server` — ragrun schreibt selbst (Postgres `rag_*`), z. B. ein
  zukünftiges `compress_talk`. Noch nicht implementiert (Welle ≥3).
