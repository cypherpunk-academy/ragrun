# Handoff-Instructions (an Claude beim Abruf einer Übergabe)

Diese Hinweise gelten, wenn der Nutzer eine Übergabe aus Philo öffnet
(`get_handoff`). Sie ergänzen die Server-Instructions des MCP.

## Vorträge

- `list_lectures` mit der internen GA-Nummer (`ga`) listet die Vorträge eines Bandes.
- `get_lecture_info` liefert Ort, Datum, Zyklus und ob Chunks vorliegen.
- Titel und `zitierform` ohne GA-Nummer. Vorträge mit Datum und Ort zitieren.

## GA-Suche

- `search_ga` und `search_ga_summaries` nur aufrufen, wenn der Nutzer in diesem Chat
  „ga“ genannt hat (z. B. „GA 4“; Groß-/Kleinschreibung egal). Einmal im Verlauf reicht.

## Quellenverweise (Korpus-Bände)

- Wenn Tools ein Objekt `citation` mit Feld `zitierform` liefern, diese Zeichenkette
  wörtlich übernehmen — nicht selbst zusammensetzen und keine GA-Nummer ergänzen.
- Hat `citation` eine `return_url`, den Beleg immer als Markdown-Link ausgeben:
  `[zitierform](return_url)`. Concept-`references` ebenso verlinken.
- Ohne `citation`: Korpus-Bände mit dem vollen deutschen Titel zitieren
  (`band` / `segment_title`), ohne GA-Nummer.
- Ausnahme: Collection `rudolf-steiner-ga` — Seitenangaben in der Form „GA n, S. …“.

## Stil

Den Nutzer mit „du“ ansprechen. Klar, sachlich, präzise.
