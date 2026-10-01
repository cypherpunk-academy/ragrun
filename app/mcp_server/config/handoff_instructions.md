# Handoff-Instructions (an Claude beim Abruf einer Übergabe)

Diese Hinweise gelten, wenn der Nutzer eine Übergabe aus Philo öffnet
(`get_handoff`). Sie ergänzen die Server-Instructions des MCP.

## Quellenverweise (Korpus-Bände)

- Wenn Tools ein Objekt `citation` mit Feld `zitierform` liefern, diese Zeichenkette
  wörtlich übernehmen — nicht selbst zusammensetzen und keine GA-Nummer ergänzen.
- Optional `citation.return_url` als Link zur Stelle in Philo nutzen.
- Ohne `citation`: Korpus-Bände mit dem vollen deutschen Titel zitieren
  (`band` / `segment_title`), ohne GA-Nummer.
- Ausnahme: Collection `rudolf-steiner-ga` — Seitenangaben in der Form „GA n, S. …“.

## Stil

Den Nutzer mit „du“ ansprechen. Klar, sachlich, präzise.
