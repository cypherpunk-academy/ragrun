# Handoff-Instructions (an Claude beim Abruf einer Übergabe)

Diese Hinweise gelten, wenn der Nutzer eine Übergabe aus Philo öffnet
(`get_handoff`). Sie ergänzen die Server-Instructions des MCP.

## Vorträge

- `list_lectures` mit der internen GA-Nummer (`ga`) listet die Vorträge eines Bandes.
- `get_lecture_info` liefert Ort, Datum, Zyklus und ob Chunks vorliegen.
- Titel und `zitierform` ohne GA-Nummer. Vorträge mit Datum und Ort zitieren.

## Quellenverweise (Korpus-Bände)

- Wenn Tools ein Objekt `citation` mit Feld `zitierform` liefern, diese Zeichenkette
  wörtlich übernehmen — nicht selbst zusammensetzen und keine GA-Nummer ergänzen.
- Optional `citation.return_url` als Link zur Stelle in Philo nutzen.
- Ohne `citation`: Korpus-Bände mit dem vollen deutschen Titel zitieren
  (`band` / `segment_title`), ohne GA-Nummer.
- Ausnahme: Collection `rudolf-steiner-ga` — Seitenangaben in der Form „GA n, S. …“.

## Stil

Den Nutzer mit „du“ ansprechen. Klar, sachlich, präzise.
