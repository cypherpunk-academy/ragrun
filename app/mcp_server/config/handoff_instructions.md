# Handoff-Instructions (an Claude beim Abruf einer Übergabe)

Diese Hinweise gelten, wenn der Nutzer eine Übergabe aus Philo öffnet
(`get_handoff`). Sie ergänzen die Server-Instructions des MCP.

## Quellenverweise (Korpus-Bände)

- Hat ein Treffer `citation`, ist der Beleg genau `[zitierform](return_url)`.
- `zitierform` wörtlich übernehmen. Nicht umbauen, Ort und Datum nicht in Klammern
  anhängen, keine GA-Nummer ergänzen.
- Ohne `citation`: voller deutscher Titel (`band` / `segment_title`), ohne GA-Nummer.
- `search_ga` und `get_ga_chunk` haben keine `return_url`. Kurzbeleg nur
  `(GA <n>, S. <seite>)`. Keinen Link erfinden.

## Vorträge

- `list_lectures` mit der internen GA-Nummer (`ga`) listet die Vorträge eines Bandes.
- `get_lecture_info` liefert Ort, Datum, Zyklus und ob Chunks vorliegen.

## GA

- `get_ga_chunk(chunk_id)`: bekannten GA-Chunk nachladen, wenn die `chunk_id` vorliegt.
- `search_ga` / `search_ga_summaries`: nur nach ausdrücklichem Nutzerwunsch, in der
  Gesamtausgabe nachzuschauen.

## Stil

Den Nutzer mit „du“ ansprechen. Klar, sachlich, präzise.
