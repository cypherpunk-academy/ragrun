# Plan: Sokratischer Dialog – Verfeinerung von Manifest und Prompt

## Ziel

Den sokratischen Dialog zu einem echten philosophischen Prüfinstrument machen:
- Der Assistent prüft das Gesagte des Users gegen die **primären Quellen** (Steiner et al.)
- Er findet aktiv **Unstimmigkeiten, Denkfehler, Unklarheiten**
- Er fragt nach — sokratisch, nicht belehrend
- Er bleibt **quellengebunden**, aber spricht mit Witz und Tiefe

---

## 1. `action-manifest.yaml` – Änderungen

### Ist-Zustand (Probleme)

| Query | Problem |
|-------|---------|
| `primary` | `chunk_types: [book]` — korrekt, aber `k: 6` könnte zu wenig sein für Prüfung |
| `secondary` | `method: dense` — für Ideen/Gedankengänge ok, aber `essay` und `talk` sollten voll dabei sein |
| `quotes` | `k: 6` — für Würze zu viele; max. 2–3 Zitate sinnvoll |
| `concepts` | `method: lemma-lookup` — gut, aber `k` fehlt explizit |

### Soll-Zustand (Anpassungen)

```yaml
id: socratic-dialogue
label: Sokratischer Dialog
description: Prüfender Dialog – findet Denkfehler, Unklarheiten und Widersprüche zu den Primärquellen.

requires_retrieval: true
requires_prompt: true

queries:
  - name: primary
    chunk_types: [book]
    k: 8
    method: hybrid
    # Primärwissen: Steiner's Originalwerke. Antworten DÜRFEN dem nicht widersprechen.
    # Höheres k (8) damit genug Kontext zum Prüfen da ist.

  - name: secondary
    chunk_types: [secondary_book, essay, talk]
    k: 6
    method: dense
    # Hilfreiche Ideen und Gedankengänge zur Frage. Ergänzend, nicht normativ.

  - name: quotes
    chunk_types: [quote]
    k: 3
    method: dense
    # Zitate als Würze — bewusst wenige, damit das Beste ausgewählt wird.

  - name: concepts
    chunk_types: [begriff_list]
    method: lemma-lookup
    # Begriffsdefinitionen: helfen dem LLM, den User präzise zu verstehen.

follow_ups:
  - type: detail
    question: "Soll ich einen bestimmten Widerspruch vertiefen?"
  - type: summary
    question: "Sollen wir die Kernpunkte unseres Gesprächs zusammenfassen?"
    condition: "turn_count >= 3"
  - type: socratic_continue
    question: "Möchtest du deinen Standpunkt nochmals formulieren?"
```

**Kernänderungen:**
- `k` bei `primary` auf **8** erhöht (mehr Prüfmaterial)
- `k` bei `quotes` auf **3** reduziert (Qualität statt Quantität)
- Labels und Beschreibung präzisiert
- Follow-up angepasst: sokratisch weiterführend statt neutral

---

## 2. `prompt.prompt` – Änderungen

### Ist-Zustand (Probleme)

- Variablen `{thesis}` und `{counter}` matchen **nicht** die Query-Namen im Manifest
  - Manifest: `primary`, `secondary`, `quotes`, `concepts`
  - Prompt: `{thesis}`, `{counter}` → falsch verbunden, funktioniert ggf. nicht
- Ton: nüchtern, akademisch, nicht sokratisch
- Keine explizite Anweisung zu: nachfragen, Denkfehler benennen, Witz

### Soll-Zustand (neuer Prompt)

Der Prompt soll:
1. **Variable-Namen** mit Manifest-Queries synchronisieren: `{primary}`, `{secondary}`, `{quotes}`, `{concepts}`
2. Den **sokratischen Charakter** explizit einfordern:
   - Fragen stellen, nicht nur antworten
   - Unstimmigkeiten benennen und zurückfragen
   - Denkfehler freundlich aber klar aufzeigen
3. **Quellenbindung** differenzieren: Primary ist Maßstab, Secondary ist Hilfe
4. **Ton**: tiefgründig, witzig, ein bisschen provokativ — wie Sokrates selbst

```
{user_prompt}
-------------------------------

Du führst einen sokratischen Dialog. Deine Aufgabe ist es nicht, einfach zu antworten —
sondern zu prüfen, nachzufragen und gemeinsam mit dem User zur Klarheit zu kommen.

Konkret: Prüfe das Gesagte des Users auf:
- **Unstimmigkeiten** gegenüber den Primärquellen (s.u.)
- **Denkfehler** oder logische Sprünge
- **Unklarheiten** in Begriffen oder Argumenten
- **Widersprüche** zu dem, was die Quellen eigentlich sagen

Wenn du etwas findest: Benenne es direkt, aber mit Witz — nicht als Lehrer, sondern als
neugieriger Gesprächspartner. Stelle eine klärende Gegenfrage. Lass den User selbst denken.

---

--- Primäre Quellen (Maßstab) ---
**Das ist das Kernwissen. Antworten dürfen dem nicht widersprechen. Wenn der User etwas sagt,
das hier widerlegt wird — zeige es auf.**
{primary}

--- Sekundäre Quellen (Ergänzung) ---
**Hilfreiche Ideen und Gedankengänge. Nutze sie zur Vertiefung, nicht als Urteil.**
{secondary}

--- Begriffe ---
**Präzisiere damit dein Verständnis des Users — nicht für die Antwort, sondern für dich.**
{concepts}

--- Zitate (Würze) ---
**Maximal ein Zitat, das den Kern des Gesprächs trifft. Eher am Ende als am Anfang.**
{quotes}

---

Konversationskontext (falls vorhanden):
{conversation_context}

---

Antworte sokratisch: eine These, eine Frage, vielleicht ein Witz. Nicht zu lang.
```

**Kernänderungen:**
- Variable-Namen korrigiert: `{primary}`, `{secondary}`, `{quotes}`, `{concepts}`
- Explizite Liste was zu prüfen ist: Unstimmigkeiten, Denkfehler, Unklarheiten, Widersprüche
- Klare Rollendefinition: nicht belehren, sondern fragen
- Ton-Direktive: Witz, Neugier, Sokrates-Charakter
- Kürze eingefordert: "eine These, eine Frage, vielleicht ein Witz"

---

## 3. Umsetzungsschritte

- [x] `action-manifest.yaml`: k-Werte anpassen, `explanation` in secondary, Follow-ups überarbeiten, Beschreibung präzisieren
- [x] `prompt.prompt`: Komplett neu geschrieben mit korrekten Variablen (`{primary}`, `{secondary}`, `{quotes}`, `{concepts}`) und sokratischem Ton
- [x] `action_prompt_service.py`: `thesis`/`counter` aus `all_placeholder_names` entfernt, `primary`/`secondary` hinzugefügt (beide Stellen: retrieval-path + no-retrieval-path)
- [x] `action_prompt.py`: `thesis`/`counter` aus `allowed_slots` entfernt, `primary`/`secondary` hinzugefügt
- [ ] Testen: Einen Dialog mit einer absichtlich falschen These führen → prüft ob der Assistent widerspricht
- [ ] Testen: Unklaren Begriff einwerfen → prüft ob der Assistent nachfragt

---

## 4. Offene Fragen

- **Wie werden `{thesis}` / `{counter}` aktuell im Code gemappt?** → Prüfen ob die falschen Variablennamen bereits einen Bug verursachen oder ob das Mapping dynamisch ist.
- **Soll der sokratische Dialog auch `explanation`-Chunks einbeziehen?** (aktuell nicht im Manifest) — könnte hilfreich sein für Begriffserklärungen die Steiner selbst gibt.
- **Follow-up `socratic_continue`**: Ist dieser Typ im System bereits registriert? Falls nicht, evtl. auf bestehenden Typ mappen.
