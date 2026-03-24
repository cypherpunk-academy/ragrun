# Plan: Aktionen als vollwertige Chat-Flow-Einheiten

## Problem

Die aktuellen `follow_ups` in den Manifesten sind lose Typ-Strings (`type: detail`,
`type: quote_request`) ohne direkte Verbindung zu konkreten Aktionen. Damit ist kein
strukturierter Chat-Flow möglich, und der Client muss die Logik selbst erfinden.

## Leitidee

**Jedes Follow-up ist eine vollwertige Aktion** — referenziert per `action_id`.
Die Aktionen selbst definieren, wo sie im Chat stehen dürfen (`position-in-chat`)
und ob der User-Prompt leer sein darf (kontextbasiertes Retrieval).

---

## 1. Neue Felder im `action-manifest.yaml`

### `position-in-chat`

```yaml
position-in-chat: [start, continue]   # Wo darf diese Aktion aufgerufen werden?
```

| Wert       | Bedeutung |
|------------|-----------|
| `start`    | Kann einen Chat eröffnen (wird im Startmenü angeboten) |
| `continue` | Kann nach einer anderen Aktion folgen (Follow-up) |
| `end`      | Schließt den Chat ab (kein weiterer Follow-up außer `give-feedback`) |

Aktionen können mehrere Positionen haben: `[start, continue]`, `[continue, end]`.

### `allows-empty-prompt`

```yaml
allows-empty-prompt: true
```

Wenn `true`: Der User-Prompt darf leer sein. Das System nutzt dann den
Konversationskontext für das Retrieval. Relevant für `find-quote` und `find-in-works`
als Follow-up ("Finde Zitate dazu!" ohne weitere Eingabe).

### `follow_ups` → Liste von `action_id`-Referenzen

```yaml
follow_ups:
  - action_id: socratic-dialog        # default (erster Eintrag)
  - action_id: find-quote
  - action_id: find-in-works
  - action_id: summarize
  - action_id: give-feedback
```

Der erste Eintrag ohne Bedingung ist der **Default** — wird im Client vorausgewählt.
Bedingte Einträge erscheinen erst ab einer bestimmten Turn-Zahl:

```yaml
follow_ups:
  - action_id: socratic-dialog
  - action_id: find-quote
    allows-empty-prompt: true         # Override für diesen spezifischen Follow-up-Kontext
  - action_id: summarize
    condition: "turn_count >= 3"
  - action_id: give-feedback
```

---

## 2. Aktionsübersicht (neu)

| Alte ID           | Neue ID           | Label (DE)                      | position-in-chat    | Global Default |
|-------------------|-------------------|---------------------------------|---------------------|:--------------:|
| `general-question`| `general-question`| Allgemeine Frage                | `[start, continue]` | ✓ (Chat-Start) |
| `clarify-concept` | `clarify-concept` | Begriff klären                  | `[start, end]`      |                |
| `find-quote`      | `find-quote`      | Zitat suchen                    | `[start, continue]` |                |
| `locate-in-works` | `find-in-works`   | In Werken suchen                | `[start, continue]` |                |
| `socratic-dialog` | `socratic-dialog` | Sokratischer Dialog             | `[start, continue]` |                |
| `thanks-feedback` | `give-feedback`   | Bewertung geben                 | `[continue, end]`   |                |
| *(neu)*           | `summarize`       | Zusammenfassen                  | `[continue, end]`   |                |

**Hinweis zur Umbenennung:** `locate-in-works` → `find-in-works` und
`thanks-feedback` → `give-feedback` spiegeln den Charakter der Aktion besser wider.
Die Umbenennung erfordert: neues Verzeichnis, Manifest-id, Frontend-Update, ggf. Redirect.

### Neue Aktion: `summarize`

```yaml
id: summarize
label: Zusammenfassen
description: Fasst das bisherige Gespräch strukturiert zusammen.
requires_retrieval: false
requires_prompt: false
position-in-chat: [continue, end]
allows-empty-prompt: true

follow_ups:
  - action_id: give-feedback
```

Kein Retrieval. Das LLM bekommt nur `{conversation_context}` und erstellt eine
Zusammenfassung. Schließt den inhaltlichen Teil des Chats ab.

---

## 3. Follow-up-Matrix

Wer kann auf wen folgen (✓ = sinnvoll, D = Default):

|                   | general-q | find-quote | find-in-works | socratic | summarize | give-feedback |
|-------------------|:---------:|:----------:|:-------------:|:--------:|:---------:|:-------------:|
| **general-q**     | ✓         | ✓          | ✓             | ✓        | ✓         | ✓             |
| **clarify**       | —         | —          | —             | —        | —         | D             |
| **find-quote**    | ✓         | D          | ✓             | ✓        | ✓         | ✓             |
| **find-in-works** | ✓         | ✓          | D             |          | ✓         | ✓             |
| **socratic**      |           | ✓          | ✓             | D        | ✓         | ✓             |
| **summarize**     |           |            |               |          |           | D             |
| **give-feedback** | —         | —          | —             | —        | —         | —             |

`D` = Default (erster Follow-up, wird vorausgewählt).
`clarify-concept` hat nur `give-feedback` als Follow-up (Chat-Ende).
`give-feedback` und `summarize` haben keine inhaltlichen Follow-ups.

---

## 4. Leertext-Retrieval (`allows-empty-prompt`)

Wenn der User-Prompt leer ist, soll das Retrieval trotzdem funktionieren.
Der Service muss dann auf den Konversationskontext zurückfallen:

```python
effective_query = user_prompt.strip() or extract_last_topic(conversation_context)
```

`extract_last_topic`: Nimmt die letzte Frage/Antwort aus dem Konversationskontext
als Retrieval-Query. Einfachste Implementierung: letzten Satz / letzten User-Turn nehmen.

---

## 5. Client-Logik (Konzept)

```
Chat öffnet
  → zeige alle Aktionen mit position-in-chat: start
  → Default vorausgewählt: general-question

User wählt Aktion A, gibt Prompt ein (oder leer bei allows-empty-prompt)
  → Aktion ausführen

Nach Antwort
  → lade follow_ups von Aktion A
  → zeige als Chips; erster Eintrag = Default (vorausgewählt)
  → User tippt frei ohne Chip zu wählen → Default-Aktion wird verwendet
  → User wählt Chip mit allows-empty-prompt → Prompt-Feld ausblenden / optional
  → Aktion hat position-in-chat: end → Chat als abgeschlossen markieren,
    keine weiteren Eingaben außer give-feedback
```

---

## 6. Umsetzungsschritte

### Phase 1 – Manifest-Schema erweitern
- [ ] `position-in-chat` zu allen bestehenden Manifesten hinzufügen
- [ ] `allows-empty-prompt` zu `find-quote` und `find-in-works` (als Follow-up relevant)
- [ ] `follow_ups` auf `action_id`-Referenzen umstellen (alle Manifeste)

### Phase 2 – Neue Aktionen
- [ ] `summarize` erstellen: Verzeichnis, Manifest, Prompt
- [ ] `give-feedback` (Umbenennung von `thanks-feedback`): Verzeichnis umbenennen oder Alias

### Phase 3 – Service-Anpassung
- [ ] `action_prompt_service.py`: Leertext-Fallback auf Konversationskontext
- [ ] `list_actions()`: `position-in-chat` im Response mitliefern
- [ ] `generate-prompt` Response: `follow_ups` als aufgelöste Aktion-Objekte zurückgeben
  (nicht nur IDs, sondern `{id, label, description, allows_empty_prompt}`)

### Phase 4 – Umbenennung (optional, breaking)
- [ ] `locate-in-works` → `find-in-works`
- [ ] `thanks-feedback` → `give-feedback`
- [ ] Frontend und Tests anpassen

---

## 7. Entscheidungen

- **Rückwärtskompatibilität**: Wird nicht berücksichtigt. Harter Schnitt auf das neue Schema.
- **Freie Eingabe nach Follow-up**: Es muss immer ein Default definiert sein. Tippt der User
  frei ohne Aktion zu wählen, wird der Default verwendet. Für den Chat-Beginn (kein vorheriger
  Kontext) ist `general-question` der globale Default.
- **`clarify-concept`**: Beendet den Chat. Position: `[start, end]` — kann einen Chat beginnen,
  schließt ihn aber ab. Kein `continue`. Follow-up ist nur `give-feedback`.
- **Mehrsprachigkeit**: Nur Deutsch. Ein `label`-Feld pro Manifest reicht.
