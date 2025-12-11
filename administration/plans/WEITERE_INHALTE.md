## Weitere Inhalte (Ergänzungen zu PLAN.md)

Dieses Dokument bündelt inhaltliche Bausteine, die in `PLAN.md` noch nicht beschrieben sind. Es dient als Kontext- und Funktionsentwurf für inhaltliche „Karten“, Zusammenfassungen und Grundbegriffe im Umfeld von `ragkeep`/`ragprep`/`ragrun`.

### 1) Umfang in ragkeep
- Ragkeep enthält die Bücher (wie in `PLAN.md`).
- Zusätzlich enthält ragkeep alle inhaltlichen Karten („Cards“), die als atomare Wissenseinheiten so gestaltet sind, dass sie „auf eine RAG‑Seite passen“.

### 2) Karten (Cards)
- **Definition**: Eine Karte ist eine kompakte, zitierfähige Wissenseinheit (Frage/Antwort, Zitat, Exzerpt, Begriffsklärung, Zusammenfassung), die eigenständig nutzbar ist und in Retrieval‑Szenarien als kleinste Einheit dient.
- **Zielgröße**: So kurz, dass der Inhalt auf eine Karte bzw. eine RAG‑Seite passt.

#### 2a) Kartentypen
- **Standardfragen‑Karten**: Klar beantwortbare, standardisierte Fragen (z. B. „Was sind die 12 Sinne?“) oder ähnliche strukturierte Kataloge.
- **Perspektiv‑/Weltanschauungs‑Karten**: Antworten auf die gleiche Frage aus verschiedenen weltanschaulichen Richtungen (z. B. „Was ist Wissen?“ in 12 Richtungen; „Was ist Freiheit?“ in zwei verschiedenen Richtungen).
- **Zitat‑/Exzerpt‑Karten**: Kuratierte Zitate oder Gedankenauszüge aus Kernwerken mit genauer Quellenangabe.
- **Zusammenfassungs‑Karten**: Buch‑Zusammenfassungen; ggf. hierarchisch organisiert (eine sehr kurze „Zusammenfassung der Zusammenfassung“ und eine etwas größere Kurzfassung). Mehrere Karten können zusammen eine Werk‑Zusammenfassung bilden.
- **Grundbegriffe‑Karten**: Grundbegriffe je Weltanschauung; sie können durch Werke bestimmter Autor:innen angereichert werden und stellen den Begriff knapp (Karte) und ggf. in längerer Ausformulierung dar.

#### 2b) Mögliche Felder pro Karte (Entwurf)
- `id` (stabile Karten‑ID)
- `typ` (standardfrage | perspektive | zitat | exzerpt | zusammenfassung | grundbegriff)
- `frage` / `thema` (falls zutreffend)
- `antwort` / `inhalt` (kompakter Hauptinhalt der Karte)
- `weltanschauung` / `perspektive` (falls zutreffend; z. B. „idealistisch“, „empiristisch“, …)
- `quelle` (Werk, Seitenangabe, bibliographische Referenz; Pflicht bei Zitaten/Exzerpten)
- `buch_id` (Referenz auf ein Buch in ragkeep, falls inhaltlich verknüpft)
- `begriffe` / `stichworte` (Tags)
- `version`, `updated_at`, `checksum` (Nachverfolgbarkeit/Integrität)

### 3) Verarbeitung & Aktualisierung (ragprep → ragkeep)
- Karten werden von `ragprep`‑Funktionen erzeugt, validiert und bearbeitet.
- Updates erfolgen über stabile Karten‑IDs; Änderungen können versioniert werden.
- Nach der Bearbeitung findet ein Update‑Schritt statt, der die Karte anhand ihrer `id` in ragkeep aktualisiert.
- Validierungsideen: Längenbegrenzung („passt auf eine Seite“), Pflichtfelder (z. B. Quelle bei Zitaten), Formatsch checks.

### 4) Beziehung zu Büchern
- Zitat‑/Exzerpt‑ und Zusammenfassungs‑Karten verweisen auf konkrete Bücher/Werke in ragkeep (`buch_id`).
- Werk‑Zusammenfassungen bestehen aus mehreren Karten; eine zusätzliche, sehr kurze Karte kann als Meta‑Zusammenfassung dienen.

### 5) Grundbegriffe je Weltanschauung
- Für zentrale Begriffe jeder Weltanschauung existieren eigene Karten.
- Diese Begriffe können durch Inhalte aus maßgeblichen Werken bereichert werden (Quellenpflicht).
- Darstellung zweistufig: knappe Karten‑Zusammenfassung und ggf. ausführlichere Ausformulierung (mehrere verbundene Karten).

### 6) Offene Punkte (zur Spezifikation)
- Endgültiges Kartenschema (verbindliche Felder, Typen, Längenrichtlinien).
- ID‑Konventionen und Versionierungsstrategie.
- Ablagestruktur in ragkeep (z. B. Ordner für Karten global vs. pro Buch).
- Schnittstellen in `ragprep` für Erzeugung, Aktualisierung, Validierung und Publikation.

Diese Punkte ergänzen `PLAN.md`, ohne dessen bestehende Veröffentlichungs‑ und Datenlayout‑Vorgaben zu verändern.


