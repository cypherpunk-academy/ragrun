"""Speaker-turn plaintext. Same rule as ragprep segmentsToTurnCuedText."""

from __future__ import annotations

from typing import Any


def segments_to_turn_cued_text(segments: list[dict[str, Any]]) -> str:
    prepared: list[tuple[str, str]] = []
    for seg in segments:
        text = " ".join(str(seg.get("text") or "").split())
        if not text:
            continue
        speaker = str(seg.get("speaker") or "").strip()
        prepared.append((text, speaker))

    speakers = {speaker for _text, speaker in prepared if speaker}
    if len(speakers) < 2:
        text = " ".join(piece for piece, _speaker in prepared)
        return f"{text}\n" if text else ""

    blocks: list[str] = []
    last_speaker: str | None = None
    for text, speaker in prepared:
        new_turn = bool(speaker) and speaker != last_speaker
        if not blocks:
            blocks.append(f">> {text}" if speaker else text)
            if speaker:
                last_speaker = speaker
            continue
        if new_turn:
            blocks.append(f">> {text}")
            last_speaker = speaker
            continue
        blocks[-1] = f"{blocks[-1]} {text}"

    if not blocks:
        return ""
    return "\n\n".join(blocks) + "\n"
