from faster_whisper import WhisperModel

model = WhisperModel("medium.en", device="cpu", compute_type="int8")

segments, info = model.transcribe("/Users/michael/Downloads/dn2016-0725.mp4",
                                  language="en",
                                  vad_filter=True)  # optional, entfernt Pausen

print(f"Erkannte Sprache: {info.language} (Wahrscheinlichkeit: {info.language_probability:.2f})")
print(f"Dauer: {info.duration:.2f}s")

with open("dn2016-0725.txt", "w", encoding="utf-8") as f:
    for segment in segments:
        f.write(segment.text.strip() + "\n")

print("Transkript fertig in dn2016-0725.txt!")