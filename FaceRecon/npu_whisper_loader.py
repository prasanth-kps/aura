import sys
from faster_whisper import WhisperModel

print("Loading model...", flush=True)
model = WhisperModel("base", device="cpu", compute_type="int8")
print("Ready!", flush=True)

def transcribe(file_path):
    print(f"\nTranscribing: {file_path}", flush=True)

    segments, info = model.transcribe(
        file_path,
        beam_size=5,
        language="en",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        no_repeat_ngram_size=4,
    )

    print(f"Language: {info.language} ({info.language_probability:.2f})", flush=True)

    full_text = []
    for segment in segments:
        text = segment.text.strip()
        print(f"  [{segment.start:.1f}s → {segment.end:.1f}s] {text}", flush=True)
        full_text.append(text)

    return " ".join(full_text)

if __name__ == "__main__":
    file   = sys.argv[1] if len(sys.argv) > 1 else input("File path: ")
    result = transcribe(file)

    print("\n─── TRANSCRIPT ───────────────────────────────")
    print(result)
    print("──────────────────────────────────────────────")

    out_path = file.rsplit(".", 1)[0] + ".txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"Saved to: {out_path}")