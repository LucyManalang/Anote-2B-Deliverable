import whisper
import os

def ingest_audio(folder_path: str, model_size: str = "small") -> list[dict]:
    """
    Run Whisper ASR on all audio files in a folder.
    Returns list of {text, metadata (timestamps)}.
    """
    model = whisper.load_model(model_size)
    outputs = []

    for file in os.listdir(folder_path):
        if not file.lower().endswith((".wav", ".mp3", ".m4a")):
            continue

        audio_path = os.path.join(folder_path, file)
        result = model.transcribe(audio_path, verbose=False)

        for seg in result["segments"]:
            outputs.append({
                "text": seg["text"].strip(),
                "metadata": {
                    "source": audio_path,
                    "modality": "audio",
                    "start": seg["start"],
                    "end": seg["end"]
                }
            })

    return outputs

if __name__ == "__main__":
    from pprint import pprint
    pprint(ingest_audio("audio/")[:2])
