import cv2
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def ingest_video(folder_path: str, frame_interval_sec: int = 5) -> list[dict]:
    """
    Extract keyframes and caption them using BLIP.
    Returns list of {text, metadata (timestamp, frame path)}.
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    outputs = []

    for file in os.listdir(folder_path):
        if not file.lower().endswith((".mp4", ".mov", ".mkv")):
            continue

        video_path = os.path.join(folder_path, file)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = int(fps * frame_interval_sec)

        for i in range(0, frame_count, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, frame = cap.read()
            if not success:
                continue

            frame_path = f"{folder_path}/frames_{file}_{i}.jpg"
            cv2.imwrite(frame_path, frame)

            img = Image.open(frame_path)
            inputs = processor(img, return_tensors="pt")
            caption_ids = model.generate(**inputs)
            caption = processor.decode(caption_ids[0], skip_special_tokens=True)

            timestamp = i / fps
            outputs.append({
                "text": caption,
                "metadata": {
                    "source": video_path,
                    "modality": "video",
                    "timestamp_sec": timestamp,
                    "frame_path": frame_path
                }
            })

        cap.release()

    return outputs

if __name__ == "__main__":
    from pprint import pprint
    pprint(ingest_video("videos/")[:1])
