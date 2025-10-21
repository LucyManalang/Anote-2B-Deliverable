from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import pytesseract
import os

def ingest_images(folder_path: str) -> list[dict]:
    """
    Run OCR + captioning (BLIP) over all images in a folder.
    Returns list of {text, metadata}.
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    outputs = []
    for file in os.listdir(folder_path):
        if not file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(folder_path, file)
        image = Image.open(img_path)

        # Caption
        inputs = processor(image, return_tensors="pt")
        caption_ids = model.generate(**inputs)
        caption = processor.decode(caption_ids[0], skip_special_tokens=True)

        # OCR (optional)
        ocr_text = pytesseract.image_to_string(image)

        combined_text = f"Caption: {caption}. OCR: {ocr_text.strip()}"
        outputs.append({
            "text": combined_text,
            "metadata": {"source": img_path, "modality": "image"}
        })

    return outputs

if __name__ == "__main__":
    from pprint import pprint
    pprint(ingest_images("images/")[:1])
