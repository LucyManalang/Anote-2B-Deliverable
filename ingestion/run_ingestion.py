from text_ingest import ingest_text
from image_ingest import ingest_images
from audio_ingest import ingest_audio
from video_ingest import ingest_video

def run_ingestion(file_path: str, file_type: str):
    """
    Run the appropriate ingestion function based on file type.
    """
    if file_type == "text":
        return ingest_text(file_path)
    elif file_type == "image":
        return ingest_images(file_path)
    elif file_type == "audio":
        return ingest_audio(file_path)
    elif file_type == "video":
        return ingest_video(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
def run_all(files: list[dict]): # dict with 'path' and 'type' keys
    """
    Run ingestion on a list of files with specified types.
    Each item in files should be a dict with 'path' and 'type' keys.
    """
    all_outputs = []
    for file in files:
        path = file['path']
        ftype = file['type']
        outputs = run_ingestion(path, ftype)
        all_outputs.extend(outputs)
    return all_outputs