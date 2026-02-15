import os
import shutil
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File
from typing import List
from redis import Redis
from rq import Queue
from pydantic import BaseModel


# Import the processing function we wrote earlier
from services.processing import process_voice_dataset
# We must import the generation function
from services.preview import generate_preview

router = APIRouter(prefix="/voice", tags=["voice"])

RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

# Setup Redis queue connection
redis_conn = Redis()
q = Queue('default', connection=redis_conn)

@router.post("/upload")
async def upload_voice_samples(files: List[UploadFile] = File(...)):
    """
    Task A & B: Receive files, save raw data, and trigger preprocessing job.
    """
    # 1. Generate Voice ID
    voice_id = str(uuid.uuid4())[:8]
    
    # 2. Setup Directories
    raw_dir = os.path.join(RAW_DATA_PATH, voice_id)
    os.makedirs(raw_dir, exist_ok=True)
    
    # 3. Save Files
    saved_count = 0
    for file in files:
        file_path = os.path.join(raw_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_count += 1
        
    # 4. Define the output directory
    processed_dir = os.path.join(PROCESSED_DATA_PATH, voice_id)
    
    # 5. Enqueue the heavy lifting (Resampling, Chunking, Transcribing)
    # We add a longer timeout (3600s = 1 hr) because Whisper transcription takes time
    job = q.enqueue(process_voice_dataset, voice_id, raw_dir, processed_dir, job_timeout=3600)
            
    return {
        "voice_id": voice_id,
        "job_id": job.get_id(),  # We return the Job ID so the frontend can poll its status!
        "status": "processing_queued",
        "file_count": saved_count,
        "message": "Files saved. Whisper transcription & chunking started in background."
    }

# Define the expected JSON payload
class PreviewRequest(BaseModel):
    voice_id: str
    text: str

@router.post("/preview")
def create_preview(request: PreviewRequest):
    """
    Task F Endpoint: Trigger generation using the finetuned model.
    """
    # Enqueue the generation task so it doesn't freeze the API
    job = q.enqueue(generate_preview, request.voice_id, request.text)
    
    return {
        "job_id": job.get_id(),
        "status": "queued",
        "message": "Preview generation started."
    }

@router.get("/profiles")
def list_voice_profiles():
    """
    Task G: List all trained voice profiles.
    """
    profiles = []
    if os.path.exists(PROCESSED_DATA_PATH):
        for voice_id in os.listdir(PROCESSED_DATA_PATH):
            model_dir = os.path.join("data", "models", voice_id)
            
            # Check if training actually finished for this profile
            status = "processing"
            ckpt_path = None
            if os.path.exists(model_dir):
                subdirs = [os.path.join(model_dir, d) for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
                if subdirs:
                    status = "trained"
                    ckpt_path = subdirs[0]

            profiles.append({
                "id": voice_id,
                "status": status,
                "ckpt_path": ckpt_path
            })
            
    return {"profiles": profiles}

@router.delete("/{voice_id}")
def delete_voice_profile(voice_id: str):
    """
    Task G: Delete a voice profile and all its data.
    """
    raw_dir = os.path.join(RAW_DATA_PATH, voice_id)
    processed_dir = os.path.join(PROCESSED_DATA_PATH, voice_id)
    model_dir = os.path.join("data", "models", voice_id)
    
    deleted = False
    for d in [raw_dir, processed_dir, model_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
            deleted = True
            
    if deleted:
        return {"message": f"Voice profile {voice_id} successfully deleted."}
    return {"message": "Profile not found.", "status": 404}