from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from redis import Redis
from rq import Queue
from rq.job import Job
from ml.infer_wrapper import process_text_task

from routers import voice

app = FastAPI()

# Enable CORS for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder to serve WAV files
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(voice.router) # Importing the voice router

# Redis connection
redis_conn = Redis()
q = Queue(connection=redis_conn)

class TTSRequest(BaseModel):
    text: str

@app.get("/health")
def health_check():
    return {"status": "ok", "gpu": "RTX 3070 Ti Ready"}

@app.post("/tts")
def generate_audio(request: TTSRequest):
    # Enqueue the job.
    # Note: We removed 'job_id=None'. 
    # RQ automatically assigns an ID, and the worker now fetches it itself.
    job = q.enqueue(process_text_task, request.text)
    return {"job_id": job.get_id()}

@app.get("/job/{job_id}")
def get_job_status(job_id: str):
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.is_finished:
        return {
            "status": "finished", 
            "result": job.result  # This matches the return value of process_text_task
        }
    elif job.is_failed:
        return {"status": "failed", "error": str(job.exc_info)}
    else:
        return {"status": "queued/started"}