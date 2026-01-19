import os
import soundfile as sf
from ml.infer import synthesize
from rq import get_current_job  # Import this utility

# Ensure output directory exists
OUTPUT_DIR = "static"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_text_task(text: str) -> str:
    """
    Wrapper to call synthesis and save file.
    """
    # 1. Get the current Job ID dynamically
    job = get_current_job()
    # Fallback to "test" if running manually without a queue
    current_job_id = job.id if job else "manual_test"

    print(f"[{current_job_id}] processing: {text[:20]}...")
    
    # 2. Run Inference
    wav_data = synthesize(text)
    
    # 3. Save to disk (22050 is standard for VITS)
    filename = f"output_{current_job_id}.wav"
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    sf.write(file_path, wav_data, 22050)
    
    print(f"[{current_job_id}] saved to {file_path}")
    return filename