import os
import shutil
import glob
from pydub import AudioSegment
from pydub.silence import split_on_silence
import whisper
import torch

# Configuration for FastPitch
TARGET_SR = 22050 

def process_voice_dataset(voice_id: str, raw_dir: str, processed_dir: str):
    """
    Pipeline: Raw Audio -> Clean Chunks -> Transcribed CSV
    """
    print(f"[{voice_id}] 1. Starting preprocessing...")
    
    # Ensure processed directory exists
    wavs_dir = os.path.join(processed_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)
    
    # 1. Load all audio files from raw dir
    combined_audio = AudioSegment.empty()
    files = glob.glob(os.path.join(raw_dir, "*"))
    
    for f in files:
        try:
            audio = AudioSegment.from_file(f)
            combined_audio += audio
        except Exception as e:
            print(f"Skipping bad file {f}: {e}")

    # 2. Convert to Mono & Resample (Task B)
    combined_audio = combined_audio.set_frame_rate(TARGET_SR).set_channels(1)
    
    # 3. Split by Silence (Chunking)
    # This creates chunks of 3-10 seconds ideally
    chunks = split_on_silence(
        combined_audio,
        min_silence_len=500,  # 0.5s of silence marks a break
        silence_thresh=-40,   # dB
        keep_silence=200      # Keep 200ms padding
    )
    
    # 4. Save Chunks & Prepare for Transcription
    chunk_paths = []
    for i, chunk in enumerate(chunks):
        # Filter too short chunks (noise)
        if len(chunk) < 1000: continue 
        
        filename = f"wav_{i}.wav"
        out_path = os.path.join(wavs_dir, filename)
        
        # Export as WAV (Task B)
        chunk.export(out_path, format="wav")
        chunk_paths.append(filename)
        
    print(f"[{voice_id}] 2. Audio segmented into {len(chunk_paths)} chunks.")

    # 5. Transcribe (Task D) using Whisper
    # FastPitch NEEDS text to train. We cannot use blank transcripts.
    print(f"[{voice_id}] 3. Running ASR (Whisper) to generate transcripts...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load small model for speed (sufficient for alignment)
    asr_model = whisper.load_model("small", device=device)
    
    metadata = []
    
    for filename in chunk_paths:
        full_path = os.path.join(wavs_dir, filename)
        
        # Whisper Inference
        result = asr_model.transcribe(full_path)
        text = result["text"].strip()
        
        if len(text) > 2: # Keep only valid text
            # Format: wav_filename|transcript|normalized_transcript
            # We just pass the text twice to satisfy the LJSpeech format
            metadata.append(f"{filename.replace('.wav', '')}|{text}|{text}")
            
    # 6. Save metadata.csv (Task D)
    meta_path = os.path.join(processed_dir, "metadata.csv")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata))
        
    print(f"[{voice_id}] ✅ Processing Complete. Dataset ready at {processed_dir}")

    from redis import Redis
    from rq import Queue
    from services.training import finetune_fastpitch
    
    print(f"[{voice_id}] ➡️ Enqueuing Training Job...")
    redis_conn = Redis()
    q = Queue('default', connection=redis_conn)
    # Training can take hours, set a long timeout (e.g., 24 hours = 86400s)
    q.enqueue(finetune_fastpitch, voice_id, job_timeout=86400)

    return len(metadata)