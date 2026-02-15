import os
import glob
import torch
import numpy as np
import soundfile as sf
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

def generate_preview(voice_id: str, text: str) -> str:
    """
    Task F: Generate audio using the finetuned FastPitch model + HiFi-GAN
    """
    print(f"[{voice_id}] 🎬 Preparing preview generation...")
    
    # 1. Find the specific training run folder (it has a timestamp in the name)
    base_dir = os.path.join("data", "models", voice_id)
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    if not subdirs:
        raise Exception(f"No trained model folder found for voice_id {voice_id}")
    
    model_dir = subdirs[0] # Grab the folder created by the trainer
    
    # 2. Locate the best_model.pth and config.json
    pth_files = glob.glob(os.path.join(model_dir, "best_model_*.pth"))
    if not pth_files:
        raise Exception("No best_model.pth found. Training may have failed or was interrupted.")
    
    custom_model_path = pth_files[0]
    custom_config_path = os.path.join(model_dir, "config.json")
    
    # 3. Get the standard Vocoder (HiFi-GAN) to pair with your custom Acoustic model
    manager = ModelManager()
    voc_model_path, voc_config_path, _ = manager.download_model("vocoder_models/en/ljspeech/hifigan_v2")
    
    # 4. Load the Pipeline into VRAM
    print(f"[{voice_id}] 🧠 Loading Custom Voice Model into VRAM...")
    synth = Synthesizer(
        tts_checkpoint=custom_model_path,
        tts_config_path=custom_config_path,
        vocoder_checkpoint=voc_model_path,
        vocoder_config=voc_config_path,
        use_cuda=torch.cuda.is_available()
    )
    
    # 5. Generate Audio
    print(f"[{voice_id}] 🎙️ Synthesizing: '{text}'")
    wav = synth.tts(text)
    
    # 6. Save File
    output_dir = "static"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"preview_{voice_id}.wav"
    output_path = os.path.join(output_dir, filename)
    
    sf.write(output_path, np.array(wav), 22050)
    
    print(f"[{voice_id}] ✅ Preview saved to {output_path}")
    return filename