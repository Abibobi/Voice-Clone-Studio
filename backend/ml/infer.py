import os
import torch
import numpy as np
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

# --- WINDOWS FIX: FORCE ESPEAK DLL ---
possible_dll_paths = [
    r"C:\Program Files\eSpeak NG\libespeak-ng.dll",
    r"C:\Program Files (x86)\eSpeak NG\libespeak-ng.dll",
    r"C:\Program Files\eSpeak NG\lib\libespeak-ng.dll"
]

found_dll = False
for dll_path in possible_dll_paths:
    if os.path.exists(dll_path):
        print(f"✅ Found eSpeak DLL at: {dll_path}")
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = dll_path
        os.environ["PATH"] += os.pathsep + os.path.dirname(dll_path)
        found_dll = True
        break

if not found_dll:
    print("⚠️ WARNING: Could not find libespeak-ng.dll. Audio generation might fail.")


# Global variable to hold the synthesizer
synthesizer = None

def get_synthesizer():
    """
    Manually downloads and loads FastPitch (Acoustic) + HiFi-GAN (Vocoder)
    using the lower-level Synthesizer class.
    """
    global synthesizer
    if synthesizer is None:
        print("⏳ Downloading/Loading FastPitch + HiFi-GAN to GPU...")
        
        # 1. Setup Model Manager
        manager = ModelManager()
        
        # 2. Define Models
        model_name = "tts_models/en/ljspeech/fast_pitch"
        vocoder_name = "vocoder_models/en/ljspeech/hifigan_v2"
        
        # 3. Download and get paths for Acoustic Model
        model_path, config_path, model_item = manager.download_model(model_name)
        
        # 4. Download and get paths for Vocoder
        voc_model_path, voc_config_path, voc_item = manager.download_model(vocoder_name)
        
        print(f"   Acoustic: {model_name}")
        print(f"   Vocoder:  {vocoder_name}")

        # 5. Initialize the Synthesizer with both components
        synthesizer = Synthesizer(
            tts_checkpoint=model_path,
            tts_config_path=config_path,
            vocoder_checkpoint=voc_model_path,
            vocoder_config=voc_config_path,
            use_cuda=torch.cuda.is_available()
        )
        
        print(f"✅ FastPitch + HiFi-GAN Pipeline Ready on GPU")
    
    return synthesizer

def synthesize(text: str) -> np.ndarray:
    """
    Runs inference using the manual Synthesizer pipeline.
    """
    synth = get_synthesizer()
    
    # The synthesizer.tts() output format is a simple list of floats
    wav = synth.tts(text)
    
    return np.array(wav)