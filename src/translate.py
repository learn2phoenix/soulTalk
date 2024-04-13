import whisper
import numpy as np
import os
import torch

# module load ffmpeg/6.0
# https://github.com/saba99/TTS-MultiLingual/tree/master
# https://github.com/openai/whisper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

task = "transcribe" # any to any
# # task = "translate" # any to english
# language = "fr"
# language = "hi"
language = "French"

model = whisper.load_model("large-v3")
result = model.transcribe("very_short_gladiator.wav", task=task, language=language)
print(result['text'])

from TTS.api import TTS

model_name = "tts_models/fr/mai/tacotron2-DDC"
tts = TTS(model_name, progress_bar=False, gpu=True)
tts.tts_to_file(text=result['text'], file_path="output_french.wav")

tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False, gpu=True)
tts.voice_conversion_to_file(source_wav="output_french.wav", target_wav="very_short_gladiator.wav", file_path="output_french_converted.wav")