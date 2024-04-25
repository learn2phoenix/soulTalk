import logging
import whisper
import torch
import uuid
from TTS.api import TTS

# https://github.com/saba99/TTS-MultiLingual/tree/master
# https://github.com/openai/whisper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger(__name__)

# TODO: Implement config file
transcription_model = whisper.load_model("large-v3")
tts_model_name = "tts_models/fr/mai/tacotron2-DDC"
tts_model = TTS(tts_model_name, progress_bar=False, gpu=True)


# TODO: Implement whisper and transcribe models as a service. We can't afford to load them each time. Actually depends,
#  if we are using AWS workers,then either we use a single worker for entire processing, in which case we have to load
#  inside each job. Otherwise we need to have some persistent machines for heavier models. Decision for future ;)
def translate_audio(input_audio_file: str, output_audio_file: str, task='transcribe', language='fr'):
    """Translate the audio from one language to another"""
    # TODO: Implement config file
    result = transcription_model.transcribe(input_audio_file, task=task, language=language)
    tmp_filename = uuid.uuid4()

    tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False, gpu=True)

    for segment in result['segments']:
        start = segment['start']
        end = segment['end']
        text = segment['text']

        tts_model.tts_to_file(text=text, file_path=f"/tmp/{tmp_filename}_{start}_{end}.wav")
        tts.voice_conversion_to_file(source_wav=f"/tmp/{tmp_filename}_{start}_{end}.wav",
                                 target_wav="data/audio/examples/very_short_gladiator.wav",
                                 file_path=f"/tmp/{tmp_filename}_{start}_{end}_converted.wav")