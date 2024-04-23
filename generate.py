""""Initial file for a POC for the pipeline"""

import argparse
import logging
import os
from logging import handlers

from soulTalk_diff2lip import generate
from src import translate, voice_mod

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate the pipeline')
    parser.add_argument('--input', type=str, help='Input video file')
    return parser.parse_args()


def get_log_handlers(args):
    # Create handlers
    c_handler = logging.StreamHandler()
    # TODO: Implement UUID based logging for each run
    f_handler = handlers.RotatingFileHandler(f'soulTalk.log', maxBytes=int(1e6), backupCount=1000)
    c_handler.setLevel(logging.DEBUG)
    f_handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    return c_handler, f_handler


def extract_audio(input_video_file: str, output_audio_file: str):
    """Extract audio from the input video file"""
    os.system(f"ffmpeg -i {input_video_file} {output_audio_file}")


def translate_audio(input_audio_file: str, output_audio_file: str):
    """Translate the audio from one language to another"""
    translate.translate_audio(input_audio_file, output_audio_file)


def main(args):
    # TODO: Implement config
    extract_audio(args.input, "/tmp/output.wav")
    translate_audio("/tmp/output.wav", "/tmp/output_french.wav")
    # TODO: Replace the reference voice in the following function with the output from demucs
    voice_mod.modulate("/tmp/output_french.wav", "/tmp/output.wav", "/tmp/output_mod.wav")


if __name__ == '__main__':
    args = parse_args()
    main(args)
