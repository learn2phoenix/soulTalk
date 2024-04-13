""""Initial file for a POC for the pipeline"""

import argparse
import logging
import os
from logging import handlers

from soulTalk_diff2lip import generate
from src import translate

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


def main():
    args = parse_args()
    # TODO: Implement config
    extract_audio(args.input, "temp/output.wav")
    translate_audio("temp/output.wav", "temp/output_french.wav")

