""""Initial file for a POC for the pipeline"""

import argparse
import glob
import librosa
import logging
import os
import shutil
import subprocess
from logging import handlers

from soulTalk_diff2lip import generate
from src import translate, voice_mod

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate the pipeline')
    parser.add_argument('--input', type=str, help='Input video file')
    parser.add_argument('--ref', type=str, help='Reference audio file')
    parser.add_argument('--temp_dir', type=str, help='Temporary working dir')
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


def translate_audio(input_audio_file: str, target_audio_file: str, temp_dir: str):
    """Translate the audio from one language to another"""
    return translate.translate_audio(input_audio_file, target_audio_file, temp_dir)


def merge_wav_files_with_silence(file_list, timestamps, temp_dir, out_file):

    # TODO: Make this efficient by doing this in single ffmpeg command. Refer https://superuser.com/questions/931303/ffmpeg-concenating-audio-files-with-spacing
    prev_end = 0
    for i, file in enumerate(file_list):
        try:
            silence_duration = timestamps[i*2] - prev_end
            prev_end = timestamps[(i * 2) + 1]
        except IndexError:
            silence_duration = 0
        if i == 0:
            curr_audio_file = file
        else:
            shutil.move(f"{temp_dir}/temp_merged.wav", f"{temp_dir}/temp_merged_old.wav")
            curr_audio_file = f"{temp_dir}/temp_merged_old.wav"
        command = ['ffmpeg', '-y', '-hide_banner']
        filter_complex = ''
        command.extend(['-i', curr_audio_file])
        if i < len(file_list) - 1:
            command.extend(['-i', file_list[i + 1]])
            command.append('-filter_complex')
            filter_complex += f"[0:a]apad=pad_dur={silence_duration}[a0]; [a0][1:a]concat=n=2:v=0:a=1[out]"
            command.append(filter_complex)
            command.extend(['-map', '[out]'])
            command.append(f"{temp_dir}/temp_merged.wav")
            subprocess.run(command)
    shutil.move(f"{temp_dir}/temp_merged_old.wav", out_file)


def main(args):
    # TODO: Implement config
    os.makedirs(args.temp_dir, exist_ok=True)
    extract_audio(args.input, f"{args.temp_dir}/{args.input.split('/')[-1].split('.')[0]}.wav")
    voice_mod.voice_sep(args.ref)
    voice_mod.voice_sep(f"{args.temp_dir}/{args.input.split('/')[-1].split('.')[0]}.wav")
    tmp_filename, segments = translate_audio(f"./separated/mdx_extra/{args.input.split('/')[-1].split('.')[0]}/vocals.wav",
                                             f"./separated/mdx_extra/{args.ref.split('/')[-1].split('.')[0]}/vocals.wav",
                                             f"{args.temp_dir}")
    #
    # import pickle
    # with open(f"{args.temp_dir}/segments.pkl", 'wb') as f:
    #     pickle.dump(segments, f)
    # with open(f"{args.temp_dir}/segments.pkl", 'rb') as f:
    #     segments = pickle.load(f)
    # Joining the audio files with gaps between them
    timestamps = []
    tmp_files = glob.glob(f"{args.temp_dir}/{'3f303e25-87a1-49ec-90a3-dc0f579b8c47'}_*.wav")
    for segment in segments:
        timestamps.append(segment['start'])
        timestamps.append(segment['end'])
    merge_wav_files_with_silence(tmp_files, timestamps, args.temp_dir, f"{args.temp_dir}/output_human_final.wav")

    # Modulating the voice
    # Checking if both the files have the sample rate of 16KHz
    # TODO: Resolve the error in modulated voice
    y, sr = librosa.load(f"./separated/mdx_extra/{args.ref.split('/')[-1].split('.')[0]}/vocals.wav", sr=None)
    if sr != 16000:
        subprocess.run(['ffmpeg', '-i', f"./separated/mdx_extra/{args.ref.split('/')[-1].split('.')[0]}/vocals.wav",
                        '-ar', '16000', '-y', f"./separated/mdx_extra/{args.ref.split('/')[-1].split('.')[0]}/vocals_16k.wav"])
    y, sr = librosa.load(f"{args.temp_dir}/output_human_final.wav", sr=None)
    if sr != 16000:
        subprocess.run(['ffmpeg', '-i', f"{args.temp_dir}/output_human_final.wav", '-ar', '16000', '-y',
                        f"{args.temp_dir}/output_human_final_16k.wav"])
    voice_mod.modulate(f"{args.temp_dir}/output_human_final_16k.wav",
                       [f"separated/mdx_extra/{args.ref.split('/')[-1].split('.')[0]}/vocals_16k.wav",],
                       f"{args.temp_dir}/output_human_final_modulated.wav")



if __name__ == '__main__':
    args = parse_args()
    main(args)
