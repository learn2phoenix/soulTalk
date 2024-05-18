""""Initial file for a POC for the pipeline"""

import argparse
import configparser
import glob
import librosa
import logging
import os
import shutil
import subprocess
from logging import handlers
import gdown
import torch

from soulTalk_diff2lip.generate import generate
from soulTalk_diff2lip.guided_diffusion.guided_diffusion.script_util import (
    tfg_model_and_diffusion_defaults,
    tfg_create_model_and_diffusion,
    args_to_dict
)
from soulTalk_diff2lip.guided_diffusion.guided_diffusion import dist_util
from soulTalk_diff2lip.face_detection import LandmarksType, FaceAlignment
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




def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")



def diff2lip_get_args(config):
    defaults = dict(
        # generate from a single audio-video pair
        save_orig = True,

        use_fp16 = True,
        #tfg specific
        face_hide_percentage=0.5,
        use_ref=False,
        use_audio=False,
        audio_as_style=False,
        audio_as_style_encoder_mlp=False,
        
        #data args
        nframes=1,
        nrefer=0,
        image_size=128,
        syncnet_T = 5,
        syncnet_mel_step_size = 16,
        audio_frames_per_video = 16, #for tfg model, we use sound corresponding to 5 frames centred at that frame
        audio_dim=80,
        is_voxceleb2=True,

        video_fps=25,
        sample_rate=16000, #audio sampling rate
        mel_steps_per_sec=80.,

        #sampling args
        clip_denoised=True, # not used in training
        sampling_batch_size=2,
        use_ddim=False,
        model_path="",
        sample_path="d2l_gen",
        sample_partition="",
        sampling_seed=1111,
        sampling_use_gt_for_ref=False,
        sampling_ref_type='gt', #one of ['gt', 'first_frame', 'random']
        sampling_input_type='gt', #one of ['gt', 'first_frame']
        
        # face detection args
        face_det_batch_size=64,
        pads = "0,0,0,0"
    )
    defaults.update(tfg_model_and_diffusion_defaults())
    diff2lip_args = argparse.Namespace()
    #setting default arguments
    for k,v in defaults.items():
        setattr(diff2lip_args, k, v)
    #adding arguments from config to args
    for k,v in config.items('diff2lip'):
        default_v = defaults.get(k)
        if default_v is None: # if default is none, then the value can be of anytype
            pass
        elif isinstance(default_v, bool):
            v = str2bool(v)
        elif isinstance(default_v, int):
            v = int(v)
        elif isinstance(default_v, float):
            v = float(v)
        setattr(diff2lip_args, k, v)
    return diff2lip_args

# def args_to_dict(args, keys, out_path):
#     return {k: getattr(args, k) for k in keys}

def diff2lip_download_checkpoint(url, output_path):
    gdown.download(url, output_path, quiet=False)


def diff2lip_generate(config, video_path, audio_path, out_path):
    diff2lip_args = diff2lip_get_args(config)
    if diff2lip_args.model_path.startswith('http'):
        os.makedirs('checkpoints', exist_ok=True)
        ckpt_path = 'checkpoints/diff2lip_model.pt'
        if not os.path.exists(ckpt_path):
            diff2lip_download_checkpoint(diff2lip_args.model_path, ckpt_path)
        diff2lip_args.model_path = ckpt_path
    model, diffusion = tfg_create_model_and_diffusion(
        **args_to_dict(diff2lip_args, tfg_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
            dist_util.load_state_dict(diff2lip_args.model_path, map_location='cpu')
    )
    model.to('cuda')
    if diff2lip_args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    detector = FaceAlignment(LandmarksType._2D, flip_input=False, device='cuda' if torch.cuda.is_available() else 'cpu')
    generate(video_path, audio_path, model, diffusion, detector,  diff2lip_args, out_path=out_path, save_orig=diff2lip_args.save_orig)



def main(args):
    config = configparser.ConfigParser()
    config.read('config.ini')

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
    # tmp_files = glob.glob(f"{args.temp_dir}/{'3f303e25-87a1-49ec-90a3-dc0f579b8c47'}_*.wav")
    tmp_files = glob.glob(f"{args.temp_dir}/{str(tmp_filename)}_*.wav")
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

    # # TODO: Call diff2Lip here. Use output_human_final_modulated.wav as audio and args.input as video. We have already
    # # imported from soulTalk_diff2lip import generate, so work from there

    diff2lip_generate(
        config=config, 
        video_path = args.input, 
        audio_path = f"{args.temp_dir}/output_human_final_modulated.wav", 
        out_path = f"{args.temp_dir}/translated_video.mp4"
    )




if __name__ == '__main__':
    args = parse_args()
    main(args)
