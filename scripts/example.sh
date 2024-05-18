#!/bin/bash


cd ../

python generate.py \
--input data/video/examples/hp_english_trim_face.mp4 \
--ref data/audio/examples/hp_english_trim_face.wav \
--temp_dir temp 