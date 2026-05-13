#!/bin/bash

. ../path.sh || exit 1;

data_dir=/gscratch/tial/kpever/workspace/mqtts_training/datasets

pretrained_model_dir=../../../../pretrained_models/CosyVoice2-0.5B

# activate cosyvoice conda env
source /gscratch/tial/kpever/miniconda3/bin/activate cosyvoice

python ../../../../tools/extract_speech_token.py \
      --dir ../data/dev \
      --onnx_path $pretrained_model_dir/speech_tokenizer_v2.onnx \
      --hdf5_file $data_dir/segments.h5 \
      --sample_rate 16000