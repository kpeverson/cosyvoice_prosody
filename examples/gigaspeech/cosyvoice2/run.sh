#!/bin/bash
# GigaSpeech training recipe using HDF5-stored audio.
. ./path.sh || exit 1;

stage=5
stop_stage=5

# Directory containing the GigaSpeech data files:
#   segments.h5, training.txt, validation.txt,
#   utt2spkr.json, train.json, dev.json
data_dir=/gscratch/tial/kpever/workspace/mqtts_training/datasets

pretrained_model_dir=../../../pretrained_models/CosyVoice2-0.5B

# activate cosyvoice conda env
source /gscratch/tial/kpever/miniconda3/bin/activate cosyvoice

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Stage 0: Data preparation — generate wav.scp/text/utt2spk/spk2utt"
  python local/prepare_data.py \
    --data_dir $data_dir \
    --des_dir data
  echo "Stage 0: Data preparation completed"
fi

# NOTE: embedding/token extraction is not strictly required — online feature
# extraction is supported — but pre-extracting speeds up training significantly.
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1: Extract CampPlus speaker embeddings"
  for split in train dev; do
    python ../../../tools/extract_embedding.py \
      --dir data/$split \
      --onnx_path $pretrained_model_dir/campplus.onnx \
      --hdf5_file $data_dir/segments.h5 \
      --sample_rate 16000
  done
  echo "Stage 1: Extract CampPlus speaker embeddings completed"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage 2: Extract discrete speech tokens"
  for split in train dev; do
    python ../../../tools/extract_speech_token.py \
      --dir data/$split \
      --onnx_path $pretrained_model_dir/speech_tokenizer_v2.onnx \
      --hdf5_file $data_dir/segments.h5 \
      --sample_rate 16000
  done
  echo "Stage 2: Extract discrete speech tokens completed"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage 3: Build Parquet shards"
  for split in train dev; do
    mkdir -p data/$split/parquet
    python ../../../tools/make_parquet_list.py \
      --num_utts_per_parquet 1000 \
      --num_processes 10 \
      --src_dir data/$split \
      --des_dir data/$split/parquet \
      --hdf5_file $data_dir/segments.h5 \
      --sample_rate 16000
  done
  echo "Stage 3: Build Parquet shards completed"
fi

# ---------- Training ----------

export CUDA_VISIBLE_DEVICES="0,1,2,3"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1987
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Stage 5: Train LLM / Flow / HiFiGAN"
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi
  cp data/train/parquet/data.list data/train.data.list
  cp data/dev/parquet/data.list data/dev.data.list
  for model in llm flow hifigan; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
      ../../../cosyvoice/bin/train.py \
      --train_engine $train_engine \
      --config conf/cosyvoice2.yaml \
      --train_data data/train.data.list \
      --cv_data data/dev.data.list \
      --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
      --onnx_path $pretrained_model_dir \
      --model $model \
      --checkpoint $pretrained_model_dir/$model.pt \
      --model_dir `pwd`/exp/cosyvoice2/$model/$train_engine \
      --tensorboard_dir `pwd`/tensorboard/cosyvoice2/$model/$train_engine \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --use_amp \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer
  done
  echo "Stage 5: Train LLM / Flow / HiFiGAN completed"
fi

average_num=5
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  for model in llm flow hifigan; do
    decode_checkpoint=`pwd`/exp/cosyvoice2/$model/$train_engine/${model}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python ../../../cosyvoice/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path `pwd`/exp/cosyvoice2/$model/$train_engine \
      --num ${average_num} \
      --val_best
  done
  echo "Stage 6: Model averaging completed"
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Stage 7: Export model for inference"
  python ../../../cosyvoice/bin/export_jit.py --model_dir $pretrained_model_dir
  python ../../../cosyvoice/bin/export_onnx.py --model_dir $pretrained_model_dir
fi
