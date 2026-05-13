#!/bin/bash

. ../path.sh || exit 1;

data_dir=/gscratch/tial/kpever/workspace/mqtts_training/datasets
cosyvoice_data_dir=../data

pretrained_model_dir=../../../../pretrained_models/CosyVoice2-0.5B

prosody_encoder_path=/gscratch/tial/kpever/workspace/prosodyvec/prosodyenc_weights.pt

# activate cosyvoice conda env
source /gscratch/tial/kpever/miniconda3/bin/activate cosyvoice

export CUDA_VISIBLE_DEVICES="0,1,2,3"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=5002
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp

echo "Train Flow with continuous prosody encoder (option 3)"
cp ../data/train/parquet/data.list ../data/train.data.list
cp ../data/dev/parquet/data.list ../data/dev.data.list
for model in flow; do
torchrun --nnodes=1 --nproc_per_node=$num_gpus \
      --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:5002" \
../../../../cosyvoice/bin/train.py \
--train_engine $train_engine \
--config ../conf/cosyvoice2_prosody_encoder.yaml \
--train_data ../data/train.data.list \
--cv_data ../data/dev.data.list \
--qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
--onnx_path $pretrained_model_dir \
--model $model \
--checkpoint $pretrained_model_dir/$model.pt \
--prosody_encoder_path $prosody_encoder_path \
--model_dir `pwd`/../exp/cosyvoice2_w_prosody_encoder/$model/$train_engine \
--tensorboard_dir `pwd`/../tensorboard/cosyvoice2_w_prosody_encoder/$model/$train_engine \
--ddp.dist_backend $dist_backend \
--num_workers ${num_workers} \
--prefetch ${prefetch} \
--pin_memory \
--use_amp \
--deepspeed_config ../conf/ds_stage2.json \
--deepspeed.save_states model+optimizer
done
echo "Flow training completed"
