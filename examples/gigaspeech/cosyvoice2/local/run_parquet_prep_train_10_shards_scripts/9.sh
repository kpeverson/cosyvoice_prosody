
#!/bin/bash

. ../../path.sh || exit 1;

data_dir=/gscratch/tial/kpever/workspace/mqtts_training/datasets
cosyvoice_dir=/gscratch/tial/kpever/workspace/CosyVoice
cosyvoice_data_dir=$cosyvoice_dir/examples/gigaspeech/cosyvoice2/data

pretrained_model_dir=$cosyvoice_dir/pretrained_models/CosyVoice2-0.5B

# activate cosyvoice conda env
source /gscratch/tial/kpever/miniconda3/bin/activate cosyvoice

split=train
mkdir -p $cosyvoice_data_dir/$split/parquet
python $cosyvoice_dir/tools/make_parquet_list.py       --num_utts_per_parquet 1000       --num_processes 10       --src_dir $cosyvoice_data_dir/$split       --des_dir $cosyvoice_data_dir/$split/parquet       --hdf5_file $data_dir/segments.h5       --sample_rate 16000       --nshard 10       --shard_rank 8
