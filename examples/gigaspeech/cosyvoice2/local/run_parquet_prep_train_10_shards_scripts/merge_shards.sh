#!/bin/bash

parquet_dir=/gscratch/tial/kpever/workspace/CosyVoice/examples/gigaspeech/cosyvoice2/data/train/parquet

cat $parquet_dir/data.*.list > $parquet_dir/data.list