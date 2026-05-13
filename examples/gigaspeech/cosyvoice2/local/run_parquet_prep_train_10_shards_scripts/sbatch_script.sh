
jid=`sbatch --array 1-10 -J cosyvoice2_parquet_prep_10shards ./submit_array_ckpt.sh ./ | sed "s/Submitted batch job //"`

echo "Submitted array job $jid"