#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=./logs/whisper-large-v2-cv11.0-%j.log

module load gcc arrow python/3.8.10 ffmpeg/4.3.2 cuda # comment this if you want to run on your local or on your server
source ~/venv/bin/activate # activate your environment if you haven't, comment it if you don't need this

while [ $# -gt 0 ]; do
    if [[ $1 == "--"* ]]; then
        v="${1/--/}"
        declare "$v"="$2"
        shift
    fi
    shift
done

dataset_name="${dataset_name:-fleurs}"
dataset_dir="${dataset_dir:-/home/awaheed/scratch/ASRs/data/fleurs.hf}"
dataset_duration="${dataset_duration:-4.39}"
max_training_duration="${max_training_duration:-16.0}"
do_preprocessing="${do_preprocessing:-True}" # this is not related to data processing (ie: feature extraction) but rather whether we apply processing on text script
load_from_local="${load_from_local:-False}"
load_from_disk="${load_from_disk:-True}" # this  means data is processed
audio_column="${audio_column:-path}"
text_column="${text_column:-text}"
cache_dir="${cache_dir:-$SLURM_TMPDIR/cache}"
output_dir="${output_dir:-./results/}"
num_train_epochs="${num_train_epochs:-100}"
model_name_or_path="${model_name_or_path:-openai/whisper-large-v2}"
train_batch_size="${train_batch_size:-32}"
eval_batch_size="${eval_batch_size:-32}"
gradient_accumulation_steps="${gradient_accumulation_steps:-1}"

deepspeed --num_gpus=4 train.py \
    --output_dir $output_dir \
    --overwrite output_dir \
    --cache_dir "./cache" \
    --dataset_name $dataset_name \
    --dataset_dir $dataset_dir \
    --dataset_duration $dataset_duration \
    --training_duration $dataset_duration \
    --max_training_duration $max_training_duration \
    --audio_column $audio_column \
    --text_column $text_column \
    --load_from_disk $load_from_disk \
    --model_name_or_path $model_name_or_path \
    --do_train \
    --fp16 \
    --learning_rate "1e-5" \
    --warmup_step 500 \
    --num_train_epochs $num_train_epochs \
    --do_eval \
    --do_predict \
    --predict_with_generate \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 100 \
    --load_best_model_at_end \
    --metric_for_best_model "wer" \
    --greater_is_better "False" \
    --generation_max_length 225 \
    --eval_accumulation_steps 10 \
    --per_device_train_batch_size $train_batch_size \
    --per_device_eval_batch_size $eval_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --gradient_checkpointing \
    --save_total_limit 1 \
    --logging_step 10 \
    --deepspeed "ds_config2.json"