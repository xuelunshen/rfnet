#!/usr/bin/env bash
root="/home/sxl/DL/runs"
export CUDA_VISIBLE_DEVICES=$1
fn=$2
model=$3

save=""$root"/"$fn""
resume=""$save"/model/"$model""

echo "Resume "$resume""

log=""$save"/"$fn"_resume.txt"
nohup python -utt train.py --save=$save --det-step=1 --des-step=2 --resume=$resume > $log &
sleep 1s
tail -f $log

