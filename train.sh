#!/usr/bin/env bash
gpuid=$1

time="$(date "+%m_%d_%H_%M")"
source="$(pwd)"
save="$(dirname "$(pwd)")/runs/"$time""

export CUDA_VISIBLE_DEVICES=$gpuid

mkdir $save
mkdir ""$save"/log"
mkdir ""$save"/model"
mkdir ""$save"/image"
cp -r $source $save

echo "Saved to "$save""

nohup python -utt train.py --ver=$ver --save=$save --det-step=1 --des-step=2 > $save/$time.txt &
sleep 1s
tail -f $save/$time.txt

