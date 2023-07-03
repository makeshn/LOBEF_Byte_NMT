#!/usr/bin/env bash
lang_pairs="de-en,hi-en,km-en,lo-en,ne-en,ta-en,te-en"
lang_list="../data/lang-list.txt"
# Download the data from the google drive link in the readme
path_2_data="../data/byte-bin/xx"

python train.py $path_2_data \
  --arch transformer \
  --save-dir $save_dir \
  --task translation_multi_simple_epoch \
  --sampling-method "temperature" \
  --sampling-temperature 1.5\
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs" \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --lr 5e-4 --warmup-updates 4000 --max-update 1000000 --min-lr '1e-09' --warmup-init-lr '1e-07' \
  --dropout 0.1 --weight-decay 0.0001  --left-pad-source False \
  --max-tokens 7000 --update-freq 3 \
  --max-source-positions 2048 --max-target-positions 2048 \
  --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 10 --no-epoch-checkpoints \
  --seed 222 --log-format simple --log-interval 20 \
  --no-encoder-embed --no-decoder-embed --byte-ncf \
  --skip-invalid-size-inputs-valid-test\
  --encoder-layers 2 \
  --byte-encoder-layers 4 \

