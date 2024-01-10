data_dir=/ssd1/data/LRS/LRS3-TED
detector=retinaface
root_dir=/ssd1/data/processed/LRS3
dataset=lrs3
subset=train
seg_duration=24
groups=



python preprocess_lrs2lrs3.py \
    --data-dir $data_dir \
    --detector $detector \
    --root-dir $root_dir \
    --dataset $dataset \
    --subset $subset \
    --seg-duration $seg_duration \
    --groups $groups


# python merge.py \
#     --root-dir $root_dir \
#     --dataset $dataset \
#     --subset $subset \
#     --seg-duration $seg_duration \
#     --groups $groups