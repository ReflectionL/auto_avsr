modality=video
root_dir=/ssd1/data/processed/LRS3
test_file=lrs3_test_transcript_lengths_seg24s.csv
pretrained_model_path=/work/lixiaolou/program/auto_avsr/pretrained_model/vsr_trlrwlrs2lrs3vox2avsp_base.pth

python eval.py data.modality=$modality \
            data.dataset.root_dir=$root_dir \
            data.dataset.test_file=$test_file \
            pretrained_model_path=$pretrained_model_path