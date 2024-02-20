audio_noisy=(999999 12.5 7.5 2.5 -2.5 -7.5)
video_noisy=(2 5)


for ((i=0;i<${#audio_noisy[@]};i++)); do
    for ((j=0;j<${#video_noisy[@]};j++)); do
        echo "audio_noisy: ${audio_noisy[$i]}, video_noisy: ${video_noisy[$j]}"

        root_dir=/work/lixiaolou/program/auto_avsr
        file_path=$root_dir/data/LAV-DF/label.csv
        infer_path=$root_dir/infer/LAV-DF/audio_${audio_noisy[$i]}_video_${video_noisy[$j]}_vsr.json
        asr_infer_path=$root_dir/infer/LAV-DF/audio_${audio_noisy[$i]}_video_${video_noisy[$j]}_asr.csv

        CUDA_VISIBLE_DEVICES=2 python predict_asr.py \
                                decode.snr_target=${audio_noisy[$i]} \
                                file_path=$file_path \
                                infer_path=$infer_path \
                                asr_infer_path=$asr_infer_path

        CUDA_VISIBLE_DEVICES=2 python predict_vsr.py \
                                decode.snr_target=${video_noisy[$j]} \
                                file_path=$asr_infer_path \
                                infer_path=$infer_path
    done
done