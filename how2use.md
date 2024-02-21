## Setup

1. Set up environment:

```
conda create -y -n auto_avsr python=3.8
conda activate auto_avsr
```



1. Clone repository:

```
git clone https://github.com/mpc001/auto_avsr
cd auto_avsr
```



1. Install fairseq within the repository:

```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
cd ..
```



1. Install PyTorch (tested pytorch version: v2.0.1) and other packages:

```
pip install torch torchvision torchaudio
pip install pytorch-lightning==1.5.10
pip install sentencepiece
pip install av
pip install hydra-core --upgrade
```



1. Install ffmpeg:

```python
conda install "ffmpeg<5" -c conda-forge
```



## Pretrained Model

[VSR](https://drive.google.com/file/d/19GA5SqDjAkI5S88Jt5neJRG-q5RUi5wi/view?usp=sharing)

[ASR](https://drive.google.com/file/d/12vigJjL_ipgRz5CMYYQPdn8edEXD-Cuq/view?usp=sharing)

[AVSR]([`avsr_trlrwlrs2lrs3vox2avsp_base.pth`](https://drive.google.com/file/d/1mU6MHzXMiq1m6GI-8gqT2zc2bdStuBXu/view?usp=sharing))

下载之后统一放在`pretrained_model/`下

## Label

FakeAVCeleb和LAV-DF的label分别保存为

`data/FakeAVCeleb/label_test_updated.csv`

`data/LAV-DF/label.csv`

## Set

修改`configs/df.yaml`文件，主要是

```yaml
ckpt_path: /work/lixiaolou/program/auto_avsr/pretrained_model/avsr_trlrwlrs2lrs3vox2avsp_base.pth
audio_ckpt_path: /work/lixiaolou/program/auto_avsr/pretrained_model/asr_trlrwlrs2lrs3vox2avsp_base.pth
video_ckpt_path: /work/lixiaolou/program/auto_avsr/pretrained_model/vsr_trlrwlrs2lrs3vox2avsp_base.pth
```

这三项，改成上述的pretrained mode文件路径

## Run

参考`run_noisy.sh`

Audio的snr 999999为不加噪

Video的0为不加噪