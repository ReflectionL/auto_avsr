import os
import hydra
import logging
import random
import json

import torch
import torchaudio
import torchvision

import torch.nn.functional as F

from tqdm import tqdm

from predict import FunctionalModule, AddNoise
from predict_asr import filelist

from datamodule.transforms import TextTransform
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.nets.scorers.ctc import CTCPrefixScorer

from pytorch_lightning import seed_everything

NOISE_FILENAME = '/work/lixiaolou/program/auto_avsr/datamodule/babble_noise.wav'


def calculate_cosine_similarity(audio_feat, video_feat):
    # 将二维特征向量裁剪为相同的长度
    min_length = min(audio_feat.shape[0], video_feat.shape[0])
    audio_feat = audio_feat[:min_length]
    video_feat = video_feat[:min_length]

    # 将裁剪后的特征向量展平为一维
    audio_feat_1d = audio_feat.view(-1)
    video_feat_1d = video_feat.view(-1)

    # 标准化一维特征向量
    audio_feat_norm = F.normalize(audio_feat_1d, dim=0)
    video_feat_norm = F.normalize(video_feat_1d, dim=0)

    # 计算并返回余弦相似度
    return torch.dot(audio_feat_norm, video_feat_norm).item()


def calculate_average_cosine_similarity_per_row(audio_feat, video_feat):
    # 获取较短的张量长度
    min_length = min(audio_feat.shape[0], video_feat.shape[0])

    total_cosine_similarity = 0.0

    # 对每一行计算余弦相似度
    for i in range(min_length):
        audio_feat_row = F.normalize(audio_feat[i], dim=0)
        video_feat_row = F.normalize(video_feat[i], dim=0)
        total_cosine_similarity += torch.dot(audio_feat_row, video_feat_row)

    # 计算平均余弦相似度
    return (total_cosine_similarity / min_length).item()



@hydra.main(version_base="1.3", config_path="configs", config_name="df")
def main(cfg):
    logging.basicConfig(level=logging.INFO)
    seed_everything(42, workers=True)
    cfg.gpus = torch.cuda.device_count()
    device = cfg.device
    
    audio_pipeline = torch.nn.Sequential(
        AddNoise(snr_target=cfg.decode.snr_target)
        if cfg.decode.snr_target is not None
        else FunctionalModule(lambda x: x),
        FunctionalModule(
            lambda x: torch.nn.functional.layer_norm(x, x.shape, eps=1e-8)
        )
    )
    video_pipeline = torch.nn.Sequential(
        FunctionalModule(lambda x: x / 255.0),
        torchvision.transforms.CenterCrop(88),
        torchvision.transforms.Grayscale(),
        torchvision.transforms.Normalize(0.421, 0.165),
    )
    
    text_transform = TextTransform()
    token_list = text_transform.token_list
    
    audio_model = E2E(len(token_list), cfg.model.audio_backbone).to(device)
    video_model = E2E(len(token_list), cfg.model.visual_backbone).to(device)
    
    audio_model.load_state_dict(torch.load(cfg.audio_ckpt_path, map_location=lambda storage, loc: storage))
    video_model.load_state_dict(torch.load(cfg.video_ckpt_path, map_location=lambda storage, loc: storage))
    
    audio_model.eval()
    video_model.eval()
    
    logging.info('loadign fns...')
    fns = filelist(cfg.file_path)
    
    total_cos = 0
    total_cos_row = 0
    total_num = 0
    infos = {}
    
    logging.info('Go find some chips...')
    for i, (fn, length, label, audio_label, video_label) in enumerate(tqdm(fns)):
        audio_path = fn.replace('/video/', '/audio/').replace('.mp4', '.wav')
        video_path = fn
        
        audio, _ = torchaudio.load(audio_path, normalize=True)
        audio = audio.transpose(1, 0)
        audio = audio_pipeline(audio)
    
        video = torchvision.io.read_video(fn, pts_unit="sec", output_format="THWC")[0]
        video = video.permute((0, 3, 1, 2)).to(device)
        video = video_pipeline(video)
        
        with torch.no_grad():
            audio_enc_feat = audio_model.encoder(audio.unsqueeze(0).to(device), None)[0].squeeze(0)
            video_enc_feat = video_model.encoder(video.unsqueeze(0).to(device), None)[0].squeeze(0)
            
            cos = calculate_cosine_similarity(audio_enc_feat, video_enc_feat)
            cos_row = calculate_average_cosine_similarity_per_row(audio_enc_feat, video_enc_feat)
            
            total_cos += cos
            total_cos_row += cos_row
            total_num += 1
            info = {
                'fn': fn,
                'label': label,
                'audio label': audio_label,
                'video label': video_label,
                'cos': cos,
                'cos in row': cos_row,
            }
            
            json.dump(info, open(cfg.infer_path, 'a'), indent=4, ensure_ascii=False)
            infos[fn] = info
            
    print(f'total cos: {total_cos / total_num}')
    json.dump(infos, open(cfg.infer_path, 'w'), indent=4, ensure_ascii=False)
    
    
if __name__ == '__main__':
    main()
            