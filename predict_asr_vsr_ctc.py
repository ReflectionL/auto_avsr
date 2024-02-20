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


def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1, seq2)

def greedy_decode_keep_unk(ctc_output):
    """
    对 CTC 输出执行贪婪解码，保留所有元素包括 'unk'。

    :param ctc_output: CTC 模型的输出，形状为 [num_classes, seq_length]，例如 torch.Size([5049, 102])
    :return: 解码后的序列
    """
    # 获取每个时间步最可能的类别
    _, max_indices = torch.max(ctc_output, 0)

    # 保留所有元素，包括 'unk'
    decoded_sequence = max_indices.tolist()

    return decoded_sequence


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
    
    total_length = 0
    total_edit_distance = 0
    infos = {}
    
    logging.info('Go find some chips...')
    for i, (fn, length, label, audio_label, video_label) in enumerate(tqdm(fns)):
        audio_path = fn.replace('/video/', '/audio/').replace('.mp4', '.wav')
        video_path = fn
        
        audio, _ = torchaudio.load(audio_path, normalize=True)
        audio = audio.transpose(1, 0)
        audio = audio_pipeline(audio)
    
        video = torchvision.io.read_video(video_path, pts_unit="sec", output_format="THWC")[0]
        video = video.permute((0, 3, 1, 2)).to(device)
        video = video_pipeline(video)
        
        with torch.no_grad():
            audio_enc_feat = audio_model.encoder(audio.unsqueeze(0).to(device), None)[0].squeeze(0)
            video_enc_feat = video_model.encoder(video.unsqueeze(0).to(device), None)[0].squeeze(0)
            
            audio_hat = audio_model.ctc.ctc_lo(audio_enc_feat).transpose(0, 1)
            video_hat = video_model.ctc.ctc_lo(video_enc_feat).transpose(0, 1)
            
            tensor1_transposed = audio_hat.transpose(0, 1)  # 形状变为 [n, 5049]
            tensor2_transposed = video_hat.transpose(0, 1) 
            
            max_length = max(tensor1_transposed.size(0), tensor2_transposed.size(0))
            
            tensor1_padded = torch.zeros(max_length, 5049)
            tensor2_padded = torch.zeros(max_length, 5049)

            # 将原始数据复制到填充张量中
            tensor1_padded[:tensor1_transposed.size(0), :] = tensor1_transposed
            tensor2_padded[:tensor2_transposed.size(0), :] = tensor2_transposed

            # 计算余弦相似度
            cosine_similarities = F.cosine_similarity(tensor1_padded, tensor2_padded, dim=1)

            # 计算平均余弦相似度
            average_cosine_similarity = torch.mean(cosine_similarities)
            # logging.info(f'average_cosine_similarity: {average_cosine_similarity}')

            audio_pred = torch.tensor(greedy_decode_keep_unk(audio_hat))
            video_pred = torch.tensor(greedy_decode_keep_unk(video_hat))
            
            # audio_pred = text_transform.post_process(audio_pred_id).replace("<eos>", "")
            # video_pred = text_transform.post_process(video_pred_id).replace("<eos>", "")
            
            eer = compute_word_level_distance(audio_pred, video_pred)
            length = len(video_pred)
            if length == 0:
                wer = 0
            else:
                wer = eer / length
            
            total_edit_distance += eer
            total_length += length
            info = {
                'fn': fn,
                'label': label,
                'audio label': audio_label,
                'video label': video_label,
                'predicted': video_pred.numpy().tolist(),
                'target': audio_pred.numpy().tolist(),
                'wer': wer,
                'cos': average_cosine_similarity.item()
            }
            
            json.dump(info, open(cfg.infer_path, 'a'), indent=4, ensure_ascii=False)
            infos[fn] = info
            
    print(f'total wer: {total_edit_distance / total_length}')
    json.dump(infos, open(cfg.infer_path, 'w'), indent=4, ensure_ascii=False)
    
    
if __name__ == '__main__':
    main()
            