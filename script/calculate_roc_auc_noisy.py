import json
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os

def roc(fpr, tpr, roc_auc, FFfpr, FFtpr, FFroc_auc, FRfpr, FRtpr, FRroc_auc, RFfpr, RFtpr, RFroc_auc):
    # 绘制 ROC 曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot(FFfpr, FFtpr, color='red', lw=2, label='FF ROC curve (area = %0.2f)' % FFroc_auc)
    plt.plot(FRfpr, FRtpr, color='green', lw=2, label='FR ROC curve (area = %0.2f)' % FRroc_auc)
    plt.plot(RFfpr, RFtpr, color='blue', lw=2, label='RF ROC curve (area = %0.2f)' % RFroc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    file_path = f'/work/lixiaolou/program/auto_avsr/pic/{SAVE_PATH}/nor_roc_curve_full.png'
    # 如果文件夹不存在，则创建文件夹
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
        
    plt.savefig(file_path)

def roc_s(fpr, tpr, roc_auc):
    # 绘制 ROC 曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    file_path = f'/work/lixiaolou/program/auto_avsr/pic/{SAVE_PATH}/nor_roc_curve.png'
    # 如果文件夹不存在，则创建文件夹
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    plt.savefig(file_path)
    
def wer(wp, wn, name='WER'):
    # 绘制正例和副例的 WER 分布
    plt.figure()
    plt.hist(wp, alpha=0.5, label='Positive (label=1)', bins=10, density=True)
    plt.hist(wn, alpha=0.5, label='Negative (label=0)', bins=10, density=True)
    plt.xlabel('WER')
    plt.ylabel('Proportion')
    plt.title(f'{name} Distribution for Positive and Negative Examples (Proportional)')
    plt.legend(loc='upper right')
    
    file_path = f'/work/lixiaolou/program/auto_avsr/pic/{SAVE_PATH}/{name}.png'
    # 如果文件夹不存在，则创建文件夹
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    plt.savefig(file_path)

def calculate_roc_auc_noisy(audio_noisy, video_noisy):
    # 读取 JSON 文件
    with open(f'/work/lixiaolou/program/auto_avsr/infer/{DATA_PATH}.json', 'r') as file:
        data = json.load(file)

    # 提取 label 和 wer，并将 wer 反转为得分，同时确保 WER 在 0 到 1 之间
    labels = [int(entry['label']) for entry in data.values()]
    scores = [1 - min(entry[CHECK], 1) for entry in data.values()]  # 限制 WER 在 0 到 1 范围内并反转
    # min_score = min(scores)
    # max_score = max(scores)
    # scores = [(score - min_score) / (max_score - min_score) for score in scores]  # 归一化 (0, 1)

    FFlabels = [int(entry['label']) for entry in data.values() if (entry['video label'] == '0' and entry['audio label'] == '0') or entry['label'] == '1']
    FFscores = [1 - min(entry[CHECK], 1) for entry in data.values() if (entry['video label'] == '0' and entry['audio label'] == '0') or entry['label'] == '1']  # 限制 WER 在 0 到 1 范围内并反转

    FRlabels = [int(entry['label']) for entry in data.values() if (entry['video label'] == '0' and entry['audio label'] == '1') or entry['label'] == '1']
    FRscores = [1 - min(entry[CHECK], 1) for entry in data.values() if (entry['video label'] == '0' and entry['audio label'] == '1') or entry['label'] == '1']  # 限制 WER 在 0 到 1 范围内并反转

    RFlabels = [int(entry['label']) for entry in data.values() if (entry['video label'] == '1' and entry['audio label'] == '0') or entry['label'] == '1']
    RFscores = [1 - min(entry[CHECK], 1) for entry in data.values() if (entry['video label'] == '1' and entry['audio label'] == '0') or entry['label'] == '1']  # 限制 WER 在 0 到 1 范围内并反转

    # 分别提取正例和副例的 WER
    wer_positive = [entry[CHECK] for entry in data.values() if int(entry['label']) == 1]
    wer_negative = [entry[CHECK] for entry in data.values() if int(entry['label']) == 0]

    FFwer_positive = [entry[CHECK] for entry in data.values() if int(entry['label']) == 1]
    FFwer_negative = [entry[CHECK] for entry in data.values() if int(entry['label']) == 0 and (entry['video label'] == '0' and entry['audio label'] == '0')]

    FRwer_positive = [entry[CHECK] for entry in data.values() if int(entry['label']) == 1]
    FRwer_negative = [entry[CHECK] for entry in data.values() if int(entry['label']) == 0 and (entry['video label'] == '0' and entry['audio label'] == '1')]

    RFwer_positive = [entry[CHECK] for entry in data.values() if int(entry['label']) == 1]
    RFwer_negative = [entry[CHECK] for entry in data.values() if int(entry['label']) == 0 and (entry['video label'] == '1' and entry['audio label'] == '0')]


    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(labels, scores)
    FFfpr, FFtpr, FFthresholds = roc_curve(FFlabels, FFscores)
    FRfpr, FRtpr, FRthresholds = roc_curve(FRlabels, FRscores)
    RFfpr, RFtpr, RFthresholds = roc_curve(RFlabels, RFscores)

    # 计算 AUC
    roc_auc = auc(fpr, tpr)
    FFroc_auc = auc(FFfpr, FFtpr)
    FRroc_auc = auc(FRfpr, FRtpr)
    RFroc_auc = auc(RFfpr, RFtpr)

    # 将下面这几行输出输出到文件中

    print(f'AUC: {roc_auc}')
    print(f'FF AUC: {FFroc_auc}')
    print(f'FR AUC: {FRroc_auc}')
    print(f'RF AUC: {RFroc_auc}')

    file_path = f'/work/lixiaolou/program/auto_avsr/pic/{SAVE_PATH}/output.txt'
    # 如果文件夹不存在，则创建文件夹
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    
    # Open the file in write mode
    with open(f'/work/lixiaolou/program/auto_avsr/pic/{SAVE_PATH}/output.txt', 'w') as f:
        # Write the AUC values to the file
        f.write(f'AUC: {roc_auc}\n')
        f.write(f'FF AUC: {FFroc_auc}\n')
        f.write(f'FR AUC: {FRroc_auc}\n')
        f.write(f'RF AUC: {RFroc_auc}\n')

    roc(fpr, tpr, roc_auc, FFfpr, FFtpr, FFroc_auc, FRfpr, FRtpr, FRroc_auc, RFfpr, RFtpr, RFroc_auc)
    roc_s(fpr, tpr, roc_auc)
    wer(wer_positive, wer_negative, name='WER')
    wer(FFwer_positive, FFwer_negative, name='FFWER')
    wer(FRwer_positive, FRwer_negative, name='FRWER')
    wer(RFwer_positive, RFwer_negative, name='RFWER')



audio_noisy = [999999, 12.5, 7.5, 2.5, -2.5, -7.5]
video_noisy = [0, 0.1, 0.5, 1, 2, 5]
CHECK = 'wer'
for audio in audio_noisy:
    for video in video_noisy:
        SAVE_PATH = f'Noisy/LAV-DF/audio_{audio}_video_{video}'
        DATA_PATH = f'LAV-DF/audio_{audio}_video_{video}_vsr'
        calculate_roc_auc_noisy(audio, video)

# SAVE_PATH = 'Noisy/FakeAVCeleb/'
# DATA_PATH = f'Noisy/fakeavceleb_noisy_audio_{audio_noisy}_video_{video_noisy}_vsr'







