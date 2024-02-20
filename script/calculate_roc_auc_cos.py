import json
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

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

    plt.savefig(f'/work/lixiaolou/program/auto_avsr/pic/{SAVE_PATH}/nor_roc_curve_full.png')

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

    plt.savefig(f'/work/lixiaolou/program/auto_avsr/pic/{SAVE_PATH}/nor_roc_curve.png')
    
def wer(wp, wn, name='WER'):
    # 绘制正例和副例的 WER 分布
    plt.figure()
    plt.hist(wp, alpha=0.5, label='Positive (label=1)', bins=10, density=True)
    plt.hist(wn, alpha=0.5, label='Negative (label=0)', bins=10, density=True)
    plt.xlabel('COS')
    plt.ylabel('Proportion')
    plt.title(f'{name} Distribution for Positive and Negative Examples (Proportional)')
    plt.legend(loc='upper right')

    plt.savefig(f'/work/lixiaolou/program/auto_avsr/pic/{SAVE_PATH}/{name}.png')

SAVE_PATH = 'FakeAVCeleb/ctc_cos'
DATA_PATH = 'fakeavceleb_ctc'
CHECK = 'cos'
# 读取 JSON 文件
with open(f'/work/lixiaolou/program/auto_avsr/infer/{DATA_PATH}.json', 'r') as file:
    data = json.load(file)

# 提取 label 和 cos
labels = [int(entry['label']) for entry in data.values()]
scores = [entry[CHECK] for entry in data.values()]  
min_score = min(scores)
max_score = max(scores)
scores = [(score - min_score) / (max_score - min_score) for score in scores]  # 归一化 (0, 1)

FFlabels = [int(entry['label']) for entry in data.values() if (entry['video label'] == '0' and entry['audio label'] == '0') or entry['label'] == '1']
FFscores = [entry[CHECK] for entry in data.values() if (entry['video label'] == '0' and entry['audio label'] == '0') or entry['label'] == '1']  # 限制 WER 在 0 到 1 范围内并反转

FRlabels = [int(entry['label']) for entry in data.values() if (entry['video label'] == '0' and entry['audio label'] == '1') or entry['label'] == '1']
FRscores = [entry[CHECK] for entry in data.values() if (entry['video label'] == '0' and entry['audio label'] == '1') or entry['label'] == '1']  # 限制 WER 在 0 到 1 范围内并反转

RFlabels = [int(entry['label']) for entry in data.values() if (entry['video label'] == '1' and entry['audio label'] == '0') or entry['label'] == '1']
RFscores = [entry[CHECK] for entry in data.values() if (entry['video label'] == '1' and entry['audio label'] == '0') or entry['label'] == '1']  # 限制 WER 在 0 到 1 范围内并反转

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

print(f'AUC: {roc_auc}')
print(f'FF AUC: {FFroc_auc}')
print(f'FR AUC: {FRroc_auc}')
print(f'RF AUC: {RFroc_auc}')

roc(fpr, tpr, roc_auc, FFfpr, FFtpr, FFroc_auc, FRfpr, FRtpr, FRroc_auc, RFfpr, RFtpr, RFroc_auc)
roc_s(fpr, tpr, roc_auc)
wer(wer_positive, wer_negative, name='COS')
wer(FFwer_positive, FFwer_negative, name='COS')
wer(FRwer_positive, FRwer_negative, name='COS')
wer(RFwer_positive, RFwer_negative, name='COS')





