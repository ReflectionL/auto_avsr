import csv

# 读取原始 CSV 文件
with open('/work/lixiaolou/program/auto_avsr/data/FakeAVCeleb/label_test.csv', 'r') as file:
    reader = csv.reader(file)
    rows = list(reader)

# 处理每一行
for row in rows:
    path = row[0]
    # 判断标记并设置相应的 label
    if 'FakeVideo-RealAudio' in path:
        video_label = 0  # Fake video
        audio_label = 1  # Real audio
    elif 'RealVideo-FakeAudio' in path:  # 假设还有这种情况
        video_label = 1
        audio_label = 0
    elif 'FakeVideo-FakeAudio' in path:
        video_label = 0
        audio_label = 0
    else:
        video_label = 1  # 假设默认为真实
        audio_label = 1

    # 添加 label 到行
    row.extend([audio_label, video_label])

# 写入新的 CSV 文件
with open('/work/lixiaolou/program/auto_avsr/data/FakeAVCeleb/label_test_updated.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rows)
