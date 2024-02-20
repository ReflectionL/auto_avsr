import csv
import shutil
import os
from tqdm import tqdm

def copy_files(csv_file_path, destination_folder):
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        # next(reader)  # 跳过头部行

        for row in tqdm(reader):
            file_path = row[0]  # 读取文件路径
            file_path = file_path[:-4] + '.wav'
            if os.path.isfile(file_path):
                shutil.copy(file_path, destination_folder)
            else:
                print(f"File not found: {file_path}")

# 使用示例
csv_file_path = '/work/lixiaolou/program/auto_avsr/data/LAV-DF/filenames.csv'  # CSV文件路径
destination_folder = '/ssd1/data/processed/LAV-DF/audio/'  # 目标文件夹路径
copy_files(csv_file_path, destination_folder)
