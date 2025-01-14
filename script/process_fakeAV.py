import os
import subprocess
import threading
import shutil
from queue import Queue
from tqdm import tqdm
import csv
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from multiprocessing import Lock
logging.basicConfig(level=logging.INFO)

dir_lock = Lock()
THREAD_NUM = 20
dic_name = 'DFDC'

input_path = f'/ssd1/DF/{dic_name}'
output_base_dir = f'/ssd1/data/standard/{dic_name}'


# 定义处理文件的函数
def process_file(input_path):
    relative_path = input_path.split(f"/{dic_name}/")[1]
    output_path = os.path.join(output_base_dir, relative_path)
    output_dir = os.path.dirname(output_path)
    with dir_lock:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    if input_path.endswith(".txt"):
        # 如果是文本文件，直接复制
        try:
            shutil.copy2(input_path, output_path)
            return output_path, None
        except IOError as e:
            return input_path, str(e)
    else:
        # 调用ffmpeg处理视频
        cmd = f"ffmpeg -y -i \"{input_path}\" -r 25 -ar 16000 \"{output_path}\""
        try:
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            return output_path, None
        except subprocess.CalledProcessError as e:
            print(str(e.stderr.decode()))
            return input_path, str(e.stderr.decode())

def get_files_from_folders(csv_file):
    """
    Read the CSV file and extract the last column as folder paths.
    Then get all files from these folders and store them in a list.

    :param csv_file: Path to the CSV file
    :return: A list of all files in the folders mentioned in the CSV file
    """
    folders = []
    files_in_folders = []

    # Read CSV file and extract the last column
    with open(csv_file, newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:  # Check if row is not empty
                folder = row[-1]  # Get the last element as folder path
                folders.append(folder)

    # Get all files from the folders
    for folder in folders:
        if not folder == '':
            fold_name = os.path.join('/ssd1/DF/FakeAVCeleb_v1.2/video', folder.split('/', 1)[1])
            file_list = os.listdir(fold_name)
            for file in file_list:
                files_in_folders.append(os.path.join(fold_name, file))

    return files_in_folders

# Example usage
# csv_file = '/ssd1/DF/FakeAVCeleb_v1.2/meta_data.csv'  # Replace with your CSV file path

def find_mp4_files(directory):
    mp4_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".mp4"):
                mp4_files.append(os.path.join(root, file))
    return mp4_files

# video_paths = get_files_from_folders(csv_file)

video_paths = find_mp4_files(input_path)
# random.shuffle(video_paths)
# video_paths = video_paths[:2000]


# 创建输出目录

if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

failed_files_name = f"failed_{dic_name}.txt"
success_files_name = f"success_{dic_name}.txt"

# 使用线程池处理文件
with ProcessPoolExecutor(max_workers=THREAD_NUM) as executor:
    future_to_path = {executor.submit(process_file, path): path for path in video_paths}
    with tqdm(total=len(video_paths)) as pbar:
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                input_path, error = future.result()
                if error:
                    with open(failed_files_name, "a") as f:
                        f.write(f"{input_path} : {error}\n")
                else:
                    with open(success_files_name, "a") as f:
                        f.write(f"{input_path}\n")
            except Exception as e:
                with open(failed_files_name, "a") as f:
                        f.write(f"{path} : {e}\n")
            pbar.update(1)

print("处理完成。")
