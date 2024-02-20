import ffmpeg
import os
import glob
import csv
from tqdm import tqdm
import threading
import random

def get_media_info(media_path):
    """
    Get the frame rate of the video and the sample rate of the audio from a media file.

    :param media_path: Path to the media file
    :return: A tuple containing the video frame rate and audio sample rate
    """
    try:
        probe = ffmpeg.probe(media_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)

        video_frame_rate = None
        if video_stream and 'avg_frame_rate' in video_stream:
            num, denom = map(int, video_stream['avg_frame_rate'].split('/'))
            video_frame_rate = num / denom if denom != 0 else num

        audio_sample_rate = None
        if audio_stream and 'sample_rate' in audio_stream:
            audio_sample_rate = int(audio_stream['sample_rate'])

        return video_frame_rate, audio_sample_rate

    except Exception as e:
        print(f"Error processing {media_path}: {e}")
        return None, None

def process_media_files(media_directory, output_file):
    media_files = media_directory  # Change the extension if needed

    non_standard_media = []
    
    random.shuffle(media_files)
    media_files = media_files

    for media_file in tqdm(media_files):
        video_frame_rate, audio_sample_rate = get_media_info(media_file)
        print('video_frame_rate: ', video_frame_rate)
        print('audio_sample_rate: ', audio_sample_rate)
        if video_frame_rate != 25 or audio_sample_rate != 16000:
            non_standard_media.append((media_file, video_frame_rate, audio_sample_rate))

    with open(output_file, 'w') as f:
        f.write("Media files with non-standard frame rate (not 25fps) or sample rate (not 16000Hz):\n")
        for media in non_standard_media:
            f.write(f"{media[0]} - Frame Rate: {media[1]} fps, Audio Sample Rate: {media[2]} Hz\n")

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
                if not file.endswith('.txt'):
                    files_in_folders.append(os.path.join(fold_name, file))

    return files_in_folders

def find_mp4_files(directory):
    mp4_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".mp4"):
                mp4_files.append(os.path.join(root, file))
    return mp4_files

# Example usage
# csv_file = '/work/lixiaolou/program/auto_avsr/data/LAV-DF/filenames.csv'  # Replace with your CSV file path
# all_files = get_files_from_folders(csv_file)

output_file = 'output.txt'  # Replace with your desired output file path
# input_filefolder = '/ssd1/data/standard/DFDC'

with open('/work/lixiaolou/program/auto_avsr/data/LAV-DF/filenames.csv', 'r') as f:
    all_files = f.read().split('\n')
    
# success_files = [x for x in success_files if x.endswith('.mp4')]

# check_file = find_mp4_files(input_filefolder)

process_media_files(all_files, output_file)
