import argparse
import math
import os
import pickle
import shutil
import warnings

import ffmpeg
import multiprocessing
from multiprocessing import Manager, Value, Lock
from data.data_module import AVSRDataLoader
from tqdm import tqdm
from utils import save_vid_aud, save2vid

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="FakeAVCeleb Preprocessing")
parser.add_argument(
    "--data-lst",
    type=str,
    required=True,
    help="CSV file of dataset",
)
parser.add_argument(
    "--detector",
    type=str,
    default="retinaface",
    help="Type of face detector. (Default: retinaface)",
)
parser.add_argument(
    "--root-dir",
    type=str,
    required=True,
    help="Root directory of preprocessed dataset",
)
parser.add_argument(
    "--groups",
    type=int,
    default=1,
    help="Number of threads to be used in parallel.",
)
parser.add_argument(
    "--seg-duration",
    type=int,
    default=24,
    help="Max duration (second) for each segment, (Default: 24)",
)
parser.add_argument(
    "--label",
    type=str,
    required=True,
    help="label generated",
)
args = parser.parse_args()

# Constants
seg_vid_len = args.seg_duration * 25
seg_aud_len = args.seg_duration * 16000

# Load video files
with open(args.data_lst, 'r') as f:
    filenames = f.read().split('\n')
filenames = [x for x in filenames if x.endswith('.mp4')]
filenames = filenames[:300]

# Set up a shared counter and lock for multiprocessing progress tracking
manager = Manager()
progress_counter = Value('i', 0)
lock = Lock()

# split files into groups
num_processes = args.groups
file_chunks = [filenames[i::num_processes] for i in range(num_processes)]

def process_files(file_list, progress_counter, lock, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Load data
    vid_dataloader = AVSRDataLoader(
        modality="video", detector=args.detector, convert_gray=False
    )
    # aud_dataloader = AVSRDataLoader(modality="audio")
    
    for vid_filename in file_list:
        try:
            landmarks = None
            video_data = vid_dataloader.load_data(vid_filename, landmarks)
        except (UnboundLocalError, TypeError, OverflowError, AssertionError):
            continue
        if video_data is None:
            continue
        
        # Process Segments
        for i, start_idx in enumerate(range(0, len(video_data), seg_vid_len)):
            relative_path = vid_filename.split("/standard/FakeAVCeleb/")[1]
            if relative_path.split('/')[0] == 'RealVideo-RealAudio':
                label = 1
            else:
                label = 0
            dst_vid_filename = os.path.join(args.root_dir, 'video', relative_path)
            trim_vid_data = video_data
            
            if trim_vid_data is None:
                continue
            video_length = len(trim_vid_data)
            
            save2vid(dst_vid_filename, trim_vid_data, 25)
            with open(args.label, 'a') as f:
                f.write(f"{dst_vid_filename}, {video_length}, {label}\n")
        
        with lock:
            progress_counter.value += 1
   
            

processes = []
for i, file_chunk in enumerate(file_chunks):
    p = multiprocessing.Process(target=process_files, args=(file_chunk, progress_counter, lock, 5 - (i % 6)))
    processes.append(p)
    p.start()
    
with tqdm(total=len(filenames), desc='Processing Files') as pbar:
    while progress_counter.value < len(filenames):
        pbar.update(progress_counter.value - pbar.n)

for p in processes:
    p.join()

