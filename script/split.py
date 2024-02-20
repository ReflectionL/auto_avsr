#生成和MP4目录结构相同的wav文件
def generate_wav(input_directory, output_directory):
    import os
    import subprocess
    import logging
    
    logger = logging.getLogger()
    file_handler = logging.FileHandler(f'generate_wav_{dic_name}.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    logger1 = logging.getLogger('error')
    file_handler1 = logging.FileHandler(f'generate_wav_error_{dic_name}.log')
    file_handler1.setFormatter(formatter)
    logger1.addHandler(file_handler1)
    logger1.setLevel(logging.ERROR)

    def convert_mp4_to_wav(mp4_path, wav_path):
        """使用ffmpeg将MP4文件转换为WAV格式"""
        try:
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "panic",
                "-threads", "1",
                "-y",
                "-i", mp4_path,
                "-async", "1",
                "-ac", "1",
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                wav_path
            ]
            subprocess.run(command, check=True)
            logger.info(f"Converted {mp4_path} to {wav_path}")
        except subprocess.CalledProcessError as e:
            logger1.error(f"Failed to convert {mp4_path} to {wav_path}: {e}")

    def process_directory(input_dir, output_dir):
        """遍历目录并处理每个MP4文件"""
        for root, dirs, files in sorted(os.walk(input_dir)):
            for file in files:
                if file.lower().endswith(".mp4"):
                    mp4_path = os.path.join(root, file)
                    relative_path = os.path.relpath(root, input_dir)
                    output_directory = os.path.join(output_dir, relative_path)

                    if not os.path.exists(output_directory):
                        os.makedirs(output_directory)

                    wav_path = os.path.join(output_directory, os.path.splitext(file)[0] + ".wav")
                    convert_mp4_to_wav(mp4_path, wav_path)


    process_directory(input_directory, output_directory)

dic_name = 'LAV-DF'
input_directory = f'/ssd1/data/standard/{dic_name}'
output_directory = f'/ssd1/data/processed/{dic_name}/audio'
generate_wav(input_directory, output_directory)