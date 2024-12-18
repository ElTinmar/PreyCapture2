from video_tools import VideoWriter, OpenCV_VideoReader, FFMPEG_VideoWriter_CPU
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
from image_tools import im2single, im2gray, im2uint8, enhance
import numpy as np

# Installation:
#   pip install numpy
#   pip install tqdm 
#   pip install git+https://github.com/ElTinmar/video_tools.git@main
#   pip install git+https://github.com/ElTinmar/image_tools.git@main
 
datafolder = Path('/media/martin/MARTIN_8TB_0/Work/Sumbre_New/Mecp2/reindexed')
n_cores = 1

def create_diff_video(p, video_writer_constructor: VideoWriter = FFMPEG_VideoWriter_CPU, steps: int = 15):

    print(p)

    video_reader = OpenCV_VideoReader()
    video_reader.open_file(
        filename = p, 
        safe = False
    )
    height = video_reader.get_height()
    width = video_reader.get_width()
    fps = video_reader.get_fps()  
    num_frames = video_reader.get_number_of_frame()
    
    video_writer = video_writer_constructor(
        height = height,
        width = width,
        fps = fps,
        filename = p.with_suffix('.diff.avi'),
    )

    buffer = np.zeros((height, width, steps), dtype=np.float32)

    for i in tqdm(range(num_frames)):

        (rval, frame) = video_reader.next_frame()
        if not rval:
            print(f'error reading video frame {i}')
            continue
        
        frame_gray = im2single(im2gray(frame))
        diff_frame = np.zeros_like(frame_gray)
        for s in range(steps):
            diff_frame += np.clip(frame_gray - buffer[:,:,(i+s)%steps], 0, 1)/steps
        buffer[:,:,i%steps] = frame_gray

        diff_enhanced = enhance(
            diff_frame, 
            contrast = 3.0,
            gamma = 0.75,
            brightness = 0.0, 
            blur_size_px = 5, 
            medfilt_size_px = None
        )

        video_writer.write_frame(im2uint8(diff_enhanced))

    video_writer.close()
    video_reader.close()

files = [p for p in datafolder.rglob("*.avi")]

with Pool(n_cores) as pool:
    pool.map(create_diff_video, files)

