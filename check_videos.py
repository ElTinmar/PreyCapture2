# check if there is a discrepancy between number of frames in video and timestamp file

from video_tools import CPU_VideoProcessor
from config import cleandatafolder
import pandas as pd

RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"

movies = sorted(cleandatafolder.rglob("*.avi"))

for p in movies:

    csvfile = p.with_suffix('.txt') 
    processor = CPU_VideoProcessor(p)
    width, height, fps, num_frames, duration = processor.get_input_video_metadata()
    timestamp = pd.read_csv(csvfile, delim_whitespace=True, header=None,  names=['index', 'timestamp', 'frame_num'], index_col=0)
    
    discrepancy = abs(num_frames - (timestamp.index.max()+1)) 
    if discrepancy == 0:
        print(GREEN + p.stem + f": {num_frames}, {timestamp.index.max()+1}, {duration},  {timestamp['timestamp'].max()/1000}" + RESET)
    elif discrepancy <= 10:
        print(BLUE + p.stem + f": {num_frames}, {timestamp.index.max()+1}, {duration},  {timestamp['timestamp'].max()/1000}" + RESET)
    else:
        print(RED + p.stem + f": {num_frames}, {timestamp.index.max()+1}, {duration},  {timestamp['timestamp'].max()/1000}" + RESET)