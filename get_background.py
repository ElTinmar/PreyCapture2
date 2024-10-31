from pathlib import Path
from video_tools import OpenCV_VideoReader, InpaintBackground, Polarity
import numpy as np
from config import resultfolder

for p in resultfolder.rglob("*fish[1-2].avi"):

    video_reader = OpenCV_VideoReader()
    video_reader.open_file(
        filename = p, 
        safe = False
    )
    height = video_reader.get_height()
    width = video_reader.get_width()
    fps = video_reader.get_fps()  
    num_frames = video_reader.get_number_of_frame()

    background = InpaintBackground(polarity=Polarity.DARK_ON_BRIGHT, video_reader=video_reader)
    background.initialize()
    np.save(resultfolder / p.stem + '.npy', background.get_background())
