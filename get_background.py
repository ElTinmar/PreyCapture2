from pathlib import Path
from video_tools import OpenCV_VideoReader, InpaintBackground, Polarity, StaticBackground
import numpy as np
from config import resultfolder
import cv2
from functools import partial

background_subtracter = partial(StaticBackground, polarity=Polarity.DARK_ON_BRIGHT, num_sample_frames = 200)

for p in resultfolder.rglob("*fish[1-2].avi"):

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

    ok = False
    
    while not ok: 
        background = background_subtracter(video_reader=video_reader)
        background.initialize()
        img = background.get_background_image()
        cv2.imshow('background', img)
        if cv2.waitKey(0) == ord('y'):
            ok = True
        
    cv2.destroyAllWindows()
    np.save(resultfolder / p.with_suffix('.npy'), img)
