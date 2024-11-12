from pathlib import Path
from video_tools import OpenCV_VideoReader, InpaintBackground, Polarity, StaticBackground
import numpy as np
from config import resultfolder, n_background_samples
import cv2
from functools import partial
from image_tools import polymask, CloneTool
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QDialog

background_subtracter = partial(
    StaticBackground, 
    polarity=Polarity.DARK_ON_BRIGHT, 
    num_sample_frames = n_background_samples
)

class CloneDialog(QDialog):
    '''wrapper class around CloneTool to be able to run in a loop'''
    
    def __init__(self, img):
        super().__init__()
        self.clone = CloneTool(img)
        layout = QVBoxLayout(self)
        layout.addWidget(self.clone)

    def get_image(self):
        return self.clone.get_image()

app = QApplication([])

for p in resultfolder.rglob("*fish[1-2]_chunk*.avi"):

    print(p)

    outfile = resultfolder / p.with_suffix('.npy')

    if outfile.exists():
        print(f'{outfile} already exists, skipping')
        continue

    video_reader = OpenCV_VideoReader()
    video_reader.open_file(
        filename = p, 
        safe = False
    )
    height = video_reader.get_height()
    width = video_reader.get_width()
    fps = video_reader.get_fps()  
    num_frames = video_reader.get_number_of_frame()

    background = background_subtracter(video_reader=video_reader)
    background.initialize()
    img = background.get_background_image()

    clone = CloneDialog(img)
    clone.exec()

    img = clone.get_image()

    cv2.destroyAllWindows()
    np.save(outfile, img)


