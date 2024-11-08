# Merge segments from Hungarian tracking

from PyQt5.QtWidgets import (
    QWidget, 
    QApplication,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
)
from PyQt5.QtCore import pyqtSignal
from qt_widgets import LabeledSliderDoubleSpinBox
from pathlib import Path
from video_tools import OpenCV_VideoReader
from image_tools import ImageViewer 
import pandas as pd
import numpy as np

class ParameciaClicker(ImageViewer):

    clicked = pyqtSignal(int, int)

    def __init__(self, image: np.ndarray, *args, **kwargs) -> None:

        super().__init__(image, *args, **kwargs)
        self.setMouseTracking(True)

    def mousePressEvent(self, event):

        widget_pos = event.pos()
        scene_pos = self.mapToScene(widget_pos)
        self.clicked.emit(scene_pos.x, scene_pos.y)

class TrackMerger(QWidget):
    
    def __init__(
            self,
            videofile: Path,
            timestampfile: Path,
            trackingfile: Path,
            *args, 
            **kwargs
        ):

        super().__init__(*args, **kwargs)

        self.videofile = videofile
        self.timestampfile = timestampfile
        self.trackingfile = trackingfile
        self.current_frame_index = 0
        
        self.video_reader = OpenCV_VideoReader()
        self.video_reader.open_file(
            filename = videofile, 
            safe = False
        )
        self.height = self.video_reader.get_height()
        self.width = self.video_reader.get_width()

        self.timestamps = pd.read_csv(timestampfile)

        self.tracking = pd.read_csv(trackingfile)
        
    def create_components(self):
        
        self.clicker = ParameciaClicker(image=np.zeros(512,512))

        self.set_time = LabeledSliderDoubleSpinBox()
        self.set_time.setText('time (s)')
        self.set_time.valueChanged.connect(self.jump_to)

    def layout_components(self):
        
        layout = QVBoxLayout(self)
        layout.addWidget(self.clicker)
        layout.addWidget(self.set_time)

    def jump_to(self, time_sec: float):

        # maybe load video chunk around time point 
        index = (self.timestamps['time'] - time_sec).abs().argmin()
        self.video_reader.seek_to(index)
        self.current_frame_index = index

if __name__ == "__main__":

    app = QApplication([])
    main = TrackMerger(
        videofile='/home/martin/Desktop/tracking/processed/2024_10_03_04_WT-1.70lux_fish2_chunk_005.avi',
        timestampfile='/home/martin/Desktop/tracking/processed/timestamp.csv',
        trackingfile='/home/martin/Desktop/tracking/processed/2024_10_03_04_WT-1.70lux_fish2_chunk_005.paramecia.csv'
    )
    main.show()
    app.exec_()

    