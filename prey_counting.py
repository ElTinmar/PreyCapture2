from PyQt5.QtWidgets import (
    QWidget, 
    QApplication,
    QLabel
)
from PyQt5.QtCore import pyqtSignal
from qt_widget import LabeledSpinbox
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

class ParameciaCounter(QWidget):
    
    def __init__(
            self,
            videofile: Path,
            timestampfile: Path,
            *args, 
            **kwargs
        ):

        super().__init__(*args, **kwargs)

        self.videofile = videofile
        self.timestampfile = timestampfile
        self.current_frame = 0
        
        self.video_reader = OpenCV_VideoReader()
        self.video_reader.open_file(
            filename = videofile, 
            safe = False
        )
        self.height = self.video_reader.get_height()
        self.width = self.video_reader.get_width()

        self.timestamps = pd.read_csv(timestampfile)
        
    def create_components(self):
        
        self.image = ParameciaClicker(image=np.zeros(512,512))

        self.set_time = LabeledSpinbox()
        self.set_time.setText('time (s)')
        self.set_time.valueChanged.connect(self.jump_to)

    def layout_components(self):
        pass

    def jump_to(self, time_sec: float):
        # maybe load video chunk around time point 
        index = (self.timestamps['time'] - time_sec).abs().argmin()
        self.video_reader.seek_to(index)
        self.current_frame = index

if __name__ == "__main__":

    app = QApplication([])
    main = ParameciaCounter()
    main.show()
    app.exec_()