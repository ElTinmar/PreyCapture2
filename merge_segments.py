# Merge segments from Hungarian tracking

from PyQt5.QtWidgets import (
    QWidget, 
    QApplication,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton
)
from PyQt5.QtCore import pyqtSignal, QTimer
from qt_widgets import LabeledSliderDoubleSpinBox
from pathlib import Path
from video_tools import OpenCV_VideoReader
from image_tools import ImageViewer 
import pandas as pd
import numpy as np
from numpy.typing import NDArray

class ParameciaClicker(ImageViewer):

    clicked = pyqtSignal(int, int)

    def __init__(self, image: NDArray, *args, **kwargs) -> None:

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
        self.fps = self.video_reader.get_fps()

        self.timestamps = pd.read_csv(timestampfile)
        self.tracking = pd.read_csv(trackingfile)

        self.play_timer = QTimer()
        self.play_timer.setInterval(int(1000//self.fps))  
        self.play_timer.timeout.connect(self.next_frame)

        self.create_components()
        self.layout_components()
        self.jump_to(0)
        
    def create_components(self):
        
        self.clicker = ParameciaClicker(image=np.zeros((self.height, self.width)))

        self.play_pause_button = QPushButton()
        self.play_pause_button.setStyleSheet("background-color : lightgrey")
        self.play_pause_button.setCheckable(True)
        self.play_pause_button.setText('Play')
        self.play_pause_button.clicked.connect(self.play_pause)

        self.time_slider = LabeledSliderDoubleSpinBox()
        self.time_slider.setText('time (s)')
        self.time_slider.setRange(0, self.timestamps['time'].max())
        self.time_slider.valueChanged.connect(self.jump_to)
        self.time_slider.setValue(0)

    def play_pause(self):
        
        if self.play_pause_button.isChecked():
            self.play_pause_button.setStyleSheet("background-color : lightblue")
            self.play_timer.start()
 
        else:
            self.play_pause_button.setStyleSheet("background-color : lightgrey")
            self.play_timer.stop()

    def layout_components(self):

        navigation_bar = QHBoxLayout()
        navigation_bar.addWidget(self.play_pause_button)
        navigation_bar.addWidget(self.time_slider)
        
        layout = QVBoxLayout(self)
        layout.addWidget(self.clicker)
        layout.addLayout(navigation_bar)

    def next_frame(self):

        rval, image = self.video_reader.next_frame()
        if rval:
            self.current_frame_index += 1
            img = self.overlay_tracking(image)
            self.clicker.set_image(img)

            # update slider
            time = self.timestamps[self.timestamps['index']==self.current_frame_index]['time'].values[0]
            self.time_slider.blockSignals(True)
            self.time_slider.setValue(time)
            self.time_slider.blockSignals(False)

    def overlay_tracking(self, image: NDArray) -> NDArray:

        row = self.tracking.loc[self.current_frame_index]
        print([(row.iloc[i],row.iloc[i+1],row.iloc[i+2]) for i in range(0, len(row), 3)])
        return image

    def jump_to(self, time_sec: float):

        index = (self.timestamps['time'] - time_sec).abs().argmin()
        self.video_reader.seek_to(index)
        self.current_frame_index = index

        rval, image = self.video_reader.next_frame()
        if rval:
            img = self.overlay_tracking(image)
            self.clicker.set_image(img)

if __name__ == "__main__":

    app = QApplication([])
    main = TrackMerger(
        videofile='/home/martin/Desktop/tracking/processed/2024_10_03_04_WT-1.70lux_fish2_chunk_005.avi',
        timestampfile='/home/martin/Desktop/tracking/processed/timestamp.csv',
        trackingfile='/home/martin/Desktop/tracking/processed/2024_10_03_04_WT-1.70lux_fish2_chunk_005.paramecia.csv'
    )
    main.show()
    app.exec_()

