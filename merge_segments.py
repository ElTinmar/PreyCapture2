# Merge segments from Hungarian tracking

from PyQt5.QtWidgets import (
    QWidget, 
    QApplication,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QAction,
    QTreeWidget,
    QTreeWidgetItem
)
from PyQt5.QtCore import pyqtSignal, QTimer, Qt
from qt_widgets import LabeledSliderDoubleSpinBox, LabeledDoubleSpinBox
from pathlib import Path
from video_tools import OpenCV_VideoReader
from image_tools import ImageViewer 
import pandas as pd
import numpy as np
from numpy.typing import NDArray
import cv2
from scipy.spatial.distance import cdist
import networkx as nx
import pyqtgraph as pg
from scipy.signal import savgol_filter

def count(tracking):
    
    param_number = []
    frame = []
    for group, data in tracking.groupby('frame'):
        frame.append(group)
        param_number.append(data.shape[0])
    return frame, param_number

def auto_merge(tracking):
    # TODO: two paramecia can get the same merged index

    # extract trajectories per idx
    trajectories = {}
    for group, data in tracking.groupby('index'):
        trajectories[group] = data[['frame', 'x', 'y']].to_numpy()

    # get segment start and stop points
    idx = np.array([], int)
    segment_start = np.zeros((0,3), np.float32)
    segment_stop = np.zeros((0,3), np.float32)
    for key, val in trajectories.items():
        idx = np.hstack((idx, key))
        segment_start = np.vstack((segment_start, val[0,:] / [20,1,1]))
        segment_stop = np.vstack((segment_stop, val[-1,:] / [20,1,1]))

    # compute cost matrix and find connections between segment stop and start
    cost = cdist(segment_stop, segment_start)
    np.fill_diagonal(cost, np.inf)
    argmin = cost.argmin(axis=1)
    mincost = cost.min(axis=1)
    valid = (mincost <= 40) & (segment_start[argmin,0] > segment_stop[:,0])
    edges = np.vstack((idx[valid], idx[argmin[valid]])).T
    G = nx.Graph()
    G.add_edges_from(edges)
    G.add_nodes_from(idx[~valid])
    
    merge = {f'M{n:05d}' : sorted(component) for n, component in enumerate(nx.connected_components(G))}
    reversed_dict = {}
    for key, values in merge.items():
        for value in values:
            reversed_dict[value] = key 

    new_column = [reversed_dict[idx] for idx in tracking['index']]
    tracking['merged'] = new_column

    return tracking


# TODO add merging paramecias together 
class ParameciaClicker(ImageViewer):

    clicked = pyqtSignal(float, float)

    def __init__(self, image: NDArray, *args, **kwargs) -> None:

        super().__init__(image, *args, **kwargs)
        self.setMouseTracking(True)

    def mousePressEvent(self, event):

        widget_pos = event.pos()
        scene_pos = self.mapToScene(widget_pos)
        self.clicked.emit(scene_pos.x(), scene_pos.y())

class TrackMerger(QWidget):

    yrange_param = [5, 70]
    yrange_eyes = [-np.pi/2, np.pi/2]
    yrange_tail = [-60,60]
    
    def __init__(
            self,
            videofile: Path,
            timestampfile: Path,
            param_tracking: Path,
            fish_tracking: Path,
            *args, 
            **kwargs
        ):

        super().__init__(*args, **kwargs)

        self.videofile = videofile
        self.timestampfile = timestampfile
        self.param_tracking = param_tracking
        self.current_frame_index = 0
        self.trajectories = {}
        self.merge = {}
        self.selected_merged = []
        self.selected_original = []

        self.video_reader = OpenCV_VideoReader()
        self.video_reader.open_file(
            filename = videofile, 
            safe = False
        )
        self.height = self.video_reader.get_height()
        self.width = self.video_reader.get_width()
        self.fps = self.video_reader.get_fps()

        self.timestamps = pd.read_csv(timestampfile, delim_whitespace=True, header=None,  names=['index', 'time', 'frame_num'], index_col=0)
        self.param_tracking = pd.read_csv(param_tracking)
        self.param_tracking = auto_merge(self.param_tracking)

        self.fish_tracking = pd.read_csv(fish_tracking)

        self.max_time = self.timestamps['time'].max()/1000
        self.frames, self.num_param = count(self.param_tracking)
        self.num_param_smooth = savgol_filter(self.num_param, window_length=1800, polyorder=2)

        self.play_timer = QTimer()
        self.play_timer.setInterval(int(1000//self.fps))  
        self.play_timer.timeout.connect(self.next_frame)

        self.create_components()
        self.layout_components()
        self.jump_to(0)
        
    def create_components(self):

        self.clicker = ParameciaClicker(image=np.zeros((self.height, self.width)))
        self.clicker.clicked.connect(self.get_closest_param)

        self.play_pause_button = QPushButton()
        self.play_pause_button.setStyleSheet("background-color : lightgrey")
        self.play_pause_button.setCheckable(True)
        self.play_pause_button.setText('Play')
        self.play_pause_button.toggled.connect(self.play_pause)

        self.time_slider = LabeledSliderDoubleSpinBox()
        self.time_slider.setText('time (s)')
        self.time_slider.setRange(0, self.max_time)
        self.time_slider.valueChanged.connect(self.jump_to)
        self.time_slider.setValue(0)

        self.fps_spinbox = LabeledDoubleSpinBox()
        self.fps_spinbox.setText('fps')
        self.fps_spinbox.setRange(0, 1000)
        self.fps_spinbox.valueChanged.connect(self.change_fps)
        self.fps_spinbox.setValue(self.fps)

        self.plot_param_widget = pg.plot()
        self.plot_param_widget.setFixedHeight(150)
        self.plot_param_widget.setYRange(*self.yrange_param)
        self.plot_param_widget.plot(self.frames, self.num_param)
        self.plot_param_widget.plot(self.frames, self.num_param_smooth, pen='r') 
        self.current_loc1 = self.plot_param_widget.plot([0,0], self.yrange_param, pen='g')
        self.text_item = pg.TextItem(str(self.num_param_smooth[0]), anchor=(0.5, 0.5))  # Centered text
        self.plot_param_widget.addItem(self.text_item)
        self.text_item.setPos(10, self.yrange_param[1]) 

        self.plot_fish_widget = pg.plot()
        self.plot_fish_widget.setFixedHeight(150)
        self.plot_fish_widget.setYRange(*self.yrange_eyes)
        self.plot_fish_widget.plot(self.fish_tracking['image_index'], self.fish_tracking['left_eye_angle'], pen='b')
        self.plot_fish_widget.plot(self.fish_tracking['image_index'], self.fish_tracking['right_eye_angle'], pen='y')
        self.current_loc2 = self.plot_fish_widget.plot([0,0], self.yrange_eyes, pen='g')
        self.plot_fish_widget.setXRange(-150,150)

        self.plot_fish_widget_tail = pg.plot()
        self.plot_fish_widget_tail.setFixedHeight(150)
        self.plot_fish_widget_tail.setYRange(*self.yrange_tail)
        self.plot_fish_widget_tail.plot(self.fish_tracking['image_index'], self.fish_tracking[['tail_point_037_x','tail_point_038_x','tail_point_039_x']].mean(axis=1), pen='m')
        self.current_loc3 = self.plot_fish_widget_tail.plot([0,0], self.yrange_tail, pen='g')
        self.plot_fish_widget_tail.setXRange(-150,150)

        self.split_button = QPushButton('Split')

        self.merge_button = QPushButton('Merge')

        self.merge_widget = QTreeWidget()
        self.merge_widget.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.merge_widget.setColumnCount(2)
        self.merge_widget.setHeaderLabels(['Merged ID', 'Original ID'])
        for merged_id, data in self.param_tracking.groupby('merged'):
            merged_item = QTreeWidgetItem(self.merge_widget, [str(merged_id), ''])
            for original_id in data['index'].unique():
                QTreeWidgetItem(merged_item, ['', str(original_id)])
        self.merge_widget.itemSelectionChanged.connect(self.merge_selection_changed)

        self.play_action = QAction("Play/Pause", self)
        self.play_action.setShortcut("k")
        self.play_action.triggered.connect(self.play_pause_button.toggle)
        
        self.rewind_action = QAction("Rewind", self)
        self.rewind_action.setShortcut("j")
        self.rewind_action.triggered.connect(self.rewind)

        self.forward_action = QAction("Rewind", self)
        self.forward_action.setShortcut("l")
        self.forward_action.triggered.connect(self.forward)

        self.addAction(self.play_action) 
        self.addAction(self.rewind_action) 
        self.addAction(self.forward_action) 

    def change_fps(self):
        self.play_timer.setInterval(int(1000//self.fps_spinbox.value()))

    def play_pause(self):
        
        if self.play_pause_button.isChecked():
            self.play_pause_button.setStyleSheet("background-color : lightblue")
            self.play_timer.start()
 
        else:
            self.play_pause_button.setStyleSheet("background-color : lightgrey")
            self.play_timer.stop()

    def get_closest_param(self, x: float, y: float):
        data = self.param_tracking[self.param_tracking['frame']==self.current_frame_index]
        dist = np.sqrt((data['x'] - x)**2 + (data['y'] - y)**2)
        original_id = data.iloc[dist.argmin()]['index']
        merged_id = data.iloc[dist.argmin()]['merged']
        matching_items = self.merge_widget.findItems(merged_id, Qt.MatchExactly, column=0)
        if matching_items:
            self.merge_widget.setCurrentItem(matching_items[0])

    def merge_selection_changed(self):

        self.selected_merged = []
        self.selected_original = []

        selected_items = self.merge_widget.selectedItems()
        for item in selected_items:
            if item.parent() is None:
                merged_id = item.text(0)
                self.selected_merged.append(merged_id)
            else:
                original_id = int(item.text(1))
                self.selected_original.append(original_id)

        target_time = self.timestamps.loc[self.current_frame_index]['time']/1000
        self.jump_to(target_time)

    def layout_components(self):

        navigation_bar = QHBoxLayout()
        navigation_bar.addWidget(self.play_pause_button)
        navigation_bar.addWidget(self.time_slider)
        navigation_bar.addWidget(self.fps_spinbox)
        
        layout_plots = QVBoxLayout()
        layout_plots.addWidget(self.plot_param_widget)
        layout_plots.addWidget(self.plot_fish_widget)
        layout_plots.addWidget(self.plot_fish_widget_tail)
        layout_plots.addWidget(self.clicker)
        layout_plots.addLayout(navigation_bar)

        layout_controls = QHBoxLayout()
        layout_controls.addWidget(self.split_button)
        layout_controls.addWidget(self.merge_button)

        layout_tree = QVBoxLayout()
        layout_tree.addLayout(layout_controls)
        layout_tree.addWidget(self.merge_widget)
        
        layout = QHBoxLayout(self)
        layout.addLayout(layout_plots)
        layout.addLayout(layout_tree)
        
    def next_frame(self):

        rval, image = self.video_reader.next_frame()
        if rval:
            self.current_frame_index += 1
            img = self.overlay_tracking(image)
            self.clicker.set_image(img)

            # update slider
            time = self.timestamps[self.timestamps.index==self.current_frame_index]['time'].values[0]
            self.time_slider.blockSignals(True)
            self.time_slider.setValue(time/1000)
            self.time_slider.blockSignals(False)

            self.current_loc1.setData(
                [self.current_frame_index, self.current_frame_index],
                self.yrange_param
            )
            self.current_loc2.setData(
                [self.current_frame_index, self.current_frame_index],
                self.yrange_eyes
            )
            self.current_loc3.setData(
                [self.current_frame_index, self.current_frame_index],
                self.yrange_tail
            )
            self.text_item.setText(f"{self.num_param_smooth[self.current_frame_index]:.2f}")
            self.plot_fish_widget.setXRange(self.current_frame_index-150, self.current_frame_index+150)
            self.plot_fish_widget_tail.setXRange(self.current_frame_index-150, self.current_frame_index+150)

    def overlay_tracking(self, image: NDArray) -> NDArray:

        tracking_data = self.param_tracking[self.param_tracking['frame'] == self.current_frame_index]
        z = zip(
            tracking_data['frame'],
            tracking_data['index'],
            tracking_data['merged'],
            tracking_data['x'],
            tracking_data['y']
        )
        for frame, idx, merged_id, x, y in z:
            if not np.isnan(x):
                pos = np.int32((x,y))
                
                col = [0,255,0]
                if idx in self.selected_original:
                    col = [255,0,0]
                if merged_id in self.selected_merged:
                    col = [0,0,255]

                image = cv2.circle(image, pos, radius=15, color=col,thickness=1)
                image = cv2.putText(image, str(int(idx)), np.int32((x,y))-10, cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,255,255])
                image = cv2.putText(image, merged_id, np.int32((x,y))+10, cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,255,255])

        return image
    
    def rewind(self):
        target_time = max(0, self.timestamps.loc[self.current_frame_index]['time']/1000-1)
        self.jump_to(target_time)

    def forward(self):
        target_time = min(self.max_time, self.timestamps.loc[self.current_frame_index]['time']/1000+1)
        self.jump_to(target_time)

    def jump_to(self, time_sec: float):

        index = (self.timestamps['time']/1000 - time_sec).abs().argmin()
        self.video_reader.seek_to(index)
        self.current_frame_index = index

        self.current_loc1.setData(
            [self.current_frame_index, self.current_frame_index],
            self.yrange_param
        )
        self.current_loc2.setData(
            [self.current_frame_index, self.current_frame_index],
            self.yrange_eyes
        )
        self.current_loc3.setData(
            [self.current_frame_index, self.current_frame_index],
            self.yrange_tail
        )
        self.text_item.setText(f"{self.num_param_smooth[self.current_frame_index]:.2f}")
        self.plot_fish_widget.setXRange(self.current_frame_index-150, self.current_frame_index+150)
        self.plot_fish_widget_tail.setXRange(self.current_frame_index-150, self.current_frame_index+150)

        rval, image = self.video_reader.next_frame()
        if rval:
            img = self.overlay_tracking(image)
            self.clicker.set_image(img)

if __name__ == "__main__":

    data = [
        [
            '/media/martin/DATA/Mecp2/processed/2024_10_10_01_MeCP2-7.30Klux-Direct_fish1.avi',
            '/media/martin/DATA/Mecp2/reindexed/MeCP2-7.30Klux-Direct/2024_10_10_01.txt',
            '/media/martin/DATA/Mecp2/processed/2024_10_10_01_MeCP2-7.30Klux-Direct_fish1.paramecia_tracking.csv',
            '/media/martin/DATA/Mecp2/processed/2024_10_10_01_MeCP2-7.30Klux-Direct_fish1.fish_tracking.csv'
        ],
        [
            '/media/martin/DATA/Mecp2/processed/2024_10_10_02_WT-7.30Klux-Direct_fish1.avi',
            '/media/martin/DATA/Mecp2/reindexed/WT-7.30Klux-Direct/2024_10_10_02.txt',
            '/media/martin/DATA/Mecp2/processed/2024_10_10_02_WT-7.30Klux-Direct_fish1.paramecia_tracking.csv',
            '/media/martin/DATA/Mecp2/processed/2024_10_10_02_WT-7.30Klux-Direct_fish1.fish_tracking.csv'
        ],
    ]

    exp = 1

    app = QApplication([])
    main = TrackMerger(
        videofile=data[exp][0],
        timestampfile=data[exp][1],
        param_tracking=data[exp][2],
        fish_tracking=data[exp][3]
    )
    main.show()
    app.exec_()

