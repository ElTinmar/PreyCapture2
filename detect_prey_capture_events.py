from config import resultfolder, cleandatafolder
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import numpy as np
from video_tools import OpenCV_VideoReader
import cv2

PREY_CAPTURE_SLIDING_WINDOW_FRAMES = 30
PREY_CAPTURE_THRESHOLD_PERCENTAGE = 0.40

files = sorted(cleandatafolder.rglob("*.avi"))

def get_processed_files(cleanfile: Path, fish: str):
    timestamps = cleanfile.with_suffix('.txt')
    fish_tracking = resultfolder / f'{cleanfile.stem}_{cleanfile.parent.stem}_{fish}.csv'
    return timestamps, fish_tracking

for f in files[::-1]:
    for fish in ['fish1','fish2']:
    
        timestamps, fish_tracking = get_processed_files(f, fish)
        timings = pd.read_csv(timestamps, delim_whitespace=True, header=None,  names=['index', 'time', 'frame_num'], index_col=0)
        time_sec = timings['time']/1000
        data = pd.read_csv(fish_tracking)
        X = np.rad2deg(data[['left_eye_angle', 'right_eye_angle']])
        
        remove = (data['left_eye_angle'] >= 0) | (data['right_eye_angle'] <= 0)
        X = X[~remove]

        vergence = X['right_eye_angle'] - X['left_eye_angle']
        fig = plt.figure()
        plt.hist(vergence, bins=100)
        fig.show()

        data['prey_capture_1D'] = 0
        data.loc[~remove,'prey_capture_1D'] = vergence > 40
        prey_capture = data['prey_capture_1D'].rolling(PREY_CAPTURE_SLIDING_WINDOW_FRAMES, center=False).mean() > PREY_CAPTURE_THRESHOLD_PERCENTAGE

        n_comp = 2
        gmm = GaussianMixture(n_components=n_comp, means_init=[[-15,15],[-35,35]])
        gmm.fit(X)

        components = gmm.predict(X)
        probabilities = gmm.predict_proba(X)

        fig, axs = plt.subplots(1,n_comp)
        for c in range(n_comp):
            axs[c].scatter(X['left_eye_angle'], X['right_eye_angle'], c=probabilities[:,c])
            axs[c].set_aspect('equal', adjustable='box')
        fig.show()

        data['prey_capture_2D'] = 0
        data.loc[~remove,'prey_capture_2D'] = components
        prey_capture = data['prey_capture_2D'].rolling(PREY_CAPTURE_SLIDING_WINDOW_FRAMES, center=True).mean() > PREY_CAPTURE_THRESHOLD_PERCENTAGE
        
        data['prey_capture_2D_filt'] = 0
        for n, v in enumerate(prey_capture):
            if v:
                data.loc[
                    max(0,n-PREY_CAPTURE_SLIDING_WINDOW_FRAMES//2):min(len(prey_capture),n+PREY_CAPTURE_SLIDING_WINDOW_FRAMES//2),
                    'prey_capture_2D_filt'
                ] = 1

        fig = plt.figure()
        plt.plot(data['prey_capture_2D'])
        plt.plot(data['prey_capture_2D_filt'])
        fig.show()
        
        time_sec[prey_capture]

        video_reader = OpenCV_VideoReader()
        video_reader.open_file(f)
        cv2.namedWindow('video')
        for a,b in zip(data['prey_capture_1D'], data['prey_capture_2D_filt']):
            ret, frame = video_reader.next_frame()
            if a:
                frame = cv2.circle(frame, (1000,20), 30, (0,255,0), -1)
            if b:
                frame =cv2.circle(frame, (1000,20), 30, (0,0,255), -1)

            cv2.imshow('video', frame)
            cv2.waitKey(33)





