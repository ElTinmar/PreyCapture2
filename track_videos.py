from pathlib import Path

from video_tools import OpenCV_VideoReader, InpaintBackground, Polarity, BackroundImage
from image_tools import im2single, im2gray
from tracker import (
    GridAssignment, LinearSumAssignment,
    MultiFishTracker_CPU, MultiFishOverlay_opencv, MultiFishTrackerParamTracking, MultiFishTrackerParamOverlay,
    AnimalTracker_CPU, AnimalOverlay_opencv, AnimalTrackerParamTracking, AnimalTrackerParamOverlay,
    BodyTracker_CPU, BodyOverlay_opencv, BodyTrackerParamTracking, BodyTrackerParamOverlay,
    EyesTracker_CPU, EyesOverlay_opencv, EyesTrackerParamTracking, EyesTrackerParamOverlay,
    TailTracker_CPU, TailOverlay_opencv, TailTrackerParamTracking, TailTrackerParamOverlay
)
from tqdm import tqdm
import numpy as np
import json
import cv2
from config import resultfolder

DISPLAY = True

with open('tracking_fish.json', 'r') as fp:
    settings_fish = json.load(fp)

with open('tracking_paramecia.json', 'r') as fp:
    settings_paramecia = json.load(fp)

settings = settings_fish

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

    background = BackroundImage(resultfolder / p.stem + '.npy')
    background.initialize()

    LUT = np.zeros((height, width))
    assignment = GridAssignment(LUT)

    animal_tracker = AnimalTracker_CPU(
        assignment=assignment,
        tracking_param=AnimalTrackerParamTracking(
            **settings['animal_tracking'],
            source_image_shape = (height, width)
        )
    )
    body_tracker = BodyTracker_CPU(BodyTrackerParamTracking(**settings['body_tracking'])) if settings['body'] else None
    eyes_tracker = EyesTracker_CPU(EyesTrackerParamTracking(**settings['eyes_tracking'])) if settings['eyes'] else None
    tail_tracker = TailTracker_CPU(TailTrackerParamTracking(**settings['tail_tracking'])) if settings['tail'] else None

    animal_overlay = AnimalOverlay_opencv(AnimalTrackerParamOverlay())
    body_overlay = BodyOverlay_opencv(BodyTrackerParamOverlay()) if settings['body'] else None
    eyes_overlay = EyesOverlay_opencv(EyesTrackerParamOverlay()) if settings['eyes'] else None
    tail_overlay = TailOverlay_opencv(TailTrackerParamOverlay()) if settings['tail'] else None

    tracker = MultiFishTracker_CPU(
        MultiFishTrackerParamTracking(
            accumulator=None,
            animal=animal_tracker,
            body=body_tracker, 
            eyes=eyes_tracker, 
            tail=tail_tracker
        )
    )

    overlay = MultiFishOverlay_opencv(
        MultiFishTrackerParamOverlay(
            animal_overlay,
            body_overlay,
            eyes_overlay,
            tail_overlay
        )
    )

    for i in tqdm(range(num_frames)):
        (rval, frame) = video_reader.next_frame()
        if not rval:
            raise RuntimeError('VideoReader was unable to read the whole video')
        
        # convert
        frame_gray = im2single(im2gray(frame))

        # subtract background
        frame_noback = background.subtract_background(frame_gray)

        # track
        tracking = tracker.track(frame_noback)

        # TODO save tracking to file

        # display tracking
        if DISPLAY:
            oly = overlay.overlay(tracking['animals']['image_fullres'], tracking)
            r = cv2.resize(oly,(512, 512))
            cv2.imshow('overlay',r)
            cv2.waitKey(1)

    video_reader.close()
    cv2.destroyAllWindows()