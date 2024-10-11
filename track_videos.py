from pathlib import Path

from video_tools import OpenCV_VideoReader, InpaintBackground, Polarity
from image_tools import im2single, im2gray
from tracker import (
    GridAssignment,
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

DISPLAY = True

basefolder = Path('/media/martin/MARTIN_8TB_0/Work/Sumbre_New/Mecp2')   
datafolder =  basefolder / 'data'
resultfolder = basefolder / 'processed'

with open('tracking.json', 'r') as fp:
    settings = json.load(fp)

for p in resultfolder.rglob("*.avi"):

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

    video_reader.reset_reader()
    
    LUT = np.zeros((height, width))
    assignment = GridAssignment(LUT)

    animal_tracker = AnimalTracker_CPU(
        assignment=assignment,
        tracking_param=AnimalTrackerParamTracking(
            **settings['animal_tracking'],
            source_image_shape = (height, width)
        )
    )
    body_tracker = BodyTracker_CPU(BodyTrackerParamTracking(**settings['body_tracking']))
    eyes_tracker = EyesTracker_CPU(EyesTrackerParamTracking(**settings['eyes_tracking']))
    tail_tracker = TailTracker_CPU(TailTrackerParamTracking(**settings['tail_tracking']))

    animal_overlay = AnimalOverlay_opencv(AnimalTrackerParamOverlay())
    body_overlay = BodyOverlay_opencv(BodyTrackerParamOverlay())
    eyes_overlay = EyesOverlay_opencv(EyesTrackerParamOverlay())
    tail_overlay = TailOverlay_opencv(TailTrackerParamOverlay())

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