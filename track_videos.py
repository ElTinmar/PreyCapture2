from pathlib import Path

from video_tools import (
    OpenCV_VideoReader, 
    Polarity, 
    BackroundImage, 
    VideoWriter, 
    FFMPEG_VideoWriter_CPU, 
    FFMPEG_VideoWriter_GPU
)
from image_tools import im2single, im2gray, im2uint8, im2rgb
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
from config import resultfolder, display, export_GPU, n_cores
from multiprocessing import Pool
from functools import partial

with open('tracking_fish.json', 'r') as fp:
    settings_fish = json.load(fp)

with open('tracking_paramecia.json', 'r') as fp:
    settings_paramecia = json.load(fp)

settings = settings_fish

if export_GPU:
    video_writer_constructor = FFMPEG_VideoWriter_GPU
else:
    video_writer_constructor = FFMPEG_VideoWriter_CPU

def _process(p: Path, settings: dict, display: bool, video_writer_constructor: VideoWriter):

    print(p)

    result_file = p.with_suffix('.csv')
    if result_file.exists():
        print(f'{result_file.absolute()} already exists, skipping')
        return

    n_pts_interp = settings['tail_tracking']['n_pts_interp']

    fd = open(result_file, 'w')
    headers = (
        'image_index',
        'centroid_x',
        'centroid_y',
        'pc1_x',
        'pc1_y',
        'pc2_x',
        'pc2_y',
        'left_eye_x',
        'left_eye_y',
        'left_eye_angle',
        'right_eye_x',
        'right_eye_y',
        'right_eye_angle',
    ) \
    + tuple(f'tail_point_{n:03d}_x' for n in range(n_pts_interp)) \
    + tuple(f'tail_point_{n:03d}_y' for n in range(n_pts_interp)) 
    fd.write(','.join(headers) + '\n')

    video_reader = OpenCV_VideoReader()
    video_reader.open_file(
        filename = p, 
        safe = False
    )
    
    height = video_reader.get_height()
    width = video_reader.get_width()
    fps = video_reader.get_fps()  
    num_frames = video_reader.get_number_of_frame()

    # correct height and width needs to take into account the export size in settings 
    video_writer = video_writer_constructor(
        height = height,
        width = width,
        fps = fps,
        filename = p.with_suffix('.tracking.avi'),
    )

    background = BackroundImage(p.with_suffix('.npy'), polarity=Polarity.DARK_ON_BRIGHT)
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

    tail_points = np.zeros((2*n_pts_interp,), np.float32)

    for i in tqdm(range(num_frames)):

        (rval, frame) = video_reader.next_frame()
        if not rval:
            print(f'error reading video frame {i}')
            continue
        
        # convert
        frame_gray = im2single(im2gray(frame))

        # subtract background
        frame_noback = background.subtract_background(frame_gray)

        # track
        tracking = tracker.track(frame_noback)

        # TODO save tracking to file
        tail_points[:n_pts_interp] = tracking['tail'][0]['skeleton_interp'][:,0]
        tail_points[n_pts_interp:] = tracking['tail'][0]['skeleton_interp'][:,1]

        row = (
            f"{i}",
            f"{tracking['body'][0]['centroid_original_space'][0]}",
            f"{tracking['body'][0]['centroid_original_space'][1]}",
            f"{tracking['body'][0]['heading'][0,0]}",
            f"{tracking['body'][0]['heading'][1,0]}",
            f"{tracking['body'][0]['heading'][0,1]}",
            f"{tracking['body'][0]['heading'][1,1]}",
            f"{tracking['eyes'][0]['left_eye']['centroid'][0]}",
            f"{tracking['eyes'][0]['left_eye']['centroid'][1]}",
            f"{tracking['eyes'][0]['left_eye']['angle']}",
            f"{tracking['eyes'][0]['right_eye']['centroid'][0]}",
            f"{tracking['eyes'][0]['right_eye']['centroid'][1]}",
            f"{tracking['eyes'][0]['right_eye']['angle']}",
        ) \
        + tuple(f"{tail_points[i]}" for i in range(n_pts_interp)) \
        + tuple(f"{tail_points[i]}" for i in range(n_pts_interp, 2*n_pts_interp)) 
        fd.write(','.join(row) + '\n')

        oly = overlay.overlay(tracking['animals']['image_fullres'], tracking)
        video_writer.write_frame(oly[:,:,[2,1,0]])

        # display tracking
        if display:

            img0 = cv2.resize(oly, (512, 512))
            img1 = im2rgb(cv2.resize(im2uint8(tracking['body'][0]['image']), (256, 256)))
            img2 = im2rgb(cv2.resize(im2uint8(tracking['body'][0]['mask']), (256, 256)))
            img3 = im2rgb(cv2.resize(im2uint8(tracking['eyes'][0]['image']), (256, 256)))
            img4 = im2rgb(cv2.resize(im2uint8(tracking['eyes'][0]['mask']) ,(256, 256)))
            img5 = im2rgb(cv2.resize(im2uint8(tracking['tail'][0]['image']), (512, 512)))

            montage0 = np.vstack((img1, img2))
            montage1 = np.vstack((img3, img4))
            montage = np.hstack((img0, montage0, montage1, img5))

            cv2.imshow(p.name, montage)
            cv2.waitKey(1)

    fd.close()
    video_writer.close()
    video_reader.close()
    cv2.destroyAllWindows() 

process = partial(
    _process,  
    settings = settings, 
    display = display, 
    video_writer_constructor=video_writer_constructor
)

with Pool(n_cores) as pool:
    pool.map(process, resultfolder.rglob("*fish[1-2]_chunk_[0-9][0-9][0-9].avi"))

