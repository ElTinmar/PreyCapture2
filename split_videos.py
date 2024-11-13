from video_tools import CPU_VideoProcessor
from pathlib import Path
from config import cleandatafolder, resultfolder, n_chunks, n_cores
from multiprocessing import Pool

def split(p):

    print(p)
    
    crop_processor = CPU_VideoProcessor(
        p.absolute(),
        profile = 'main',
        preset = 'fast',
        quality = 20
    )
    width, height, fps, num_frames, duration = crop_processor.get_input_video_metadata()
    
    for fish, left in zip(['fish1', 'fish2'], [0, width//2]):

        suffix = p.parent.name + '_' + fish

        crop_processor.crop(
            left = left,
            bottom = 0,
            width = width//2,
            height = height,
            suffix = suffix,
            dest_folder = resultfolder
        )

        split_processor = CPU_VideoProcessor(resultfolder / (p.stem + '_' + suffix + '.avi'))
        split_processor.split_frame(            
            n = n_chunks,
            suffix = 'chunk',
            dest_folder = resultfolder
        )

filters = ['7.30Klux']
files = [
    p for p in cleandatafolder.rglob("*.avi") 
    if any(s for s in filters if s in str(p.resolve()))
]

with Pool(n_cores) as pool:
    pool.map(split, files)

