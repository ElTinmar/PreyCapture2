from video_tools import CPU_VideoProcessor
from config import datafolder, resultfolder

for p in datafolder.rglob("*.avi"):
    print(p)
    processor = CPU_VideoProcessor(
        p.absolute(),
        profile = 'main',
        preset = 'fast',
        quality = 20
    )
    width, height, fps, num_frames, duration = processor.get_input_video_metadata()
    processor.crop(
        left=0,
        bottom=0,
        width=width//2,
        height=height,
        suffix=f'{p.parent.name}_fish1',
        dest_folder=resultfolder
    )
    processor.crop(
        left=width//2,
        bottom=0,
        width=width//2,
        height=height,
        suffix=f'{p.parent.name}_fish2',
        dest_folder=resultfolder
    )