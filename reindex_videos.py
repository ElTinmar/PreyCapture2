from video_tools import CPU_VideoProcessor
from config import datafolder, cleandatafolder

for p in datafolder.rglob("*.avi"):

    print(p)
    
    processor = CPU_VideoProcessor(p)
    processor.reindex(suffix = 'reindexed', dest_folder = cleandatafolder)