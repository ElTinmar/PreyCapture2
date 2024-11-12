from video_tools import CPU_VideoProcessor
from pathlib import Path
from config import datafolder, n_cores
import shutil
from multiprocessing import Pool

def reindex(p):
    print(p)
    
    # reindex videos
    destination = Path(str(p.parent).replace("/data", "/reindexed"))
    destination.mkdir(parents=True, exist_ok=True)
    processor = CPU_VideoProcessor(p)
    processor.reindex(suffix=None, dest_folder=destination)
    
    # copy timestamps as is
    timestamp = p.with_suffix('.txt')
    shutil.copy(timestamp, destination / timestamp.name)

with Pool(n_cores) as pool:
    pool.map(reindex, datafolder.rglob("*.avi"))