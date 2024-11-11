from video_tools import CPU_VideoProcessor
from pathlib import Path
from config import datafolder
import shutil

for p in datafolder.rglob("*.avi"):

    print(p)

    timestamp = p.with_suffix('.txt')

    destination = Path(str(p.parent).replace("/data", "/reindexed"))
    destination.mkdir(parents=True, exist_ok=True)

    processor = CPU_VideoProcessor(p)
    processor.reindex(suffix=None, dest_folder=destination)

    shutil.copy(timestamp, destination / timestamp.name)