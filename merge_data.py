from video_tools import CPU_VideoProcessor, VideoProcessor
from pathlib import Path
from config import resultfolder, n_chunks
import pandas as pd

def merge_video(
        p: Path, 
        video_processor: VideoProcessor, 
        in_suffix: str, 
        out_suffix: str
    ) -> None:
    
    tracking_movies = [p.parent / (p.stem + f'_chunk_{n:03}' + in_suffix) for n in range(1,n_chunks+1)]
    video_processor.merge(tracking_movies, suffix = p.with_suffix(out_suffix))

def merge_csv(
        p: Path, 
        in_suffix: str, 
        out_suffix: str
    ) -> None:

    tracking_csv = [p.parent / (p.stem + f'_chunk_{n:03}' + in_suffix) for n in range(1,n_chunks+1)]
    merged_data = pd.DataFrame()
    last_frame = 0
    last_index = 0
    for file in tracking_csv:
        data = pd.read_csv(file)
        data['frame'] = data['frame'] + last_frame 
        data['index'] = data['index'] + last_index 
        last_frame = data['frame'].max() + 1
        last_index = data['index'].max() + 1
        merged_data = pd.concat([merged_data, data])
    merged_data.to_csv(p.with_suffix(out_suffix), index=False)

if __name__ == '__main__':

    movies = resultfolder.rglob("*fish[1-2].avi")

    for p in movies:

        print(p)
        processor = CPU_VideoProcessor(resultfolder)
        merge_video(p, processor, in_suffix = '.paramecia_tracking.avi', out_suffix = '.paramecia_tracking.avi')
        merge_video(p, processor, in_suffix = '.tracking.avi', out_suffix = '.fish_tracking.avi')
        merge_csv(p, in_suffix='.paramecia.csv', out_suffix='.paramecia_tracking.csv')
        merge_csv(p, in_suffix='.csv', out_suffix='.fish_tracking.csv')


