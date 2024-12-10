from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from config import resultfolder, n_cores
from multiprocessing import Pool

def count(tracking):
    
    param_number = []
    frame = []
    for group, data in tracking.groupby('frame'):
        frame.append(group)
        param_number.append(data.shape[0])
    return frame, param_number

def auto_merge(p: Path, out_suffix: str = '_merged', threshold: float = 40) -> None:
    
    print(p)

    tracking = pd.read_csv(p)
    
    # extract trajectories per idx
    trajectories = {}
    for group, data in tracking.groupby('index'):
        trajectories[group] = data[['frame', 'x', 'y']].to_numpy()

    # get segment start and stop points
    idx = np.array([], int)
    segment_start = np.zeros((0,3), np.float32)
    segment_stop = np.zeros((0,3), np.float32)
    for original_id, position in trajectories.items():
        idx = np.hstack((idx, original_id))
        segment_start = np.vstack((segment_start, position[0,:] / [20,1,1]))
        segment_stop = np.vstack((segment_stop, position[-1,:] / [20,1,1]))

    # compute cost matrix and find connections between segment stop and start
    cost = cdist(segment_start, segment_stop)

    # use numpy broadcasting to remove overlapping segments
    invalid_mask = segment_start[:, 0][:, None] <= segment_stop[:, 0][None, :]
    cost[invalid_mask] = 10_000 # np.inf does not work. Using a big number instead

    # use Hungarian algorithm to a find one-to-one mapping between segments
    row_idx, col_idx = linear_sum_assignment(cost)
    valid = cost[row_idx, col_idx] <= threshold
    row_idx, col_idx = row_idx[valid], col_idx[valid]

    # create graph and find connected components
    edges = np.column_stack((idx[row_idx], idx[col_idx]))
    G = nx.Graph()
    G.add_nodes_from(idx[~valid])
    G.add_edges_from(edges)
    
    merge = {f'M{n:05d}' : sorted(component) for n, component in enumerate(nx.connected_components(G))}
    reversed_dict = {}
    for key, values in merge.items():
        for value in values:
            reversed_dict[value] = key 

    new_column = [reversed_dict[idx] for idx in tracking['index']]
    tracking['merged'] = new_column

    out = p.parent / (p.stem + out_suffix + p.suffix)
    tracking.to_csv(out)

if __name__ == '__main__':

    #files = resultfolder.rglob("*.paramecia_tracking.csv")
    files = resultfolder.rglob("*.paramecia.csv")

    with Pool(n_cores) as pool:
        pool.map(auto_merge, files)