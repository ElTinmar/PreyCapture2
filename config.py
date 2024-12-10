from pathlib import Path
import platform

folder_map = {
    'O1-596': '/media/martin/DATA/Mecp2',
    'TheNewBeast': '/media/martin/MARTIN_8TB_0/Work/Sumbre_New/Mecp2',
    'O1-619': '/home/martin/Downloads/Mecp2',
}

basefolder = Path(folder_map[platform.node()])   
datafolder =  basefolder / 'data'
cleandatafolder = basefolder / 'reindexed'
resultfolder = basefolder / 'processed'
display = False
export_GPU = False
n_chunks = 5
n_background_samples = 50
n_cores = 1