from pathlib import Path
import platform

folder_map = {
    'O1-596': '/media/martin/DATA/Mecp2',
    'TheNewBeast': '/media/martin/MARTIN_8TB_0/Work/Sumbre_New/Mecp2'
}

basefolder = Path(folder_map[platform.node()])   
datafolder =  basefolder / 'data'
resultfolder = basefolder / 'processed'
display = True
