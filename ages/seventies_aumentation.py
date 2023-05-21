import os
import pandas as pd
import shutil

path_ori = "/home/usuaris/veussd/DATABASES/Common_Voice/cv11.0/ca/clips_wav/"
path_def = "/home/usuaris/veu/david.linde/features/clips_seventies_copy/"
tsv = pd.read_csv("/home/usuaris/veu/david.linde/features/v7_clases_tdt/train.tsv", sep='\t')

seventies = tsv[tsv['age']=='seventies']
print(seventies)

paths = seventies['path'].to_numpy()

for path in paths:
    path = path.split(".")[0]+".wav"
    path_file = path_ori+path
    shutil.copy(path_file,path_def)

print("End")
