import os
import pickle
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

clas = "nineties"

txt = open(f"/home/usuaris/veu/david.linde/CommonVoice11/stats_{clas}.txt","w")
validated = pd.read_csv("/home/usuaris/veussd/DATABASES/Common_Voice/cv11.0/original/ca/validated.tsv", sep='\t')
validated.dropna(subset=['age'], inplace=True)

teens = validated.loc[(validated['age'] == clas )]
print(teens)
paths = teens['path'].tolist()
i=0
for i in range(0,len(paths)):
    paths[i] = paths[i].split(".")[0]+".pickle"

pathval = "/home/usuaris/veussd/DATABASES/Common_Voice/cv11.0/edat/ca/features_validated/"
frames = []

for path in paths:
    with open(pathval+path,'rb') as pickleFile: 
        features = pickle.load(pickleFile)
    frames.append(np.transpose(features).shape[0])

num_bins = len(set(frames))
n, bins, patches = plt.hist(frames, num_bins, facecolor='blue', alpha=0.5)
plt.title(f"Frames audio histogram: {clas}")
plt.savefig(f"/home/usuaris/veu/david.linde/CommonVoice11/hist_{clas}.jpg")

frames = np.array(frames)

p1 = np.percentile(frames,1)
p5 = np.percentile(frames,5)
p25 = np.percentile(frames,25)
p50 = np.percentile(frames,50)
p75 = np.percentile(frames,75)
p95 = np.percentile(frames,95)
p99 = np.percentile(frames,99)

mean = np.mean(frames)
desviation = np.std(frames)

txt.write("Percentil :")
txt.write(f"\n p1: {p1}")
txt.write(f"\n p5: {p5}")
txt.write(f"\n p25: {p25}")
txt.write(f"\n p50: {p50}")
txt.write(f"\n p75: {p75}")
txt.write(f"\n p95: {p95}")
txt.write(f"\n p99: {p99}")
txt.write(f"\n Mean: {mean}, Desviation: {desviation}")

txt.close()