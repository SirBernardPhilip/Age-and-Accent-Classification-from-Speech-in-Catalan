import os
import pickle
import matplotlib.pyplot as plt
import numpy as np 

txt = open("/home/usuaris/veu/david.linde/CommonVoice11/stats.txt","w")
path = "/home/usuaris/veussd/DATABASES/Common_Voice/cv11.0/edat/ca/features_validated/"
files = os.listdir(path=path)
frames = []
i = 0
for file in files:
    with open(path+file,'rb') as pickleFile: 
        features = pickle.load(pickleFile)
    frames.append(np.transpose(features).shape[0])
    i+=1
    print(str(i))

num_bins = len(set(frames))
n, bins, patches = plt.hist(frames, num_bins, facecolor='blue', alpha=0.5)
plt.title("Frames audio histogram")
plt.savefig("/home/usuaris/veu/david.linde/CommonVoice11/hist_frames.jpg")

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