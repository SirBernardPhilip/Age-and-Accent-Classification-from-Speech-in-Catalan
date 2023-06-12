import os
import pickle
import matplotlib.pyplot as plt
import numpy as np 

txt1 = open("/home/usuaris/veu/david.linde/CommonVoice11/statsu100_en_1.txt","w")


path = "/home/usuaris/veu/david.linde/features/features_en/"
files = os.listdir(path=path)

files1 = files[0:300000]
files2 = files[300000:700000]
files3 = files[700000:len(files)]


frames1 = []
i = 0
for file in files1:
    try:
        with open(path+file,'rb') as pickleFile: 
            features = pickle.load(pickleFile)
        if np.transpose(features).shape[0] <= 100:
            frames1.append(file)
    except pickle.UnpicklingError:
        print(path+file)
    print(str(i))
    i+=1
string = " ".join(frames1)   
txt1.write(string)
txt1.close()

txt2 = open("/home/usuaris/veu/david.linde/CommonVoice11/statsu100_en_2.txt","w")

frames2 = []
for file in files2:
    try:
        with open(path+file,'rb') as pickleFile: 
            features = pickle.load(pickleFile)
        if np.transpose(features).shape[0] <= 100:
            frames2.append(file)
    except pickle.UnpicklingError:
        print(path+file)
    print(str(i))
    i+=1
string = " ".join(frames2)   
txt2.write(string)
txt2.close()


frames3 = []
txt3 = open("/home/usuaris/veu/david.linde/CommonVoice11/statsu100_en_3.txt","w")
for file in files3:
    try:
        with open(path+file,'rb') as pickleFile: 
            features = pickle.load(pickleFile)
        if np.transpose(features).shape[0] <= 100:
            frames3.append(file)
    except pickle.UnpicklingError:
        print(path+file)
    print(str(i))
    i+=1
string = " ".join(frames3)   
txt3.write(string)
txt3.close()