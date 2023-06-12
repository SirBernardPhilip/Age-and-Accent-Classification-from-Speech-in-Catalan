import os
import csv
import pickle
import pandas as pd
import numpy as np
main_path = "/home/usuaris/veussd/DATABASES/Common_Voice/cv11.0/edat/ca/features_dev/"
#files = os.listdir(path)

tsv = pd.read_csv("/home/usuaris/veussd/DATABASES/Common_Voice/cv11.0/original/ca/dev.tsv", sep='\t')
tsv.dropna(subset=['age'], inplace=True)
print(tsv)
paths = tsv['path'].to_numpy()
ages = tsv['age'].to_numpy()
listages = []

for i in range(0,ages.size):
    if ages[i] in listages:
        a = 1
    else: 
        listages.append(ages[i])

print(listages)
for i in range(0,ages.size):
    if ages[i] == "teens":
        ages[i]=0
    if ages[i] == "twenties":
        ages[i]=1
    if ages[i] == "thirties":
        ages[i]=2

    if ages[i] == "fourties":
        ages[i]=3

    if ages[i] == "fifties":
        ages[i]=4

    if ages[i] == "sixties":
        ages[i]=5

    if ages[i] == "seventies":
        ages[i]=6

    if ages[i] == "eighties":
        ages[i]=7

    if ages[i] == "nineties":
        ages[i]=8

filetxt = open("/home/usuaris/veussd/DATABASES/Common_Voice/cv11.0/edat/ca/devfiles.lst","w")

for i in range(0,ages.size):
    namefile = str(paths[i]).split(".")[0]
    picklename = namefile+".pickle"
    filetxt.write(main_path+namefile+"    "+str(ages[i])+"    "+"-1" + '\n')

filetxt.close()
