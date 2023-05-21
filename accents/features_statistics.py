"""
Module to move clips from original into accents folder
"""

import os
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd

FEATURES_DIR = "/home/usuaris/veussd/DATABASES/Common_Voice/cv11.0/accents/ca/features"
TSV_DIR = "/home/usuaris/veussd/DATABASES/Common_Voice/cv11.0/accents/ca/"


def obtain_percentiles(lengths):
    print("Percentiles:")
    for i in [1, 5, 25, 50, 75, 95, 99]:
        print(i, ":", np.percentile(lengths, i))


def obtain_statistics(lengths):
    print("Mean: ", np.mean(lengths))
    print("Std: ", np.std(lengths))
    print("Max: ", np.max(lengths))
    print("Min: ", np.min(lengths))
    obtain_percentiles(lengths)
    print("# samples under 100: ", len(list(filter(lambda x: x < 100, lengths))))
    print("# samples under 200: ", len(list(filter(lambda x: x < 200, lengths))))
    print("# samples under 300: ", len(list(filter(lambda x: x < 300, lengths))))


def main(get_lengths: bool):

    if get_lengths:
        samples_train = pd.read_csv(TSV_DIR + "train.tsv", sep="\t")
        samples_dev = pd.read_csv(TSV_DIR + "dev.tsv", sep="\t")
        samples_test = pd.read_csv(TSV_DIR + "test.tsv", sep="\t")
        samples = pd.concat([samples_train, samples_dev, samples_test])
        lengths_central = []
        lengths_nord_occidental = []
        lengths_valencia = []
        lengths_balear = []
        lengths_septentrional = []
        for _, row in tqdm(samples.iterrows(), total=len(samples)):
            accent = row["accents"]
            filename = row["path"].split("/")[-1].split(".")[0]
            with open(os.path.join(FEATURES_DIR, filename + ".pickle"), "rb") as f:
                features = pickle.load(f)
            if accent == "central":
                lengths_central.append(features.shape[1])
            elif accent == "nord-occidental":
                lengths_nord_occidental.append(features.shape[1])
            elif accent == "valencià":
                lengths_valencia.append(features.shape[1])
            elif accent == "balear":
                lengths_balear.append(features.shape[1])
            elif accent == "septentrional":
                lengths_septentrional.append(features.shape[1])
        with open("lengths_central.pickle", "wb") as f:
            pickle.dump(lengths_central, f)
        with open("lengths_nord_occidental.pickle", "wb") as f:
            pickle.dump(lengths_nord_occidental, f)
        with open("lengths_valencia.pickle", "wb") as f:
            pickle.dump(lengths_valencia, f)
        with open("lengths_balear.pickle", "wb") as f:
            pickle.dump(lengths_balear, f)
        with open("lengths_septentrional.pickle", "wb") as f:
            pickle.dump(lengths_septentrional, f)
    else:
        with open("lengths_central.pickle", "rb") as f:
            lengths_central = pickle.load(f)
        with open("lengths_nord_occidental.pickle", "rb") as f:
            lengths_nord_occidental = pickle.load(f)
        with open("lengths_valencia.pickle", "rb") as f:
            lengths_valencia = pickle.load(f)
        with open("lengths_balear.pickle", "rb") as f:
            lengths_balear = pickle.load(f)
        with open("lengths_septentrional.pickle", "rb") as f:
            lengths_septentrional = pickle.load(f)

    print("General: ")
    obtain_statistics(
        lengths_central
        + lengths_nord_occidental
        + lengths_valencia
        + lengths_balear
        + lengths_septentrional
    )
    print("Central:")
    obtain_statistics(lengths_central)
    print("Nord-occidental:")
    obtain_statistics(lengths_nord_occidental)
    print("Valencia:")
    obtain_statistics(lengths_valencia)
    print("Balear:")
    obtain_statistics(lengths_balear)
    print("Septentrional:")
    obtain_statistics(lengths_septentrional)


if __name__ == "__main__":
    main(True)
