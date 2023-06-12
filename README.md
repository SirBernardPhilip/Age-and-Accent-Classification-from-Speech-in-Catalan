# Age and Accent Classification from Speech in Catalan

This repository contains a set of tools used during the Master's Thesis of students from the [Universitat Politècnica de Catalunya](https://www.upc.edu/ca). The thesis were centered on speaker characterization using the [Common Voice](https://commonvoice.mozilla.org/) dataset.

When using audio it is generally advisable to convert all files to the same sampling rate. This can be achieved with the `convert.sh` script. You should just change the directory variable accordingly and it will perform the conversion (be advised, it can be a lengthy process).

## Installation

This repository has been created using python3.6 but newer versions will likely work. You can find the python3 dependencies on requirements.txt. You can install them by running:

```bash

pip install -r requirements.txt

```

## Accents

This section of the repository contains only some auxiliary tools for the tasks of accent classification. They go in conjuction with the [Double Attention Accent Classification](https://github.com/SirBernardPhilip/DoubleAttentionAccentClassification.git) repository, forked from [Double Attention Speaker Verification](https://github.com/miquelindia90/DoubleAttentionSpeakerVerification).


### move_accents.py

This script will attempt to filter speaker without accents and map all the other speakers in the dataset to their corresponding accent in our preset list. It requires that the `validated.tsv` file is present in the `DATASET_DIR` directory. It will create a new file `validated_final.tsv` with the same format as the original one but with the accent column cleaned and the old accents in "accents_raw". To execute it you only have to adapt the `DATASET_DIR` variable and run:

```bash
python move_accents.py
```

### split_dataset.py

This script will create the train, validation and test sets from the `validated_final.tsv` file (it will also filter some corrupted or invalid files). It requires that the `validated_final.tsv` file is present in the directory. It will create a new file `train.tsv`, `validation.tsv` and `test.tsv` with the format needed to train in the [Double Attention Speaker Verification](https://github.com/SirBernardPhilip/DoubleAttentionAccentClassification.git) repository. To execute it you only have to adapt the `SPLIT_NAME` and `FEATURES_DIR` variables and run:

```bash
python split_dataset.py
```

There are some hyperparameters that you can change:
* `USER_COUNT_LIMITS`: how many samples per user we can use.
* `CENTRAL_DROP`: what percentage of samples in the Central dialect we drop.
* `TRAIN_PROPORTION`: what proportion of the total dataset will be used as training samples.

Once this script has been executed you can follow the instructions in the other [repository](https://github.com/SirBernardPhilip/DoubleAttentionAccentClassification.git) to train the model.


### features_statistics.py

This script will compute statistics on the length of the extracted features and the number of speakers per accent. It requires that the features have been extracted beforehand following the instructions [here](https://github.com/SirBernardPhilip/DoubleAttentionSpeakerVerification#feature-extraction) and that the scripts `split_dataset.py` and `move_accents.py` have been run. To execute it you only have to adapt the `TSV_DIR` and `FEATURES_DIR` variables and run:

```bash
python features_statistics.py
```

## Ages

This section of the repository contains only some auxiliary tools for the tasks of age classification. They go in conjuction with the repository [Double Attention Age Classification](https://github.com/SirBernardPhilip/DoubleAttentionAgeClassification), variation of the originally implemented [here](https://github.com/miquelindia90/DoubleAttentionSpeakerVerification).

We have less context on the usage of this scripts so we have split them into folders that group them by their usage but we advise to read the code to understand what they do and how to use them.

### data
Contains a smaller version of the `split_dataset.py` script that was used for the age classification task. It will create the train, validation and test sets from a `validated.tsv` file.

The file `trainfile.py` will parse the class names in string format into a numeric usable format.

### misc
Contains different scripts to be used in conjunction with the training repository. 

### statistics

Contains different scripts that can be used for the analysis of the dataset. They are not necessary for the training of the model but they can be useful to understand the dataset.
