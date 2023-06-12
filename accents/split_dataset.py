"""
Module to create the dataset
"""
import pandas as pd
from tqdm import tqdm
import numpy as np

SPLIT_NAME = "split1"
FEATURES_DIR = "/home/usuaris/veussd/DATABASES/Common_Voice/cv11.0/ca/features/"
OUTPUT_TSV_FILES = ["./"+SPLIT_NAME+"/dev.tsv", "./"+SPLIT_NAME+"/train.tsv", "./"+SPLIT_NAME+"/test.tsv"]
INPUT_TSV_FILES = "./validated_final.tsv"


USER_COUNT_LIMITS = 100
CENTRAL_DROP = 0.5
TRAIN_PROPORTION = 0.75


DISCARDED_CLIPS = [
    "common_voice_ca_30831868.mp3",
    "common_voice_ca_17647697.mp3",
    "common_voice_ca_17513308.mp3",
    "common_voice_ca_17508368.mp3",
    "common_voice_ca_17655242.mp3",
    "common_voice_ca_17623010.mp3",
    "common_voice_ca_17647724.mp3",
    "common_voice_ca_17647714.mp3",
    "common_voice_ca_17657376.mp3",
    "common_voice_ca_17622811.mp3",
    "common_voice_ca_17627676.mp3",
    "common_voice_ca_17505046.mp3",
    "common_voice_ca_17647643.mp3",
    "common_voice_ca_17647716.mp3",
    "common_voice_ca_17567673.mp3",
    "common_voice_ca_31216769.mp3",
    "common_voice_ca_17649593.mp3",
    "common_voice_ca_30776150.mp3",
    "common_voice_ca_17622574.mp3",
    "common_voice_ca_17647684.mp3",
    "common_voice_ca_31105338.mp3",
]

PRESET_ACCENTS = [
    "central",
    "nord-occidental",
    "nord-oriental",
    "oriental",
    "balear",
    "valencià",
    "septentrional",
    "occidental",
    "aprenent (recent, des del castellà)",
    "aprenent (recent, des d'altres llengües)",
]

DISCARDED_ACCENTS = [
    "aprenent (recent, des del castellà)",
    "aprenent (recent, des d'altres llengües)",
]

FINAL_ACCENTS = ["central", "nord-occidental", "valencià", "balear", "septentrional"]


def sample_size(population_size):
    """Calculates the sample size.
    Calculates the sample size required to draw from a population size `population_size`
    with a confidence level of 99% and a margin of error of 1%.
    Args:
      population_size (int): The population size to draw from.
    """
    margin_of_error = 0.01
    fraction_picking = 0.50
    z_score = 2.58  # Corresponds to confidence level 99%
    numerator = (z_score ** 2 * fraction_picking * (1 - fraction_picking)) / (
        margin_of_error ** 2
    )
    denominator = 1 + (z_score ** 2 * fraction_picking * (1 - fraction_picking)) / (
        margin_of_error ** 2 * population_size
    )
    return numerator / denominator


def calculate_data_set_sizes(total_size: int):
    """
    Find maximum size for the training data set in accord with sample theory
    """
    # for train_size in range(total_size, 0, -1):
    #     calculated_sample_size = int(sample_size(train_size))
    #     if (2 * calculated_sample_size + train_size <= total_size) and (
    #         train_size <= total_size * 0.8
    #     ):
    #         dev_size = calculated_sample_size
    #         test_size = calculated_sample_size
    #         break
    dev_size = int(total_size * 0.125)
    test_size = dev_size
    train_size = total_size - 2 * dev_size
    return train_size, dev_size, test_size


def limit_users_and_count(valid_data: pd.DataFrame):
    # Remove duplicate sentences while maintaining maximal user diversity at the frame's start (TODO: Make addition of user_sentence_count cleaner)
    speaker_counts = valid_data["client_id"].value_counts()
    speaker_counts = speaker_counts.to_frame().reset_index()
    speaker_counts.columns = ["client_id", "user_sentence_count"]
    valid_data = valid_data.join(speaker_counts.set_index("client_id"), on="client_id")
    valid_data = valid_data.sort_values(["user_sentence_count", "client_id"])
    validated = valid_data.groupby("sentence").head(1)

    validated = validated.sort_values(
        ["user_sentence_count", "client_id"], ascending=False
    )
    validated = validated.drop(columns="user_sentence_count")
    valid_data = valid_data.drop(columns="user_sentence_count")

    final_validated = pd.DataFrame(columns=validated.columns)

    if len(validated) > 0:
        # Determine train, dev, and test sizes
        # Split into train, dev, and test datasets
        continous_client_index, uniques = pd.factorize(validated["client_id"])
        validated["continous_client_index"] = continous_client_index

        for i in range(max(continous_client_index), -1, -1):
            final_validated = pd.concat(
                [
                    final_validated,
                    validated[validated["continous_client_index"] == i].head(USER_COUNT_LIMITS),
                ],
                sort=False,
            )

    final_validated = final_validated.drop(
        final_validated[final_validated["accents"].eq("central")].sample(frac=CENTRAL_DROP).index
    )
    return final_validated

# https://stackoverflow.com/questions/56872664/complex-dataset-split-stratifiedgroupshufflesplit
def post_process_valid_data(df_main):
    df_main = df_main.reindex(np.random.permutation(df_main.index)) # shuffle dataset

    # create empty train, val and test datasets
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    df_test = pd.DataFrame()

    hparam_mse_wgt = 0.1 # must be between 0 and 1
    assert(0 <= hparam_mse_wgt <= 1)
    train_proportion = TRAIN_PROPORTION # must be between 0 and 1
    assert(0 <= train_proportion <= 1)
    val_test_proportion = (1-train_proportion)/2

    subject_grouped_df_main = df_main.groupby(['client_id'], sort=False, as_index=False)
    category_grouped_df_main = df_main.groupby('accents').count()[['client_id']]/len(df_main)*100

    def calc_mse_loss(df):
        grouped_df = df.groupby('accents').count()[['client_id']]/len(df)*100
        df_temp = category_grouped_df_main.join(grouped_df, on = 'accents', how = 'left', lsuffix = '_main')
        df_temp.fillna(0, inplace=True)
        df_temp['diff'] = (df_temp['client_id_main'] - df_temp['client_id'])**2
        mse_loss = np.mean(df_temp['diff'])
        return mse_loss

    i = 0
    for _, group in subject_grouped_df_main:

        if (i < 3):
            if (i == 0):
                df_train = df_train.append(pd.DataFrame(group), ignore_index=True)
                i += 1
                continue
            elif (i == 1):
                df_val = df_val.append(pd.DataFrame(group), ignore_index=True)
                i += 1
                continue
            else:
                df_test = df_test.append(pd.DataFrame(group), ignore_index=True)
                i += 1
                continue

        mse_loss_diff_train = calc_mse_loss(df_train) - calc_mse_loss(df_train.append(pd.DataFrame(group), ignore_index=True))
        mse_loss_diff_val = calc_mse_loss(df_val) - calc_mse_loss(df_val.append(pd.DataFrame(group), ignore_index=True))
        mse_loss_diff_test = calc_mse_loss(df_test) - calc_mse_loss(df_test.append(pd.DataFrame(group), ignore_index=True))

        total_records = len(df_train) + len(df_val) + len(df_test)

        len_diff_train = (train_proportion - (len(df_train)/total_records))
        len_diff_val = (val_test_proportion - (len(df_val)/total_records))
        len_diff_test = (val_test_proportion - (len(df_test)/total_records)) 

        len_loss_diff_train = len_diff_train * abs(len_diff_train)
        len_loss_diff_val = len_diff_val * abs(len_diff_val)
        len_loss_diff_test = len_diff_test * abs(len_diff_test)

        loss_train = (hparam_mse_wgt * mse_loss_diff_train) + ((1-hparam_mse_wgt) * len_loss_diff_train)
        loss_val = (hparam_mse_wgt * mse_loss_diff_val) + ((1-hparam_mse_wgt) * len_loss_diff_val)
        loss_test = (hparam_mse_wgt * mse_loss_diff_test) + ((1-hparam_mse_wgt) * len_loss_diff_test)

        if (max(loss_train,loss_val,loss_test) == loss_train):
            df_train = df_train.append(pd.DataFrame(group), ignore_index=True)
        elif (max(loss_train,loss_val,loss_test) == loss_val):
            df_val = df_val.append(pd.DataFrame(group), ignore_index=True)
        else:
            df_test = df_test.append(pd.DataFrame(group), ignore_index=True)

        print ("Group " + str(i) + ". loss_train: " + str(loss_train) + " | " + "loss_val: " + str(loss_val) + " | " + "loss_test: " + str(loss_test) + " | ")
        i += 1

    return df_train, df_val, df_test


def map_accents(accents: str) -> str:
    """
    Function to map accents from the accents column
    """
    if accents in FINAL_ACCENTS:
        return accents
    if (accents == "nord-oriental") or (accents == "oriental"):
        return "central"
    if accents == "occidental":
        return "valencià"


def stringToIndex(accent: str):
    """
    Function to map accents to indexes
    """
    if accent == "central":
        return "0"
    if accent == "nord-occidental":
        return "1"
    if accent == "valencià":
        return "2"
    if accent == "balear":
        return "3"
    if accent == "septentrional":
        return "4"


def split_dataset():
    """
    Function to split the dataset into train, dev and test
    """
    print("Reading and parsing...")
    # We read the data from the tsv file
    data = pd.read_csv(INPUT_TSV_FILES, sep="\t", dtype=str)
    # We filter the accents we do not want
    data_filtered_aux = data[~data["accents"].isin(DISCARDED_ACCENTS)].copy()
    data_filtered = data_filtered_aux[
        ~data_filtered_aux["path"].isin(DISCARDED_CLIPS)
    ].copy()
    # We map the accents to the final accents
    data_filtered["accents"] = data_filtered["accents"].map(map_accents)
    print("Splitting...")
    # We split the data into train, dev and test
    # train_size, dev_size, test_size = calculate_data_set_sizes(len(data_filtered))
    # print("Train size: ", train_size)
    # print("Dev size: ", dev_size)
    # print("Test size: ", test_size)
    # train, dev_test = train_test_split(
    #     data_filtered, train_size=train_size, stratify=data_filtered["accents"]
    # )
    # dev, test = train_test_split(
    #     dev_test, train_size=dev_size, stratify=dev_test["accents"]
    # )
    final_validated = limit_users_and_count(data_filtered.copy())
    train, dev, test = post_process_valid_data(final_validated.copy())
    print("Classes train")
    print(train["accents"].value_counts(normalize=True))
    print(train["accents"].value_counts(normalize=False))
    print("Classes dev")
    print(dev["accents"].value_counts(normalize=True))
    print(dev["accents"].value_counts(normalize=False))
    print("Classes test")
    print(test["accents"].value_counts(normalize=True))
    print(test["accents"].value_counts(normalize=False))
    # We save the data into tsv files
    print("Saving...")
    train.to_csv(OUTPUT_TSV_FILES[1], sep="\t", index=False)
    dev.to_csv(OUTPUT_TSV_FILES[0], sep="\t", index=False)
    test.to_csv(OUTPUT_TSV_FILES[2], sep="\t", index=False)
    # Creating path fils
    with open("./"+SPLIT_NAME+"train_files.lst", "w") as file:
        for _, row in tqdm(list(train.iterrows())):
            dest_path = FEATURES_DIR + str(row["path"]).replace("mp3", "pickle")
            file.write(dest_path + " " + stringToIndex(row["accents"]) + " -1\n")
    with open("./"+SPLIT_NAME+"dev_files.lst", "w") as file:
        for _, row in tqdm(list(dev.iterrows())):
            dest_path = FEATURES_DIR + str(row["path"]).replace("mp3", "pickle")
            file.write(dest_path + " " + stringToIndex(row["accents"]) + " -1\n")
    with open("./"+SPLIT_NAME+"test_files.lst", "w") as file:
        for _, row in tqdm(list(test.iterrows())):
            dest_path = FEATURES_DIR + str(row["path"]).replace("mp3", "pickle")
            file.write(dest_path + " " + stringToIndex(row["accents"]) + " -1\n")
    print("Done!")


if __name__ == "__main__":
    split_dataset()
