import pandas as pd

DATASET_DIR = "/home/usuaris/veussd/DATABASES/Common_Voice/cv11.0/ca/"

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


def parse_accents(accents: str) -> str:
    """
    Function to parse accents from the accents column
    """
    accent_list = map(lambda x: x.lower().replace(" ", "-"), accents.split(","))
    for accent in accent_list:
        if accent in PRESET_ACCENTS:
            return accent
        if any(
            map(
                lambda x: x in accent,
                [
                    "barcelon",
                    "montseny",
                    "suau",
                    "xipella",
                    "interior",
                    "badalon",
                    "pened",
                    "neutre",
                    "maresme",
                    "central",
                    "natural",
                    "catala",
                    "estàndar",
                    "oson",
                    "vall",
                    "tarra",
                    "reus",
                ],
            )
        ):
            return "central"
        if "mallor" in accent:
            return "balear"
        if (
            ("giron" in accent)
            or ("empord" in accent)
            or ("nord-oriental" in accent)
            or ("andorra" in accent)
            or (accents == "Català tancat de la plana de vc")
        ):
            return "nord-oriental"
        if (
            ("lleid" in accent)
            or ("alcarr" in accent)
            or ("priorat" in accent)
            or ("nord-occidental" in accent)
            or ("ponent" in accent)
            or ("noguera" in accent)
            or ("pallar" in accent)
        ):
            return "nord-occidental"
        if ("ebre" in accent) or ("tortosí" in accent) or ("occidental" in accent):
            return "occidental"
        if ("paris" in accent) or ("estranger" in accent):
            return "aprenent (recent, des d'altres llengües)"
        if "argentí" in accent:
            return "aprenent (recent, des del castellà)"
        if ("valenci" in accent) or ("meridional" in accent):
            return "valencià"
        if ("septentrional" in accent) or ("garrotx" in accent):
            return "septentrional"
        if "oriental" in accent:
            return "oriental"
        if accents == "Accent tancat de poble":
            return "central"  # WTF ???
        if accents == "Del Sud":
            return "occidental"  # WTF ???
        if accents == "Industrial":
            return "central"  # WTF ???
        if accents == "hola que tal, com vas?":
            return "central"  # WTF ???
    raise Exception("No accent found")


def read_and_filter(tsv_file: str) -> pd.DataFrame:
    """
    Function to read and filter the tsv file
    """
    print("Reading and filtering...")
    # We read the data from the tsv file
    data = pd.read_csv(DATASET_DIR + tsv_file, sep="\t", dtype=str)
    print(len(data))
    # We filter the rows without accents information
    data_filtered = data[data["accents"].notnull()].copy()
    print(len(data_filtered))
    # Copy accents column to accents raw for parsing
    data_filtered.loc[:, "accents_raw"] = data_filtered["accents"].copy()
    unparsed_accents = data_filtered.loc[
        ~data_filtered["accents"].isin(PRESET_ACCENTS), "accents"
    ].copy()
    print(len(unparsed_accents))
    data_filtered.loc[
        ~data_filtered["accents"].isin(PRESET_ACCENTS), "accents"
    ] = unparsed_accents.apply(parse_accents)
    return data_filtered


def move_files() -> None:
    data_with_accents = read_and_filter("validated.tsv")
    print(len(data_with_accents))
    print(data_with_accents["accents"].value_counts())
    print("Writing tsv file...")
    data_with_accents.to_csv("./validated_final.tsv", sep="\t")


if __name__ == "__main__":
    move_files()
