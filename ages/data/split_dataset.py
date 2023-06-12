import os
import pandas as pd

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

def _calculate_data_set_sizes(total_size):
    # Find maximum size for the training data set in accord with sample theory
    for train_size in range(total_size, 0, -1):
        calculated_sample_size = int(sample_size(train_size))
        if 2 * calculated_sample_size + train_size <= total_size:
            dev_size = calculated_sample_size
            test_size = calculated_sample_size
            break
    return train_size, dev_size, test_size

calcula =  "/home/usuaris/veussd/DATABASES/Common_Voice/cv11.0/original/ca/validated.tsv"
pc = "C:/Users/david/Desktop/MASTER/TFM/CommonVoice11/validated.tsv"
validated = pd.read_csv(pc, sep='\t')
validated.dropna(subset=['age'], inplace=True)

print(validated)

train = pd.DataFrame(columns=validated.columns)
dev = pd.DataFrame(columns=validated.columns)
test = pd.DataFrame(columns=validated.columns)

train_size = dev_size = test_size = 0

if (len(validated) > 0):
    # Determine train, dev, and test sizes
    train_size, dev_size, test_size = _calculate_data_set_sizes(len(validated))
    print(train_size)
    print(dev_size)
    print(test_size)