#import libraries
import pandas as pd
import numpy as np
import os
import csv

#paths to files
feature_of_counts_temp = "../processed_data/feature_vectors_counts_temp.csv"
feature_of_counts = "../processed_data/feature_vectors_counts.csv"
dir_of_files = "../raw_data/feature_vectors/"
known_malware_files = "../raw_data/sha256_family.csv"


#functions used

def count_feature_set(lines):
    """
    Count how many features belong to a specific set
    :param lines: features in the text file
    :return: an array containing count values for each feature set in the fils "lines" received
    """
    # Define the features and their corresponding numerical identifiers
    FEATURES_SET = {
        "feature": 1,
        "permission": 2,
        "activity": 3,
        "service_receiver": 3,
        "provider": 3,
        "service": 3,
        "intent": 4,
        "api_call": 5,
        "real_permission": 6,
        "call": 7,
        "url": 8
    }

    # Initialize a dictionary to hold the count of each feature set
    features_map = {x: 0 for x in range(1, 9)}
    # Iterate over each line in the input file
    for l in lines:
        if l != "\n":
            set = l.split("::")[0] # Extract the feature set name
            features_map[FEATURES_SET[set]] += 1 # Increment the count for the corresponding feature set
    features = []
    for i in range(1, 9):
        features.append(features_map[i])
    return features


def read_sha_files():
    """
    Reads each application file in the directory and uses the function count_feature_set to get the property count for each feature and organises it in a mutidimensional array with the filename
    :param lines: none
    :return: a multi dimensional array containing with each element having the full properties (name and feature set count) for each application data file
    """
    feature_count = []
    for filename in os.listdir(dir_of_files):
        sha_data = open(dir_of_files+ filename)
        feature_count.append([filename] + count_feature_set(sha_data))
        sha_data.close()
    return feature_count


def create_csv_for_sha_data():
    """
    Craetes a temporary file containing ONLY FEATURE SET INPUTS
    :param lines: none
    :return: none
    """
    header = ['sha256', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
    with open(feature_of_counts_temp, "wt", newline ='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(i for i in header)
        for j in read_sha_files():
            writer.writerow(j)



#Create the temporary feature set file
create_csv_for_sha_data()

"""
    map feature_vectors sha with it's corresponding output value (whether it is a malware or not)
    using the ground truth given in the sha_family file,
"""
data = pd.read_csv(known_malware_files)
# Extract the sha256 column from the data
sha_column = data["sha256"]

feature_vectors_data = pd.read_csv(feature_of_counts_temp)
# Extract the sha256 column from the feature vectors data
sha256_data = feature_vectors_data['sha256']

# Create a boolean mask indicating whether each sha256 value from feature_vectors_data is present in sha_column
mask = np.in1d(sha256_data, sha_column)


#creates the full feature vectors file containing both inputs and output (malware or not)
#this file is created as a merger of the temporary file created and the output generated above
malware = pd.DataFrame({'malware' : mask })
feature_vectors_data = feature_vectors_data.merge(malware, left_index = True, right_index = True)
feature_vectors_data.to_csv(feature_of_counts)
