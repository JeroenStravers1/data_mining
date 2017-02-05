import numpy as np

TOTAL_OCCURENCES = "total"
ZERO = 0
ONE = 1.0


def get_occurences_per_value_from_dataset(list_of_values):
    """
    :param list_of_values: a List of String values
    :return: dict of lists(indices of occurences per unique value in list_of_values)
    """
    unique_values_occurences = dict()
    unique_values_occurences[TOTAL_OCCURENCES] = ZERO
    for index, row_value in enumerate(list_of_values):
        try:
            current_value_index_list = unique_values_occurences[row_value]
            current_value_index_list.append(index)
            unique_values_occurences[row_value] = current_value_index_list
            unique_values_occurences[TOTAL_OCCURENCES] += ONE
        except KeyError:
            unique_values_occurences[row_value] = [index]
            unique_values_occurences[TOTAL_OCCURENCES] += ONE
    return unique_values_occurences


def get_entropy_from_values(list_of_values):
    """
    :param list_of_values:
    :return:
    """
    unique_values_occurences = get_occurences_per_value_from_dataset(list_of_values)
    entropy = ZERO
    for list_of_indices in unique_values_occurences:
        if list_of_indices != TOTAL_OCCURENCES:
            number_of_occurences_for_value = len(unique_values_occurences[list_of_indices])
            relative_occurence_of_value = number_of_occurences_for_value / unique_values_occurences[
                TOTAL_OCCURENCES]
            entropy_for_value = (-1 * relative_occurence_of_value) * np.log2(relative_occurence_of_value)
            entropy += entropy_for_value
    return entropy


def get_classes_for_feature_indices(feature_indices, classification_values):
    """
    :param feature_indices: list of indices of selected features
    :param classification_values: list of ALL class values
    :return:
    """
    list_of_classes = list()
    for index in feature_indices:
        list_of_classes.append(classification_values[index])
    return list_of_classes


def calculate_gain_of_feature_on_class(features, classes):
    """
    :param features: List of String features to calculate gain of
    :param classes: List of String classes to classify
    :return: the gain of the feature
    """
    classes_entropy = get_entropy_from_values(classes)
    indices_per_feature = get_occurences_per_value_from_dataset(features)
    gain = classes_entropy
    # get classes per subset of features
    for feature_index_subset in indices_per_feature:
        if feature_index_subset != TOTAL_OCCURENCES:
            current_subset = indices_per_feature[feature_index_subset]
            relative_size_of_subset = len(current_subset) / indices_per_feature[TOTAL_OCCURENCES]
            current_subset_classes = get_classes_for_feature_indices(current_subset, classes)
            current_subset_entropy = get_entropy_from_values(current_subset_classes)
            current_subset_gain = -relative_size_of_subset * current_subset_entropy
            gain += current_subset_gain
    return gain

# FIXME: your sheets for gain/entropy calculation (lecture and Exercises 3) contain errors. This cost me a lot of time.
# FIXME: the Exercises sheets list entropy as part of gain as ".0940 - entropy...", while the 0.940 is not part of this calculation here.
# FIXME: Both math and programming require a literal notation style!

"""
GAIN:
Entropy(class)
    - (feature subset size relative to feature set) * (entropy(class values for feature subset))
    - (feature subset size relative to feature set) * (entropy(class values for feature subset))
    etc for each feature subset

class   feature
k       o
l       p
k       p
k       o

Entropy(class)
    - (3/4) * Entropy([o,p,o])
    - (1/4) * Entropy([p])
"""


if __name__ == "__main__":
    tts2classes = ["a","a","b","b","b","a","b","a","b","b","b","b","b","a"]
    tts2 =        ["30","30","3040","40","40","40","3040","30","30","40","30","3040","3040","40"]
    print(calculate_gain_of_feature_on_class(tts2, tts2classes)) # should produce ~0.247
