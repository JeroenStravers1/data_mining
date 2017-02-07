import pandas as pd
from supervised.supervised_classifier import SupervisedClassifier


class NaiveBayesClassifier(SupervisedClassifier):
    """
    Trains and tests a Naive Bayes Classifier. The dataset used here contains qualitative data.
    """ #FIXME: elaborate

    _MEAN = "_mean"
    _STD = "_standard_deviation"

    def __init__(self):
        self.trained_classifier = dict()

    def train(self, trainset):
        """
        Train a Naive Bayes classifier on a qualitative csv dataset with header. A "class" column is required. Stores
        a dict per class in self.trained_classifier, each class dict contains a further dict per feature. Each of
        these dicts contains feature_value: relative_size.
        :param trainset: Path to the dataset
        :return:
        """#FIXME: docstring
        df_trainset = self.read_csv_as_dataframe(trainset)
        self._create_probability_distributions_per_class(df_trainset)

    def _create_probability_distributions_per_class(self, df_trainset):
        divided_dataset = self._divide_dataset_by_class(df_trainset)
        features_in_dataset = df_trainset.columns.values
        self._allocate_dicts_for_feature_values(divided_dataset)
        self._get_feature_value_division_per_class(features_in_dataset, divided_dataset)

    def _divide_dataset_by_class(self, df_trainset):
        """
        Divide the dataset into subset DFs based on class. The resulting subset DFs are stored in a dict with the class
        as key
        :param df_trainset: the original dataset as DF
        :return: dict containing the divided dataset DFs
        """
        classes_in_dataset = pd.Series.unique(df_trainset[self._CLASS]).tolist()
        dataset_divided_by_classes = dict()
        for distinct_class in classes_in_dataset:
            class_subset = df_trainset[df_trainset[self._CLASS] == distinct_class]
            dataset_divided_by_classes[distinct_class] = class_subset
        return dataset_divided_by_classes

    def _allocate_dicts_for_feature_values(self, divided_dataset):
        """
        create dicts in self.trained_classifier with key = class for each class
        :param divided_dataset: the dataset dividedd by class
        """
        for distinct_class in divided_dataset:
            self.trained_classifier[distinct_class] = dict()

    def _get_feature_value_division_per_class(self, list_of_features, divided_dataset):
        """
        store relative size of feature values per class in self.trained_classifier
        :param list_of_features: list of unique feature names
        :param divided_dataset: original dataset split into class-specific dataframes
        """
        for distinct_class in divided_dataset:
            class_subset = divided_dataset[distinct_class]
            for feature in list_of_features:
                if feature != self._CLASS:
                    feature_relative_values = dict()
                    feature_std = class_subset[feature].value_counts(normalize=True)
                    values = feature_std.index
                    for value in values:
                        feature_relative_values[value] = feature_std.loc[value]
                    self.trained_classifier[distinct_class][feature] = feature_relative_values

    def test(self, testset):
        """
        :param testset:
        :return:
        """
        df_trainset = self.read_csv_as_dataframe(testset)
        pass

if __name__ == "__main__":
    test_nbc = NaiveBayesClassifier()

    test_nbc.train("../generated_sets/2017-01-13T13.55.08.756917_train_4911.csv")
    test_nbc.test("../generated_sets/2017-01-13T13.55.08.756917_test_4911.csv")

    # test_nbc.train("../generated_sets/2017-02-07T09.46.35.650308_train_9389.csv")
    # test_nbc.test("../generated_sets/2017-02-07T09.46.35.650308_test_9389.csv")

    # test_nbc.train("../generated_sets/2017-02-07T10.31.45.714731_train_8176.csv")
    # test_nbc.test("../generated_sets/2017-02-07T10.31.45.714731_test_8176.csv")