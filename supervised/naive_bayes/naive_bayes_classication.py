import pandas as pd
from supervised.supervised_classifier import SupervisedClassifier


class NaiveBayesClassifier(SupervisedClassifier):
    """
    Trains and tests a Naive Bayes Classifier. The dataset used here contains qualitative data.
    """ #FIXME: elaborate

    _MEAN = "_mean"
    _STD = "_standard_deviation"
    _NO_OCCURENCES_OF_FEATURE_VALUE_IN_CLASS = 0
    _COLUMN_CLASS = 1

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
        self.trained_classifier = self._allocate_dicts_for_feature_values(self.trained_classifier, divided_dataset)
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

    def _allocate_dicts_for_feature_values(self, target_dict, divided_dataset):
        """
        create dicts in self.trained_classifier with key = class for each class
        :param divided_dataset: the dataset divided by class (will only use key values (class names)
        """
        for distinct_class in divided_dataset:
            target_dict[distinct_class] = dict()
        return target_dict

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
        df_testset = self.read_csv_as_dataframe(testset)
        features = df_testset.columns.values
        for index, datapoint in df_testset.iterrows():
            class_feature_probabilities = dict()
            class_feature_probabilities \
                = self._allocate_dicts_for_feature_values(class_feature_probabilities, self.trained_classifier)
            class_feature_probabilities = self._get_prediction_probabilities(df_testset, features,
                                                                             class_feature_probabilities, datapoint)
            prediction = self._predict_class(class_feature_probabilities, features)

            #FIXME: handle output (print + write)



    def _get_prediction_probabilities(self, df_testset, features, class_feature_probabilities, datapoint):
        """
        Get the probabilities of the current datapoint's feature values occurring per class
        :param df_testset: the testset as datagrame
        :param features: the column names in the dataset
        :param classification_probabilities: a dict container for the retrieved probabilities per feature per class of
        the datapoint belonging to each class
        :return: classification_probabilities
        """

        for feature in features:
            if feature != self._CLASS:
                datapoint_feature_value = datapoint.loc[feature]

                for possible_class_prediction in self.trained_classifier:
                    class_feature_probabilities[possible_class_prediction][feature] \
                        = self._get_individual_feature_value_class_probability(
                                self.trained_classifier[possible_class_prediction][feature],
                                datapoint_feature_value)
        return class_feature_probabilities

    def _get_individual_feature_value_class_probability(self, current_option_feature_probabilities,
                                                        datapoint_feature_value):
        """
        Get the probability of the datapoint's current feature value occurring for the current class
        :param current_option_feature_probabilities: a dict of feature value probabilities for the current class
        :param datapoint_feature_value: the actual feature value of the datapoint
        :return: the chance of the datapoint feature value occurring
        """
        try:
            return current_option_feature_probabilities[datapoint_feature_value]
        except KeyError:
            return self._NO_OCCURENCES_OF_FEATURE_VALUE_IN_CLASS

    def _predict_class(self, class_feature_probabilities, features):
        """
        Predicts the most probable class for the datapoint based on its feature values
        :param class_feature_probabilities: the list of class-specific feature probabilities
        :param features: list of all columns (features)
        :return: string predicted class
        """
        prediction_scores = self._get_average_prediction_probabilities(class_feature_probabilities, features)
        prediction = ""
        best_prediction_score = 0.0
        for class_name in prediction_scores:
            if prediction_scores[class_name] > best_prediction_score:
                prediction = class_name
        return prediction

    def _get_average_prediction_probabilities(self, class_feature_probabilities, features):
        """
        Condenses the feature-specific probabilities per class to an average chance per class (chance of the
        current datapoint belonging to each class, based on its feature values)
        :param class_feature_probabilities: the list of class-specific feature probabilities
        :param features: list of all columns (features)
        :return: dict with classname: average_chance
        """
        average_chances = dict()
        number_of_features = len(features) - self._COLUMN_CLASS
        for possible_class in class_feature_probabilities:
            current_class_total_prediction_chance = 0.0
            for feature_value in class_feature_probabilities[possible_class]:
                current_class_total_prediction_chance += class_feature_probabilities[possible_class][feature_value]
            average_prediction_chance = current_class_total_prediction_chance / number_of_features
            average_chances[possible_class] = average_prediction_chance
        return average_chances


    def _process_test_results(self, test_results):
        """
        Store the classified rows and total accuracy data in an output file
        :param test_results: dict containing the individual row results and accuracy totals
        """
        timestamp = datetime.datetime.utcnow()
        formatted_timestamp = timestamp.isoformat().replace(":", ".")
        file_name = self._OUTPUT_PATH + formatted_timestamp + self._METHOD + self._OUTPUT_FILE_TYPE
        with open(file_name, "w") as initialised_output_file:
            initialised_output_file.write(self._OUTPUT_HEADER + "\n")
        with open(file_name, "a") as output_file:
            number_of_predictions = len(test_results) - self._GLOBAL_ACCURACY_DATA
            for i in range(0, number_of_predictions):
                output_rule = str(i) + "," + test_results[i]
                output_file.write(output_rule + "\n")

            output_file.write(self._TOTAL_DATAPOINTS
                              + str(test_results[self._TOTAL_DATAPOINTS]) + "\n")
            output_file.write(self._NUMBER_CLASSIFIED_CORRECTLY
                              + str(test_results[self._NUMBER_CLASSIFIED_CORRECTLY]) + "\n")
            output_file.write(self._NUMBER_CLASSIFIED_INCORRECTLY
                              + str(test_results[self._NUMBER_CLASSIFIED_INCORRECTLY]) + "\n")
            output_file.write(self._UNSEEN_COMBINATION
                              + str(test_results[self._UNSEEN_COMBINATION]) + "\n")
            output_file.write(self._ACCURACY
                              + str(test_results[self._ACCURACY]))

        self._print_results_to_console(test_results)


    def _print_results_to_console(self, test_results):
        """
        Display the classification results in the console
        :param test_results: dict containing the results
        """
        print(self._METHOD + "\n")
        print(self._TOTAL_DATAPOINTS + str(test_results[self._TOTAL_DATAPOINTS]))
        print(self._NUMBER_CLASSIFIED_CORRECTLY + str(test_results[self._NUMBER_CLASSIFIED_CORRECTLY]))
        print(self._NUMBER_CLASSIFIED_INCORRECTLY + str(test_results[self._NUMBER_CLASSIFIED_INCORRECTLY]))
        print(self._UNSEEN_COMBINATION + str(test_results[self._UNSEEN_COMBINATION]))
        print(self._ACCURACY + str(test_results[self._ACCURACY]))


if __name__ == "__main__":
    test_nbc = NaiveBayesClassifier()

    test_nbc.train("../generated_sets/2017-01-13T13.55.08.756917_train_4911.csv")
    test_nbc.test("../generated_sets/2017-01-13T13.55.08.756917_test_4911.csv")

    # test_nbc.train("../generated_sets/2017-02-07T09.46.35.650308_train_9389.csv")
    # test_nbc.test("../generated_sets/2017-02-07T09.46.35.650308_test_9389.csv")

    # test_nbc.train("../generated_sets/2017-02-07T10.31.45.714731_train_8176.csv")
    # test_nbc.test("../generated_sets/2017-02-07T10.31.45.714731_test_8176.csv")