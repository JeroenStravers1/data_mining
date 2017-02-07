from __future__ import division
import supervised.decision_tree.nodes as nodes
from supervised.supervised_classifier import SupervisedClassifier
import supervised.decision_tree.entropy_calculation as calc_entropy
import pandas as pd
import datetime


class ID3DecisionTree(SupervisedClassifier):
    """
    Trains an ID3 Decision Tree classifier based on a training dataset in csv format, tests the resulting tree on
    a test dataset (also in csv). The outcome (total accuracy, original class per row, classified class per row)
    is stored in the ../classification_output directory, the total accuracy is printed to the console.

    Any test data combinations that do not have matching representations in the training data (are unknown/unseen)
    are classified as "unknown". These records are taken into account when calculating the total accuracy, but the
    amount of "unknown" classifications is recorded separately.
    """
    _PURE = 1
    _CLASS = "class"
    _PREDICTED_CLASS = 0
    _NO_OFFSPRING = 0
    _UNSEEN_COMBINATION = "unknown "
    _METHOD = "_ID3"
    _OUTPUT_FILE_TYPE = ".txt"
    _OUTPUT_PATH = "../classification_output/"
    _OUTPUT_HEADER = "index, true class, predicted class"
    _ACCURACY = "accuracy "
    _TOTAL_DATAPOINTS = "total_datapoints "
    _NUMBER_CLASSIFIED_CORRECTLY = "correctly_classified "
    _NUMBER_CLASSIFIED_INCORRECTLY = "incorrectly_classified "
    _GLOBAL_ACCURACY_DATA = 5

    def __init__(self):
        self.root_node = nodes.Node()

    def train(self, trainset):
        """
        Train an ID3 Decision Tree, based on a training set.
        :param trainset: a csv file, first row are the headers, "class" column required
        """
        df_trainset = self.read_csv_as_dataframe(trainset)
        self._create_nodes_from_dataset(df_trainset, self.root_node)

    def _create_nodes_from_dataset(self, dataset_as_df, node):
        """
        populate a decision tree recursively, based on the feature with the highest gain
        :param dataset_as_df: the (sub)set used for the gain calculation of the current tree
        :param node: the current node in the tree
        :return: if the dataset is pure the node is a terminal node. Store the corresponding class in this node.
        """
        if self._is_pure(dataset_as_df[self._CLASS]):
            subset_class = pd.Series.unique(dataset_as_df[self._CLASS]).tolist()
            node.set_class(subset_class[self._PREDICTED_CLASS])
            return
        highest_gain_feature = self._get_highest_gain(dataset_as_df)
        feature_values = pd.Series.unique(dataset_as_df[highest_gain_feature]).tolist()
        node.set_decision_rule(highest_gain_feature)
        for feature_value in feature_values:
            subset = dataset_as_df[dataset_as_df[highest_gain_feature] == feature_value]
            child_node = nodes.Node()
            node.add_children(feature_value, child_node)
            self._create_nodes_from_dataset(subset, child_node)

    def _get_highest_gain(self, dataset_as_df):
        """
        get the feature with the highest gain value from a dataset in dataframe format. Requires a "class" feature
        in the dataset. Converts feature values to lists, calculates gain and selects the feature with the highest
        gain.
        :param dataset_as_df: the dataset
        :return: string feature name with highest gain
        """
        most_gain_feature = ""
        most_gain_value = float("-inf")
        features_in_dataset = dataset_as_df.columns.values
        classes_in_dataset = dataset_as_df[self._CLASS].tolist()
        for feature in features_in_dataset:
            if feature != self._CLASS:
                feature_values = dataset_as_df[feature].tolist()
                feature_gain = calc_entropy.calculate_gain_of_feature_on_class(feature_values, classes_in_dataset)
                if feature_gain > most_gain_value:
                    most_gain_feature = feature
                    most_gain_value = feature_gain
        return most_gain_feature

    def _is_pure(self, subset):
        """
        Check if a list contains only identical values
        :param subset: list of features to compare
        :return: Boolean
        """
        values_in_subset = len(set(subset))
        return values_in_subset == self._PURE

    def test(self, testset):
        """
        Test the generated decision tree by using it to classify a test set
        :param testset:
        :return:
        """
        classification_results = dict()
        df_testset = self.read_csv_as_dataframe(testset)
        amount_of_unseen_datapoints = 0
        amount_of_correct_classifications = 0.0
        amount_of_incorrect_classifications = 0
        total_number_of_datapoints = len(df_testset.index)

        for index, row in df_testset.iterrows():
            true_class = row.loc[self._CLASS]
            tree_classification = self._traverse_tree(row)
            row_classification_result = true_class + "," + tree_classification
            classification_results[index] = row_classification_result
            if tree_classification == self._UNSEEN_COMBINATION:
                amount_of_unseen_datapoints += 1
            elif tree_classification == true_class:
                amount_of_correct_classifications += 1
            else:
                amount_of_incorrect_classifications += 1

        total_accuracy = amount_of_correct_classifications / total_number_of_datapoints
        classification_results[self._ACCURACY] = total_accuracy
        classification_results[self._TOTAL_DATAPOINTS] = total_number_of_datapoints
        classification_results[self._NUMBER_CLASSIFIED_CORRECTLY] = int(amount_of_correct_classifications)
        classification_results[self._NUMBER_CLASSIFIED_INCORRECTLY] = amount_of_incorrect_classifications
        classification_results[self._UNSEEN_COMBINATION] = amount_of_unseen_datapoints
        self._process_test_results(classification_results)

    def _traverse_tree(self, data_point):
        """
        Use the generated decision tree to classify the data point.
        :param data_point: a pd.Series instance of a row in the dataset
        :return: the classification. Returns "unknown" if the datapoint contains an unseen combination of features
        """
        try:
            current_node = self.root_node
            while len(current_node.get_children()) > self._NO_OFFSPRING:
                feature = current_node.get_decision_rule()
                decision_data = data_point.loc[feature]
                node_children = current_node.get_children()
                current_node = node_children[decision_data]
            return current_node.get_class()
        except KeyError:
            return self._UNSEEN_COMBINATION

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
        print(self._TOTAL_DATAPOINTS + str(test_results[self._TOTAL_DATAPOINTS]))
        print(self._NUMBER_CLASSIFIED_CORRECTLY + str(test_results[self._NUMBER_CLASSIFIED_CORRECTLY]))
        print(self._NUMBER_CLASSIFIED_INCORRECTLY + str(test_results[self._NUMBER_CLASSIFIED_INCORRECTLY]))
        print(self._UNSEEN_COMBINATION + str(test_results[self._UNSEEN_COMBINATION]))
        print(self._ACCURACY + str(test_results[self._ACCURACY]))


if __name__ == "__main__":
    testDT = ID3DecisionTree()

    # contains 8 "unseen" feature combinations
    testDT.train("../generated_sets/2017-01-13T13.55.08.756917_train_4911.csv")
    testDT.test("../generated_sets/2017-01-13T13.55.08.756917_test_4911.csv")

    # 1.0 accuracy
    # testDT.train("../generated_sets/2017-02-07T09.46.35.650308_train_9389.csv")
    # testDT.test("../generated_sets/2017-02-07T09.46.35.650308_test_9389.csv")

    # again, 1.0 accuracy
    # testDT.train("../generated_sets/2017-02-07T10.31.45.714731_train_8176.csv")
    # testDT.test("../generated_sets/2017-02-07T10.31.45.714731_test_8176.csv")
